# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Databento parquet loader.

Scope: futures 1-min OHLCV only (Databento schema `ohlcv-1m`).
Databento does not carry NYSE breadth indices (TICK-NYSE, AD-NYSE) -
those are recorded live from IBKR's free CME entitlement (see the
recorders under `infra/recorders/`).

Canonical schema after load:
    DatetimeIndex name='timestamp' tz='America/Chicago', columns:
    [open, high, low, close, volume]

Accepted input column variants: Databento products use either
`timestamp` or `ts_event` for the index; the adapter accepts both.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DatabentoSchemaError(ValueError):
    """Raised when a parquet file fails canonical-schema validation."""


_REQUIRED = ("open", "high", "low", "close", "volume")


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    ts_col = None
    for candidate in ("timestamp", "ts_event"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise DatabentoSchemaError("parquet must have a 'timestamp' or 'ts_event' column")

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise DatabentoSchemaError(f"parquet missing required columns: {missing}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert("America/Chicago")
    df = df.set_index(ts_col).rename_axis("timestamp")

    if not df.index.is_monotonic_increasing:
        raise DatabentoSchemaError("timestamps are not monotonic increasing")

    if (df["high"] < df["low"]).any():
        raise DatabentoSchemaError("high is less than low for at least one bar")

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise DatabentoSchemaError("high is less than max(open, close) for at least one bar")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise DatabentoSchemaError("low is greater than min(open, close) for at least one bar")

    return df[list(_REQUIRED)]
