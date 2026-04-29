# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""CSV bar loader.

Expected CSV schema (header row required):
    timestamp, open, high, low, close, volume  (plus optional extras)

Timestamps in ISO 8601; converted to America/Chicago. Data quality checks
enforced on load: monotonic timestamps, no NaN in OHLC, non-negative volume,
high >= max(open, close), low <= min(open, close).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataQualityError(ValueError):
    """Raised when a loaded CSV fails on-load sanity checks."""


REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


def load_csv_bars(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataQualityError(f"CSV missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/Chicago")
    df = df.set_index("timestamp")

    if not df.index.is_monotonic_increasing:
        raise DataQualityError("timestamps are not monotonic increasing")

    ohlc = df[["open", "high", "low", "close"]]
    if ohlc.isna().any().any():
        raise DataQualityError("OHLC contains NaN values")

    if (df["volume"] < 0).any():
        raise DataQualityError("volume contains negative values")

    if (df["high"] < df["low"]).any():
        raise DataQualityError("high is less than low for at least one bar")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise DataQualityError("high is less than max(open, close) for at least one bar")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise DataQualityError("low is greater than min(open, close) for at least one bar")

    return df
