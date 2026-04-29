# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Joined ES OHLCV + NYSE breadth loader.

Wave-2 entrypoint for breadth-aware backtests. Reads the canonical ES
1-min parquet (Databento adapter validates schema, clamps OHLC, asserts
monotonic timestamps), then inner-joins per-minute TICK and AD bars
sourced from the IBKR live recorder shards . Returns the
canonical strategy-input frame:

    DatetimeIndex name='timestamp' tz='America/Chicago', columns:
    [open, high, low, close, volume, TICK, ADD]

The recorder shards (`infra/recorders/`, locked) write 1-min bars in
the shape:

    [timestamp (UTC), open, high, low, close, n_ticks, symbol]

The `close` column is the per-minute breadth value (last print in the
bar window). That is what breadth-aware strategies
read via `data["TICK"]` / `data["ADD"]`.

If only one of `tick_path` / `ad_path` is given, the joined frame omits
the absent column. breadth-aware strategies needing both must pass
both paths; the CLI enforces that pairing at the click layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_assay.data.databento_adapter import load_parquet


class BreadthSchemaError(ValueError):
    """Raised when a breadth shard fails schema validation."""


_BREADTH_REQUIRED = ("timestamp", "close")


def _load_breadth_close(path: str | Path, out_col: str) -> pd.DataFrame:
    """Load a recorder breadth shard and return a DataFrame indexed by
    timestamp (America/Chicago) with a single column `out_col` holding
    the per-minute close value.
    """
    df = pd.read_parquet(path)
    missing = [c for c in _BREADTH_REQUIRED if c not in df.columns]
    if missing:
        raise BreadthSchemaError(
            f"breadth parquet at {path!s} missing required columns: {missing}; "
            f"saw {list(df.columns)}"
        )
    if df.empty:
        raise BreadthSchemaError(f"breadth parquet at {path!s} has zero rows")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/Chicago"),
            out_col: df["close"].astype(float),
        }
    )
    out = out.set_index("timestamp").sort_index()
    # Defensive de-dup: recorder shards have a 1-min unique-key invariant
    # but we keep the contract loud so a corrupt shard fails here, not
    # silently in the inner-join.
    if out.index.duplicated().any():
        raise BreadthSchemaError(
            f"breadth parquet at {path!s} has duplicate timestamps after tz convert"
        )
    return out


def load_es_with_breadth(
    es_path: str | Path,
    tick_path: str | Path | None = None,
    ad_path: str | Path | None = None,
) -> pd.DataFrame:
    """Inner-join ES OHLCV with per-minute TICK and AD breadth values.

    Returns a DatetimeIndex frame (name='timestamp', tz='America/Chicago')
    with columns `[open, high, low, close, volume]` plus `TICK` and/or
    `ADD` when the corresponding paths are provided.

    The join is INNER on timestamp: any ES bar without matched breadth
    (and vice versa) is dropped. a breadth-aware strategy reads `data["TICK"]` /
    `data["ADD"]` and a NaN there would silently break the z-score path,
    so we'd rather lose a few boundary bars than feed NaNs into the
    strategy.
    """
    es_df = load_parquet(es_path)

    breadth_frames: list[pd.DataFrame] = []
    if tick_path is not None:
        breadth_frames.append(_load_breadth_close(tick_path, out_col="TICK"))
    if ad_path is not None:
        breadth_frames.append(_load_breadth_close(ad_path, out_col="ADD"))

    if not breadth_frames:
        return es_df

    joined = es_df
    for bdf in breadth_frames:
        joined = joined.join(bdf, how="inner")

    if joined.empty:
        raise BreadthSchemaError(
            "joined ES + breadth frame is empty; check that the ES window "
            "overlaps the recorder shard timestamps"
        )
    return joined.sort_index()
