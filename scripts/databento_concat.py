#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Concat Databento bulk-pull shards into a single parquet.

Reads every shard under
    {out_dir}/{symbol}/{schema}/{YYYY}/{YYYY-MM-DD}.parquet
and writes one merged parquet at
    {out_dir}/{symbol}_{schema}_merged.parquet
(or a path supplied via --merged-path).

Asserts the merged result is monotonic-increasing on `ts_event` or
`timestamp` — sorts first if the on-disk order is out of sequence.

Used by to feed a single-file real-data backtest into the
alpha_assay engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concat Databento daily shards")
    parser.add_argument("--symbol", required=True, help="sanitized symbol (e.g. ES_FUT)")
    parser.add_argument("--schema", required=True, help="e.g. ohlcv-1m or mbp-1")
    parser.add_argument(
        "--out-dir",
        default="data/databento",
        help="Root directory holding per-symbol shard trees",
    )
    parser.add_argument(
        "--merged-path",
        default=None,
        help="Override merged output path (default: {out_dir}/{symbol}_{schema}_merged.parquet)",
    )
    return parser.parse_args(argv)


def _ts_col(df: pd.DataFrame) -> str:
    for candidate in ("ts_event", "timestamp"):
        if candidate in df.columns:
            return candidate
    # last resort: use the index name
    if df.index.name in ("ts_event", "timestamp"):
        return df.index.name
    raise ValueError("no ts_event/timestamp column found in shard")


def main(argv: list[str] | None = None) -> None:
    args = _parse_argv(argv)

    out_dir = Path(args.out_dir)
    shard_root = out_dir / args.symbol / args.schema
    shards = sorted(shard_root.rglob("*.parquet"))

    if not shards:
        print(
            f"WARNING: no shards found under {shard_root}; nothing to merge.",
            file=sys.stderr,
        )
        return

    frames = [pd.read_parquet(p) for p in shards]
    merged = pd.concat(frames, ignore_index=True)

    ts = _ts_col(merged)
    merged[ts] = pd.to_datetime(merged[ts], utc=True)
    if not merged[ts].is_monotonic_increasing:
        merged = merged.sort_values(ts, kind="mergesort").reset_index(drop=True)

    assert merged[ts].is_monotonic_increasing, "merged parquet is not monotonic after sort"

    if args.merged_path:
        merged_path = Path(args.merged_path)
    else:
        merged_path = out_dir / f"{args.symbol}_{args.schema}_merged.parquet"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_path)

    print(
        f"merged {len(shards)} shards -> {merged_path} "
        f"({len(merged)} rows, {merged_path.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
