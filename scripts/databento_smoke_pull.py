#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""1-day Databento smoke pull for futures 1-min OHLCV bars.

Uses free-trial credits so a small evaluation pull does not require
a paid subscription. Writes to data/databento_smoke/ which is in
.gitignore; DO NOT commit the output parquet.

Usage:
    DATABENTO_API_KEY=db-XXXXX python scripts/databento_smoke_pull.py \
        --symbol ES.FUT --date 2026-04-21
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Databento 1-day smoke pull")
    parser.add_argument("--symbol", required=True, help="e.g. ES.FUT or ESM6")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--dataset",
        default="GLBX.MDP3",
        help="Databento dataset; default GLBX.MDP3 (CME Globex)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/databento_smoke",
        help="Output directory (gitignored)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_argv(argv)

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print(
            "ERROR: DATABENTO_API_KEY not set. Store it in ~/.secrets/databento.env "
            "and `source` the file, per the global CLAUDE.md convention.",
            file=sys.stderr,
        )
        sys.exit(2)

    import databento as db

    client = db.Historical(key=api_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = f"{args.date}T00:00:00Z"
    end = f"{args.date}T23:59:59Z"

    print(f"Pulling {args.symbol} OHLCV 1-min for {args.date} from {args.dataset}")
    data = client.timeseries.get_range(
        dataset=args.dataset,
        symbols=args.symbol,
        schema="ohlcv-1m",
        start=start,
        end=end,
    )
    df = data.to_df()
    if df is None:
        print("No data returned.")
        return

    out = out_dir / f"{args.symbol.replace('.', '_')}_{args.date}_ohlcv-1m.parquet"
    df.to_parquet(out)
    print(f"Wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
