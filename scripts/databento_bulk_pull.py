#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resume-safe Databento bulk pull for ES historical data.

Extends the smoke-pull pattern to multi-year, multi-schema bulk
pulls. Designed to run overnight on a paid Databento subscription:

- Per-day + per-schema parquet shards at
  {out_dir}/{symbol_sanitized}/{schema}/{YYYY}/{YYYY-MM-DD}.parquet
- Idempotent manifest.json with one entry per (date, schema) pair
- Weekend skip (Sat/Sun — harmless empty days for weekdays with no data)
- --force flag re-pulls existing days
- --estimate-only uses metadata.get_cost (v0.57+) to project total USD

Usage:
    DATABENTO_API_KEY=db-XXXXX python scripts/databento_bulk_pull.py \\
        --symbol ES.FUT \\
        --start-date 2023-01-01 --end-date 2026-04-23 \\
        --schemas ohlcv-1m,mbp-1

    # preview cost only:
    python scripts/databento_bulk_pull.py \\
        --symbol ES.FUT --start-date 2023-01-01 --end-date 2026-04-23 \\
        --estimate-only

Note on metadata.get_cost: signature in databento-python v0.57+ is
`get_cost(dataset, symbols, schema, start, end)` returning USD as float.
If the installed version differs, update _estimate_schema_cost below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path


def sanitize_symbol(symbol: str) -> str:
    """Replace filesystem-unsafe chars for directory names.

    ES.FUT -> ES_FUT, ES.c.0 -> ES_c_0, ESM6 -> ESM6
    """
    return symbol.replace(".", "_").replace("/", "_")


def iter_trading_days(start_date: str, end_date: str) -> Iterator[str]:
    """Yield YYYY-MM-DD strings for each weekday in [start, end] inclusive.

    Skips Saturdays (weekday 5) and Sundays (weekday 6). Does not attempt
    holiday filtering — a weekday with no market data is a harmless empty
    parquet.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        return
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            yield cur.isoformat()
        cur += timedelta(days=1)


def _shard_path(out_dir: Path, symbol_dir: str, schema: str, day: str) -> Path:
    year = day[:4]
    return out_dir / symbol_dir / schema / year / f"{day}.parquet"


def _manifest_path(out_dir: Path, symbol_dir: str) -> Path:
    return out_dir / symbol_dir / "manifest.json"


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _save_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True))


def _upsert_manifest(entries: list[dict], new_entry: dict) -> list[dict]:
    """Replace any existing entry matching (date, schema); else append."""
    key = (new_entry["date"], new_entry["schema"])
    filtered = [e for e in entries if (e.get("date"), e.get("schema")) != key]
    filtered.append(new_entry)
    return filtered


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Databento resume-safe bulk pull")
    parser.add_argument("--symbol", required=True, help="e.g. ES.FUT or ESM6")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument(
        "--dataset",
        default="GLBX.MDP3",
        help="Databento dataset; default GLBX.MDP3 (CME Globex)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/databento",
        help="Output directory root (gitignored)",
    )
    parser.add_argument(
        "--schemas",
        default="ohlcv-1m,mbp-1",
        help="Comma-separated schemas; default ohlcv-1m,mbp-1",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull days even if the parquet shard already exists",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Call metadata.get_cost per schema and print projected USD; no data pulled",
    )
    return parser.parse_args(argv)


def _estimate_schema_cost(client, dataset: str, symbol: str, schema: str, start: str, end: str) -> float:
    """Call Databento's metadata.get_cost for one (schema, range).

    Signature per databento-python v0.57+:
        client.metadata.get_cost(
            dataset, symbols, schema, start, end,
        ) -> float (USD)
    """
    return float(
        client.metadata.get_cost(
            dataset=dataset,
            symbols=symbol,
            schema=schema,
            start=start,
            end=end,
        )
    )


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

    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    symbol_dir = sanitize_symbol(args.symbol)
    out_dir = Path(args.out_dir)

    if args.estimate_only:
        total = 0.0
        range_start = f"{args.start_date}T00:00:00Z"
        range_end = f"{args.end_date}T23:59:59Z"
        for schema in schemas:
            cost = _estimate_schema_cost(
                client,
                args.dataset,
                args.symbol,
                schema,
                range_start,
                range_end,
            )
            print(f"[estimate] schema={schema} cost_usd={cost:.4f}")
            total += cost
        print(f"[estimate] TOTAL USD = {total:.4f}")
        return

    days = list(iter_trading_days(args.start_date, args.end_date))
    manifest_path = _manifest_path(out_dir, symbol_dir)
    manifest = _load_manifest(manifest_path)

    t0 = time.monotonic()
    pulled = 0
    skipped = 0
    total_rows = 0
    total_bytes = 0

    for day in days:
        for schema in schemas:
            shard = _shard_path(out_dir, symbol_dir, schema, day)
            if shard.exists() and not args.force:
                skipped += 1
                continue

            shard.parent.mkdir(parents=True, exist_ok=True)

            start = f"{day}T00:00:00Z"
            end = f"{day}T23:59:59Z"

            call_t0 = time.monotonic()
            data = client.timeseries.get_range(
                dataset=args.dataset,
                symbols=args.symbol,
                schema=schema,
                start=start,
                end=end,
            )
            df = data.to_df()
            call_duration = time.monotonic() - call_t0

            if df is None:
                rows = 0
                df_bytes = 0
            else:
                df.to_parquet(shard)
                rows = len(df)
                df_bytes = shard.stat().st_size

            pulled += 1
            total_rows += rows
            total_bytes += df_bytes

            manifest = _upsert_manifest(
                manifest,
                {
                    "date": day,
                    "schema": schema,
                    "rows": int(rows),
                    "bytes": int(df_bytes),
                    "pulled_at": datetime.now(UTC).isoformat(),
                    "api_duration_s": round(call_duration, 3),
                },
            )
            _save_manifest(manifest_path, manifest)

            cumulative_mb = total_bytes / (1024 * 1024)
            print(
                f"[{day}] [{schema}] rows={rows} " f"duration={call_duration:.2f}s cumulative={cumulative_mb:.2f} MB",
                file=sys.stderr,
            )

    wall = time.monotonic() - t0
    print(f"summary: pulled={pulled} skipped={skipped} rows={total_rows} " f"bytes={total_bytes} wall_s={wall:.2f}")


if __name__ == "__main__":
    main()
