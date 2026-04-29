#!/usr/bin/env python3
"""Produce a synthetic N-day ES-shaped 1-min parquet for tests and spikes.

Usage: python tests/fixtures/make_sample.py --out sample_2d.parquet --days 2
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

CT = pytz.timezone("America/Chicago")


def generate(days: int, start_date: str | None = None) -> pd.DataFrame:
    start_str = start_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = CT.localize(datetime.strptime(start_str, "%Y-%m-%d"))
    # Generate full calendar days of 1-min bars, then mask to RTH.
    # NYSE core hours in CT: 08:30 CT to 14:59 CT inclusive = 390 bars/day.
    rng = pd.date_range(start=start_dt, periods=days * 24 * 60, freq="1min", tz=CT)
    mask = ((rng.hour > 8) | ((rng.hour == 8) & (rng.minute >= 30))) & (rng.hour < 15)
    idx = rng[mask]
    assert len(idx) == days * 390, f"expected {days * 390} RTH bars, got {len(idx)}"
    n = len(idx)
    rng_state = np.random.default_rng(42)  # deterministic for tests

    close = 5000 + np.cumsum(rng_state.normal(0, 2.0, n))
    high = close + rng_state.uniform(0.5, 1.5, n)
    low = close - rng_state.uniform(0.5, 1.5, n)
    open_ = np.r_[close[0], close[:-1]]
    vol = rng_state.integers(50, 500, n)

    tick = np.clip(
        np.cumsum(rng_state.normal(0, 100, n)) * -0.05 + rng_state.normal(0, 200, n),
        -1500,
        1500,
    )
    add = np.clip(np.cumsum(rng_state.normal(0, 20, n)), -2000, 2000)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": open_,
            "ES_high": high,
            "ES_low": low,
            "ES_close": close,
            "ES_volume": vol,
            "TICK": tick,
            "ADD": add,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--start", default=None)
    args = ap.parse_args()

    df = generate(args.days, args.start)
    out_path = args.out
    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
