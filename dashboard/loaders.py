# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Data loading helpers for the backtest results dashboard.

Reads parquet, CSV, and JSON run artifacts from a directory layout::

    <runs_dir>/
        <run-name>/
            report/
                per_trade_metrics.csv
                session_metrics.json
            backtest/
                trades.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def discover_runs(runs_dir: str | Path) -> list[str]:
    """Return run names (directory basenames) sorted newest-first.

    A directory qualifies as a run if it contains at least one of the
    standard report files.  Directories that match none of the expected
    paths are silently skipped.
    """
    root = Path(runs_dir)
    if not root.is_dir():
        return []

    runs: list[tuple[float, str]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        has_report = (
            (entry / "report" / "per_trade_metrics.csv").exists()
            or (entry / "report" / "session_metrics.json").exists()
            or (entry / "backtest" / "trades.csv").exists()
        )
        if has_report:
            mtime = entry.stat().st_mtime
            runs.append((mtime, entry.name))

    runs.sort(reverse=True)
    return [name for _, name in runs]


def describe_run(run_dir: Path) -> str:
    """Return a human-readable label for a run directory.

    Format: ``<name> (<n> trades, <start> -> <end>)`` when trade data is
    available, otherwise just the directory name.
    """
    name = run_dir.name
    df = load_per_trade_metrics(run_dir)
    if df is None or df.empty or "entry_ts" not in df.columns:
        return name

    n = len(df)
    try:
        start = df["entry_ts"].min().tz_convert("UTC").strftime("%Y-%m-%d")
        end = df["entry_ts"].max().tz_convert("UTC").strftime("%Y-%m-%d")
        return f"{name} ({n} trades, {start} -> {end})"
    except Exception:
        return f"{name} ({n} trades)"


def load_all_runs(runs_dir: str | Path) -> Optional[pd.DataFrame]:
    """Load and concatenate per-trade rows across all runs under *runs_dir*.

    Returns a combined DataFrame with an added ``run_name`` column so
    individual sessions can still be identified.  Returns None when no
    run directories with trade data are found.
    """
    root = Path(runs_dir)
    run_names = discover_runs(root)
    frames: list[pd.DataFrame] = []
    for name in run_names:
        df = load_per_trade_metrics(root / name)
        if df is not None and not df.empty:
            df = df.copy()
            df["run_name"] = name
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    if "entry_ts" in combined.columns:
        combined = combined.sort_values("entry_ts").reset_index(drop=True)
    return combined


def load_per_trade_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load per-trade metrics CSV.  Returns None if the file is absent."""
    path = run_dir / "report" / "per_trade_metrics.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Parse timestamp columns as UTC-aware datetimes.
    for col in ("entry_ts", "exit_ts", "entry_signal_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Ensure numeric columns are the right dtype.
    numeric_cols = [
        "pnl_points",
        "pnl_usd",
        "hold_seconds",
        "entry_price",
        "exit_price",
        "quantity",
        "mae_points",
        "mfe_points",
        "fill_latency_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_session_metrics(run_dir: Path) -> dict:
    """Load session-level metrics JSON.  Returns empty dict if absent."""
    path = run_dir / "report" / "session_metrics.json"
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def filter_by_time_range(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    ts_col: str = "entry_ts",
) -> pd.DataFrame:
    """Return rows where *ts_col* falls within [start, end] (inclusive).

    Either bound may be None (meaning no bound on that side).
    """
    if df is None or df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df[ts_col] >= start
    if end is not None:
        mask &= df[ts_col] <= end
    return df[mask].copy()
