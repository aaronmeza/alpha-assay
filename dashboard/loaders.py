# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Load paper-trial trade records for the dashboard.

The paper-trader writes ``trades.parquet`` to ``RUNS_DIR`` via
``alpha_assay.exec.trade_log.TradeLog``. The schema is the
``TradeRecord`` dataclass:

    timestamp, signal_type, entry_price, stop, target,
    mock_fill_price, mock_pnl_dollars, account_balance_after

All loader functions tolerate a missing or empty parquet file by
returning an empty DataFrame so the UI can render an empty-state
banner without crashing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRADES_FILENAME = "trades.parquet"

_REQUIRED_COLUMNS = (
    "timestamp",
    "signal_type",
    "entry_price",
    "stop",
    "target",
    "mock_fill_price",
    "mock_pnl_dollars",
    "account_balance_after",
)

_NUMERIC_COLUMNS = (
    "entry_price",
    "stop",
    "target",
    "mock_fill_price",
    "mock_pnl_dollars",
    "account_balance_after",
)


def empty_trades_df() -> pd.DataFrame:
    """Return a typed empty DataFrame with the trade-log schema."""
    df = pd.DataFrame({col: pd.Series(dtype="float64") for col in _NUMERIC_COLUMNS})
    df.insert(0, "signal_type", pd.Series(dtype="object"))
    df.insert(0, "timestamp", pd.Series(dtype="datetime64[ns, UTC]"))
    return df


def load_trades(runs_dir: str | Path) -> pd.DataFrame:
    """Load ``trades.parquet`` from *runs_dir*.

    Returns an empty (but typed) DataFrame when the file is absent or
    empty. Timestamps are normalised to UTC; numeric columns are coerced
    to float64. Rows are sorted by timestamp ascending.
    """
    path = Path(runs_dir) / TRADES_FILENAME
    if not path.exists():
        return empty_trades_df()

    df = pd.read_parquet(path)
    if df.empty:
        return empty_trades_df()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def filter_by_date_range(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Return rows where *ts_col* falls within [start, end] (inclusive).

    Either bound may be None. Naive bounds are interpreted as UTC.
    """
    if df is None or df.empty or ts_col not in df.columns:
        return df if df is not None else empty_trades_df()

    mask = pd.Series(True, index=df.index)
    if start is not None:
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        mask &= df[ts_col] >= start
    if end is not None:
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        mask &= df[ts_col] <= end
    return df[mask].copy()


def trade_log_summary(runs_dir: str | Path) -> dict:
    """Lightweight metadata about the trade log file (for status display).

    Returns ``{"path": ..., "exists": bool, "size_bytes": int,
    "modified_at": pd.Timestamp | None, "n_rows": int}``.
    """
    path = Path(runs_dir) / TRADES_FILENAME
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "modified_at": None,
            "n_rows": 0,
        }

    stat = path.stat()
    df = load_trades(runs_dir)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC"),
        "n_rows": len(df),
    }


def required_columns() -> tuple[str, ...]:
    """Return the canonical column ordering for a trade record."""
    return _REQUIRED_COLUMNS
