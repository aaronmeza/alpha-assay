# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Aggregate metric computations for the backtest results dashboard.

All functions accept a per-trade DataFrame (as returned by
``loaders.load_per_trade_metrics``) and return plain Python scalars or
dicts so the Streamlit UI can render them without further processing.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def compute_aggregate_metrics(
    df: pd.DataFrame,
    starting_balance_usd: float = 100_000.0,
) -> dict[str, Any]:
    """Compute headline aggregate metrics over *df*.

    Returns a dict with keys:
      - total_pnl_usd
      - total_pnl_pct
      - n_trades
      - win_rate      (None if no trades)
      - profit_factor (None if no losses)
      - sharpe        (None if < 2 trades)
      - max_drawdown_usd
    """
    empty: dict[str, Any] = {
        "total_pnl_usd": 0.0,
        "total_pnl_pct": 0.0,
        "n_trades": 0,
        "win_rate": None,
        "profit_factor": None,
        "sharpe": None,
        "max_drawdown_usd": 0.0,
    }

    if df is None or df.empty or "pnl_usd" not in df.columns:
        return empty

    pnl = df["pnl_usd"].dropna()
    if pnl.empty:
        return empty

    total_pnl = float(pnl.sum())
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = len(wins) / n if n > 0 else None

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    # Annualised Sharpe on daily PnL series (250 trading days).
    sharpe: float | None = None
    if n >= 2:
        std = float(pnl.std(ddof=1))
        if std > 0:
            sharpe = round((float(pnl.mean()) / std) * math.sqrt(250), 3)

    # Max drawdown on running cumulative PnL.
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())

    return {
        "total_pnl_usd": round(total_pnl, 2),
        "total_pnl_pct": (round((total_pnl / starting_balance_usd) * 100, 4) if starting_balance_usd else None),
        "n_trades": n,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "sharpe": sharpe,
        "max_drawdown_usd": round(max_dd, 2),
    }


def compute_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns [entry_ts, cumulative_pnl_usd].

    Sorted by entry_ts ascending so Plotly can render a line chart
    directly.  Returns an empty DataFrame if input is missing.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["entry_ts", "cumulative_pnl_usd"])
    if "entry_ts" not in df.columns or "pnl_usd" not in df.columns:
        return pd.DataFrame(columns=["entry_ts", "cumulative_pnl_usd"])

    sorted_df = df.sort_values("entry_ts").reset_index(drop=True)
    sorted_df = sorted_df[["entry_ts", "pnl_usd"]].dropna()
    sorted_df["cumulative_pnl_usd"] = sorted_df["pnl_usd"].cumsum()
    return sorted_df[["entry_ts", "cumulative_pnl_usd"]]


def compute_per_day_summary(
    df: pd.DataFrame,
    starting_balance_usd: float = 100_000.0,
) -> list[dict[str, Any]]:
    """Return one summary dict per session date, sorted ascending.

    Each dict contains:
      date, n_trades, n_long, n_short, total_pnl_usd, total_pnl_pct,
      win_rate, largest_win_usd, largest_loss_usd, avg_hold_seconds
    """
    if df is None or df.empty or "entry_ts" not in df.columns:
        return []

    working = df.copy()
    working["_date"] = working["entry_ts"].dt.tz_convert("America/Chicago").dt.date

    summaries = []
    for date, group in working.groupby("_date"):
        pnl = group["pnl_usd"].dropna()
        n = len(group)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        total_pnl = float(pnl.sum())
        summaries.append(
            {
                "date": str(date),
                "n_trades": n,
                "n_long": int((group["side"] == "long").sum()) if "side" in group.columns else None,
                "n_short": (int((group["side"] == "short").sum()) if "side" in group.columns else None),
                "total_pnl_usd": round(total_pnl, 2),
                "total_pnl_pct": (round((total_pnl / starting_balance_usd) * 100, 4) if starting_balance_usd else None),
                "win_rate": round(len(wins) / n, 4) if n > 0 else None,
                "largest_win_usd": round(float(wins.max()), 2) if not wins.empty else None,
                "largest_loss_usd": round(float(losses.min()), 2) if not losses.empty else None,
                "avg_hold_seconds": (
                    round(float(group["hold_seconds"].mean()), 1) if "hold_seconds" in group.columns else None
                ),
            }
        )

    summaries.sort(key=lambda x: x["date"])
    return summaries
