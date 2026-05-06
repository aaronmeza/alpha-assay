# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Aggregate metrics for the paper-trial dashboard.

All functions accept a trade-record DataFrame as returned by
``loaders.load_trades`` (schema: timestamp, signal_type, entry_price,
stop, target, mock_fill_price, mock_pnl_dollars,
account_balance_after).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

DISPLAY_TZ = "America/Chicago"


def compute_aggregate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Headline aggregate metrics over *df*.

    Keys: total_pnl_usd, n_trades, win_rate (None when no trades),
    profit_factor (None when no losses), sharpe (None for <2 trades),
    max_drawdown_usd, current_balance (latest account_balance_after,
    None when no trades), avg_pnl_per_trade.
    """
    empty: dict[str, Any] = {
        "total_pnl_usd": 0.0,
        "n_trades": 0,
        "win_rate": None,
        "profit_factor": None,
        "sharpe": None,
        "max_drawdown_usd": 0.0,
        "current_balance": None,
        "avg_pnl_per_trade": None,
    }

    if df is None or df.empty or "mock_pnl_dollars" not in df.columns:
        return empty

    pnl = df["mock_pnl_dollars"].dropna()
    if pnl.empty:
        return empty

    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total = float(pnl.sum())

    win_rate = len(wins) / n if n > 0 else None
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    sharpe: float | None = None
    if n >= 2:
        std = float(pnl.std(ddof=1))
        if std > 0:
            sharpe = round((float(pnl.mean()) / std) * math.sqrt(250), 3)

    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    max_dd = float(drawdown.min())

    current_balance: float | None = None
    if "account_balance_after" in df.columns:
        bal_series = df.sort_values("timestamp")["account_balance_after"].dropna()
        if not bal_series.empty:
            current_balance = float(bal_series.iloc[-1])

    return {
        "total_pnl_usd": round(total, 2),
        "n_trades": n,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "sharpe": sharpe,
        "max_drawdown_usd": round(max_dd, 2),
        "current_balance": round(current_balance, 2) if current_balance is not None else None,
        "avg_pnl_per_trade": round(total / n, 2) if n > 0 else None,
    }


def compute_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``[timestamp, account_balance_after]`` sorted ascending.

    Uses the recorded balance directly rather than re-cumulating P&L,
    so deposits/adjustments (when added later) flow through honestly.
    """
    cols = ["timestamp", "account_balance_after"]
    if df is None or df.empty or not all(c in df.columns for c in cols):
        return pd.DataFrame(columns=cols)

    out = df[cols].dropna().sort_values("timestamp").reset_index(drop=True)
    return out


def compute_per_day_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-session-date summary, sorted ascending by date.

    Each entry: date, n_trades, total_pnl_usd, win_rate,
    largest_win_usd, largest_loss_usd, ending_balance.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return []

    work = df.copy()
    work["_date"] = work["timestamp"].dt.tz_convert(DISPLAY_TZ).dt.date

    out: list[dict[str, Any]] = []
    for date, group in work.groupby("_date"):
        pnl = group["mock_pnl_dollars"].dropna()
        n = len(group)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        ending_bal = None
        if "account_balance_after" in group.columns:
            bal = group.sort_values("timestamp")["account_balance_after"].dropna()
            if not bal.empty:
                ending_bal = round(float(bal.iloc[-1]), 2)
        out.append(
            {
                "date": str(date),
                "n_trades": n,
                "total_pnl_usd": round(float(pnl.sum()), 2),
                "win_rate": round(len(wins) / n, 4) if n > 0 else None,
                "largest_win_usd": round(float(wins.max()), 2) if not wins.empty else None,
                "largest_loss_usd": round(float(losses.min()), 2) if not losses.empty else None,
                "ending_balance": ending_bal,
            }
        )

    out.sort(key=lambda x: x["date"])
    return out


def compute_signal_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Count of trades per ``signal_type``, with summed P&L per type.

    Columns: signal_type, n_trades, total_pnl_usd. Sorted descending by
    n_trades.
    """
    cols = ["signal_type", "n_trades", "total_pnl_usd"]
    if df is None or df.empty or "signal_type" not in df.columns:
        return pd.DataFrame(columns=cols)

    grouped = (
        df.groupby("signal_type", dropna=False)
        .agg(
            n_trades=("signal_type", "size"),
            total_pnl_usd=("mock_pnl_dollars", "sum"),
        )
        .reset_index()
    )
    grouped["total_pnl_usd"] = grouped["total_pnl_usd"].round(2)
    grouped = grouped.sort_values("n_trades", ascending=False).reset_index(drop=True)
    return grouped[cols]
