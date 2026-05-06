# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Paper-trial P&L dashboard.

Reads ``trades.parquet`` from ``RUNS_DIR`` (default ``/runs``) and
renders aggregate stats, an equity curve, per-day P&L, and a signal
breakdown over a user-selected date range.

Launch::

    RUNS_DIR=/path/to/runs streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.loaders import filter_by_date_range, load_trades, trade_log_summary
from dashboard.metrics import (
    DISPLAY_TZ,
    compute_aggregate_metrics,
    compute_equity_curve,
    compute_per_day_summary,
    compute_signal_histogram,
)

_DEFAULT_RUNS_DIR = "/runs"
_DEFAULT_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_usd(v: float | None) -> str:
    if v is None:
        return "-"
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


def _fmt_balance(v: float | None) -> str:
    if v is None:
        return "-"
    return f"${v:,.2f}"


def _fmt_ratio(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}"


def _fmt_winrate(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def _resolve_default_range(df: pd.DataFrame) -> tuple[date, date]:
    """Pick a sensible default date range.

    If trades exist, default to the trade-data range capped at the last
    30 days. Otherwise, fall back to today minus 30 days through today.
    """
    today = pd.Timestamp.utcnow().tz_convert(DISPLAY_TZ).date()
    if df is None or df.empty:
        return today - timedelta(days=_DEFAULT_LOOKBACK_DAYS), today

    trade_start = df["timestamp"].min().tz_convert(DISPLAY_TZ).date()
    trade_end = df["timestamp"].max().tz_convert(DISPLAY_TZ).date()
    default_start = max(trade_start, trade_end - timedelta(days=_DEFAULT_LOOKBACK_DAYS))
    return default_start, trade_end


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Paper P&L",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Paper P&L")
st.caption("Live paper-trading session results. Filter by date range to see how the strategy would have performed.")

runs_dir = os.environ.get("RUNS_DIR", _DEFAULT_RUNS_DIR)
trades_df = load_trades(runs_dir)
log_status = trade_log_summary(runs_dir)

# ---------------------------------------------------------------------------
# Sidebar: date range
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Date range")

    default_start, default_end = _resolve_default_range(trades_df)

    start_input = st.date_input("From", value=default_start)
    end_input = st.date_input("To", value=default_end)

    st.divider()
    st.header("Data source")
    st.caption(f"Path: `{log_status['path']}`")
    if log_status["exists"]:
        st.caption(f"Rows: {log_status['n_rows']:,}")
        if log_status["modified_at"] is not None:
            mod_local = log_status["modified_at"].tz_convert(DISPLAY_TZ)
            st.caption(f"Updated: {mod_local.strftime('%Y-%m-%d %H:%M %Z')}")
        size_kb = log_status["size_bytes"] / 1024.0
        st.caption(f"Size: {size_kb:,.1f} KB")
    else:
        st.caption("Trade log not yet created.")

# ---------------------------------------------------------------------------
# Apply filter
# ---------------------------------------------------------------------------

start_ts = pd.Timestamp(start_input, tz=DISPLAY_TZ).tz_convert("UTC") if start_input else None
end_input_inclusive = (end_input + timedelta(days=1)) if end_input else None
end_ts = pd.Timestamp(end_input_inclusive, tz=DISPLAY_TZ).tz_convert("UTC") if end_input_inclusive else None

filtered = filter_by_date_range(trades_df, start_ts, end_ts)
if filtered is None:
    filtered = trades_df

has_trades = filtered is not None and not filtered.empty

# ---------------------------------------------------------------------------
# Empty-state banner (when zero rows in the file)
# ---------------------------------------------------------------------------

if not log_status["exists"]:
    st.info(
        "No trade log found yet. Once the paper-trader fires a signal, "
        "it will write `trades.parquet` to the configured runs directory "
        "and rows will appear here."
    )
elif log_status["n_rows"] == 0:
    st.info("Trade log exists but contains no rows yet.")

# ---------------------------------------------------------------------------
# Aggregate cards
# ---------------------------------------------------------------------------

agg = compute_aggregate_metrics(filtered if has_trades else pd.DataFrame())

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total P&L", _fmt_usd(agg["total_pnl_usd"]) if has_trades else "-")
col2.metric("Trades", str(agg["n_trades"]))
col3.metric("Win Rate", _fmt_winrate(agg["win_rate"]) if has_trades else "-")
col4.metric("Profit Factor", _fmt_ratio(agg["profit_factor"]) if has_trades else "-")
col5.metric("Avg P&L / trade", _fmt_usd(agg["avg_pnl_per_trade"]) if has_trades else "-")
col6.metric("Account Balance", _fmt_balance(agg["current_balance"]) if has_trades else "-")

st.divider()

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

st.subheader("Equity curve")

equity_df = compute_equity_curve(filtered) if has_trades else pd.DataFrame()

fig = go.Figure()
if not equity_df.empty:
    fig.add_trace(
        go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["account_balance_after"],
            mode="lines+markers",
            name="Account balance",
            line=dict(color="#4C8BF5", width=2),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>Balance: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Trade time",
        yaxis_title="Account balance (USD)",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
else:
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="No trades in this range",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
        ],
    )
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Per-day P&L bar chart + table
# ---------------------------------------------------------------------------

st.subheader("Per-day P&L")

day_summaries = compute_per_day_summary(filtered) if has_trades else []

if day_summaries:
    daily_df = pd.DataFrame(day_summaries)
    bar = go.Figure()
    bar.add_trace(
        go.Bar(
            x=daily_df["date"],
            y=daily_df["total_pnl_usd"],
            marker_color=[
                "#2EAA5C" if v >= 0 else "#D04A4A" for v in daily_df["total_pnl_usd"]
            ],
            hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
        )
    )
    bar.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Session date (CT)",
        yaxis_title="Daily P&L (USD)",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
    st.plotly_chart(bar, use_container_width=True)

    table_rows = [
        {
            "Date": s["date"],
            "Trades": s["n_trades"],
            "P&L": _fmt_usd(s["total_pnl_usd"]),
            "Win rate": _fmt_winrate(s["win_rate"]),
            "Best trade": _fmt_usd(s["largest_win_usd"]),
            "Worst trade": _fmt_usd(s["largest_loss_usd"]),
            "Ending balance": _fmt_balance(s["ending_balance"]),
        }
        for s in day_summaries
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
else:
    st.info("No trades in the selected date range.")

st.divider()

# ---------------------------------------------------------------------------
# Signal breakdown
# ---------------------------------------------------------------------------

st.subheader("Signal breakdown")

hist_df = compute_signal_histogram(filtered) if has_trades else pd.DataFrame()

if not hist_df.empty:
    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Bar(
            x=hist_df["signal_type"],
            y=hist_df["n_trades"],
            marker_color="#7B9CFE",
            customdata=hist_df["total_pnl_usd"],
            hovertemplate="<b>%{x}</b><br>Trades: %{y}<br>P&L: $%{customdata:,.2f}<extra></extra>",
        )
    )
    hist_fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Signal type",
        yaxis_title="Number of trades",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    sig_table = hist_df.rename(
        columns={
            "signal_type": "Signal",
            "n_trades": "Trades",
            "total_pnl_usd": "Total P&L",
        }
    )
    sig_table["Total P&L"] = sig_table["Total P&L"].apply(_fmt_usd)
    st.dataframe(sig_table, use_container_width=True, hide_index=True)
else:
    st.info("No signals fired in the selected date range.")

st.divider()

# ---------------------------------------------------------------------------
# Trade log table
# ---------------------------------------------------------------------------

st.subheader("Trades")

if has_trades:
    display_df = filtered.copy()
    display_df["timestamp"] = (
        display_df["timestamp"].dt.tz_convert(DISPLAY_TZ).dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    display_df = display_df.rename(
        columns={
            "timestamp": "Time",
            "signal_type": "Signal",
            "entry_price": "Entry",
            "stop": "Stop",
            "target": "Target",
            "mock_fill_price": "Fill",
            "mock_pnl_dollars": "P&L",
            "account_balance_after": "Balance after",
        }
    )
    display_df = display_df.sort_values("Time", ascending=False).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades to display.")
