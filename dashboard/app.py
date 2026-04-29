# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""alpha_assay backtest results dashboard.

Launch::

    RUNS_DIR=/path/to/runs streamlit run dashboard/app.py

The app reads run output directories from RUNS_DIR (default: /runs).
Each directory under RUNS_DIR that contains the standard report layout
is listed in the run picker.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.loaders import (
    discover_runs,
    filter_by_time_range,
    load_per_trade_metrics,
    load_session_metrics,
)
from dashboard.metrics import (
    compute_aggregate_metrics,
    compute_equity_curve,
    compute_per_day_summary,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Backtest Results",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RUNS_DIR = "/runs"
_TRADE_DISPLAY_COLS = [
    "entry_ts",
    "exit_ts",
    "side",
    "entry_price",
    "exit_price",
    "pnl_points",
    "pnl_usd",
    "hold_seconds",
    "exit_reason",
    "mae_points",
    "mfe_points",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_usd(v: Optional[float]) -> str:
    if v is None:
        return "-"
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "-"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.3f}%"


def _fmt_ratio(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}"


def _fmt_winrate(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def _build_time_range(
    preset: str, df: Optional[pd.DataFrame], custom_start=None, custom_end=None
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Translate a preset label into (start, end) UTC timestamps."""
    now = pd.Timestamp.utcnow()

    if preset == "custom":
        start = pd.Timestamp(custom_start, tz="UTC") if custom_start else None
        end = pd.Timestamp(custom_end, tz="UTC") if custom_end else None
        return start, end

    if preset == "all" or df is None or df.empty:
        return None, None

    cutoffs = {
        "1h": now - pd.Timedelta(hours=1),
        "6h": now - pd.Timedelta(hours=6),
        "1d": now - pd.Timedelta(days=1),
        "1w": now - pd.Timedelta(weeks=1),
        "1mo": now - pd.Timedelta(days=30),
        "ytd": pd.Timestamp(f"{now.year}-01-01", tz="UTC"),
    }
    start = cutoffs.get(preset)
    return start, None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

runs_dir = os.environ.get("RUNS_DIR", _DEFAULT_RUNS_DIR)
run_names = discover_runs(runs_dir)

with st.sidebar:
    st.header("Run")

    if not run_names:
        st.warning(f"No runs found in `{runs_dir}`.")
        st.stop()

    selected_run = st.selectbox(
        "Select run",
        run_names,
        index=0,
        help="Directories are sorted newest first by modification time.",
    )

    st.divider()
    st.header("Time range")

    time_preset = st.radio(
        "Preset",
        options=["all", "ytd", "1mo", "1w", "1d", "6h", "1h", "custom"],
        index=0,
        horizontal=False,
    )

    custom_start = None
    custom_end = None
    if time_preset == "custom":
        custom_start = st.date_input("From", value=None)
        custom_end = st.date_input("To", value=None)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

run_dir = Path(runs_dir) / selected_run
trades_df = load_per_trade_metrics(run_dir)
session_meta = load_session_metrics(run_dir)
starting_balance = session_meta.get("starting_balance_usd", 100_000.0) or 100_000.0

# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------

st.title("Backtest Results")
st.caption(f"Run: **{selected_run}**")

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

if trades_df is None or trades_df.empty:
    st.info("No trades recorded for this run. The run may be a liveness check or produced zero fills.")
    if session_meta:
        st.subheader("Run metadata")
        st.json(session_meta)
    st.stop()

# ---------------------------------------------------------------------------
# Apply time filter
# ---------------------------------------------------------------------------

start_ts, end_ts = _build_time_range(time_preset, trades_df, custom_start, custom_end)
filtered_df = filter_by_time_range(trades_df, start_ts, end_ts, ts_col="entry_ts")

if filtered_df.empty:
    st.warning("No trades in the selected time range.")
    st.stop()

# ---------------------------------------------------------------------------
# Aggregate header
# ---------------------------------------------------------------------------

agg = compute_aggregate_metrics(filtered_df, starting_balance_usd=starting_balance)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total PnL", _fmt_usd(agg["total_pnl_usd"]))
col2.metric("PnL %", _fmt_pct(agg["total_pnl_pct"]))
col3.metric("Trades", str(agg["n_trades"]))
col4.metric("Win Rate", _fmt_winrate(agg["win_rate"]))
col5.metric("Profit Factor", _fmt_ratio(agg["profit_factor"]))
col6.metric("Sharpe", _fmt_ratio(agg["sharpe"]))

st.divider()

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

st.subheader("Equity curve")

equity_df = compute_equity_curve(filtered_df)

if not equity_df.empty:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_df["entry_ts"],
            y=equity_df["cumulative_pnl_usd"],
            mode="lines+markers",
            name="Cumulative PnL",
            line=dict(color="#4C8BF5", width=2),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Entry timestamp",
        yaxis_title="Cumulative PnL (USD)",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data for equity curve.")

st.divider()

# ---------------------------------------------------------------------------
# Per-day summary cards + expandable trade tables
# ---------------------------------------------------------------------------

st.subheader("Per-session breakdown")

day_summaries = compute_per_day_summary(filtered_df, starting_balance_usd=starting_balance)

if not day_summaries:
    st.info("No sessions in the selected time range.")
else:
    for summary in day_summaries:
        date_str = summary["date"]
        pnl_label = (
            f"{_fmt_usd(summary['total_pnl_usd'])}  ({_fmt_pct(summary['total_pnl_pct'])})"
        )
        trades_label = (
            f"{summary['n_trades']} trades"
            + (
                f"  ({summary['n_long']}L / {summary['n_short']}S)"
                if summary["n_long"] is not None
                else ""
            )
        )

        # Card row
        with st.container(border=True):
            card_cols = st.columns([2, 2, 2, 2, 2, 2])
            card_cols[0].markdown(f"**{date_str}**")
            card_cols[1].markdown(trades_label)
            card_cols[2].markdown(pnl_label)
            card_cols[3].markdown(f"Win rate: {_fmt_winrate(summary['win_rate'])}")
            card_cols[4].markdown(
                f"Best: {_fmt_usd(summary['largest_win_usd'])}  |  Worst: {_fmt_usd(summary['largest_loss_usd'])}"
            )
            avg_hold = summary["avg_hold_seconds"]
            card_cols[5].markdown(
                f"Avg hold: {int(avg_hold)}s" if avg_hold is not None else "Avg hold: -"
            )

        # Expandable trade table for this day
        day_df = filtered_df[
            filtered_df["entry_ts"].dt.tz_convert("America/Chicago").dt.date.astype(str)
            == date_str
        ]

        display_cols = [c for c in _TRADE_DISPLAY_COLS if c in day_df.columns]
        display_df = day_df[display_cols].copy()

        # Format timestamps for readability
        for ts_col in ("entry_ts", "exit_ts"):
            if ts_col in display_df.columns:
                display_df[ts_col] = (
                    display_df[ts_col]
                    .dt.tz_convert("America/Chicago")
                    .dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                )

        with st.expander(f"Trades for {date_str} ({len(day_df)} rows)", expanded=False):
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

st.divider()

# ---------------------------------------------------------------------------
# Footer: run metadata
# ---------------------------------------------------------------------------

if session_meta:
    with st.expander("Run metadata (session_metrics.json)", expanded=False):
        st.json(session_meta)
