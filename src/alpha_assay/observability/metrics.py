# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Prometheus metric definitions (spec Section 6).

All series prefixed `alpha_assay_`. Counters suffixed `_total`.
Label keys strictly typed to prevent accidental cardinality explosions.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_PREFIX = "alpha_assay_"

# --- Signal pipeline ---

bars_processed_total = Counter(
    f"{_PREFIX}bars_processed_total",
    "Count of bars processed by the engine, per feed.",
    labelnames=("feed",),
)

signals_generated_total = Counter(
    f"{_PREFIX}signals_generated_total",
    "Count of non-zero signals generated, per strategy, per direction.",
    labelnames=("strategy", "direction"),
)

signals_filtered_total = Counter(
    f"{_PREFIX}signals_filtered_total",
    "Count of signals filtered out before order submission.",
    labelnames=("strategy", "filter_name", "reason"),
)

signals_fired_total = Counter(
    f"{_PREFIX}signals_fired_total",
    "Count of signals that survived filters and produced an order.",
    labelnames=("strategy", "direction"),
)

# --- Orders / fills ---

orders_submitted_total = Counter(
    f"{_PREFIX}orders_submitted_total",
    "Count of orders submitted (bracket parent or children).",
    labelnames=("type",),
)

orders_filled_total = Counter(
    f"{_PREFIX}orders_filled_total",
    "Count of orders resolved by the broker.",
    labelnames=("type", "status"),
)

fill_slippage_points = Histogram(
    f"{_PREFIX}fill_slippage_points",
    "Fill price minus reference price, in instrument points.",
    labelnames=("type",),
)

# --- Trade level ---

trade_pnl_points = Histogram(
    f"{_PREFIX}trade_pnl_points",
    "Realized PnL of each round trip, in instrument points.",
    labelnames=("outcome",),
)
trade_mae_points = Histogram(
    f"{_PREFIX}trade_mae_points",
    "Maximum adverse excursion per trade, in points.",
)
trade_mfe_points = Histogram(
    f"{_PREFIX}trade_mfe_points",
    "Maximum favorable excursion per trade, in points.",
)
trade_duration_seconds = Histogram(
    f"{_PREFIX}trade_duration_seconds",
    "Trade duration from entry fill to exit fill, seconds.",
)
trades_total = Counter(
    f"{_PREFIX}trades_total",
    "Count of completed trades by outcome.",
    labelnames=("outcome",),
)

# --- Equity and risk ---

equity_points = Gauge(f"{_PREFIX}equity_points", "Current equity in instrument points.")
session_pnl_points = Gauge(f"{_PREFIX}session_pnl_points", "Session PnL in points.")
drawdown_points = Gauge(
    f"{_PREFIX}drawdown_points",
    "Current drawdown from session high, in points.",
)
position_contracts = Gauge(
    f"{_PREFIX}position_contracts",
    "Signed contract count: -1 short, 0 flat, +1 long.",
)

# --- Health ---

feed_freshness_seconds = Gauge(
    f"{_PREFIX}feed_freshness_seconds",
    "Seconds since last tick per feed.",
    labelnames=("feed",),
)
ibkr_connected = Gauge(f"{_PREFIX}ibkr_connected", "1 if IBKR is connected, else 0.")
ibkr_connection_events_total = Counter(
    f"{_PREFIX}ibkr_connection_events_total",
    "IBKR lifecycle events: connected, disconnected, error.",
    labelnames=("event",),
)
ibkr_feed_freshness_seconds = Gauge(
    f"{_PREFIX}ibkr_feed_freshness_seconds",
    "Seconds since last IBKR tick or bar per feed (adapter-level, pre-aggregation).",
    labelnames=("feed",),
)
kill_switch_armed = Gauge(f"{_PREFIX}kill_switch_armed", "1 if kill-switch is armed, else 0.")
kill_switch_trips_total = Counter(
    f"{_PREFIX}kill_switch_trips_total",
    "Kill-switch trip events by reason.",
    labelnames=("reason",),
)

# --- Execution mode ---

exec_mode = Gauge(
    f"{_PREFIX}exec_mode",
    "Active execution mode: 1 for the active mode (paper|live), 0 for the other.",
    labelnames=("mode",),
)
live_lock_state = Gauge(
    f"{_PREFIX}live_lock_state",
    "State of each of the three live-mode locks: 1 engaged, 0 not.",
    labelnames=("lock",),
)

# --- Session state ---

in_session = Gauge(f"{_PREFIX}in_session", "1 if inside strategy session window, else 0.")

# --- Runtime ---

signal_eval_seconds = Histogram(
    f"{_PREFIX}signal_eval_seconds",
    "Wall time of BaseStrategy.generate_signals per bar.",
    labelnames=("strategy",),
)
bar_to_order_seconds = Histogram(
    f"{_PREFIX}bar_to_order_seconds",
    "Wall time from bar close to order submission.",
)


def start_metrics_server(port: int = 9200) -> None:
    """Start the Prometheus HTTP exporter on the given port. Safe to
    call once per process. The paper-trader container exposes this on
    :9200; Prometheus on the host scrapes it Tailscale-only.
    """
    start_http_server(port)
