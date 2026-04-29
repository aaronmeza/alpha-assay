# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Prometheus metric definitions for the IBKR breadth recorder .

Kept in a dedicated module so can edit the core
`alpha_assay.observability.metrics` catalog without a merge conflict.
All series prefixed `alpha_assay_` (consistent with the core catalog).
Counters suffixed `_total`.

The recorder service process will expose these on its own HTTP port
(:8001 container-internal, :18001 host-side Tailscale-gated).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge

_PREFIX = "alpha_assay_"

recorder_ticks_received_total = Counter(
    f"{_PREFIX}recorder_ticks_received_total",
    "Breadth ticks received from the IBKR adapter, per symbol.",
    labelnames=("symbol",),
)

recorder_bars_written_total = Counter(
    f"{_PREFIX}recorder_bars_written_total",
    "1-min bars flushed to parquet, per symbol.",
    labelnames=("symbol",),
)

recorder_write_errors_total = Counter(
    f"{_PREFIX}recorder_write_errors_total",
    "Parquet write errors by symbol + exception class.",
    labelnames=("symbol", "error_class"),
)

recorder_session_gap_seconds = Gauge(
    f"{_PREFIX}recorder_session_gap_seconds",
    "Seconds since last tick received within an RTH window, per symbol.",
    labelnames=("symbol",),
)

recorder_reconnects_total = Counter(
    f"{_PREFIX}recorder_reconnects_total",
    "Count of reconnect attempts triggered by adapter disconnect events.",
)
