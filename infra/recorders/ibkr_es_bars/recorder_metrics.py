# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Prometheus metric definitions for the ES 1-min bar recorder.

Series live in a dedicated module (mirroring the pattern) so the
core ``alpha_assay.observability.metrics`` catalog and the breadth
recorder's metrics module can evolve independently without merge
conflicts. All series are prefixed ``alpha_assay_es_bars_recorder_``
to keep them distinct from the breadth recorder's
``alpha_assay_recorder_*`` series sharing the global default registry.

Counters end in ``_total`` (Prometheus convention).

The recorder service exposes these on its own HTTP port: ``:8002``
container-internal, ``:18002`` host-side (loopback-only).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge

_PREFIX = "alpha_assay_es_bars_recorder_"

bars_received_total = Counter(
    f"{_PREFIX}bars_received_total",
    "ES 1-min bars received from the IBKR adapter, per feed label.",
    labelnames=("feed",),
)

bars_written_total = Counter(
    f"{_PREFIX}bars_written_total",
    "ES 1-min bars flushed to parquet, per feed label.",
    labelnames=("feed",),
)

write_errors_total = Counter(
    f"{_PREFIX}write_errors_total",
    "Parquet write errors by feed label + exception class.",
    labelnames=("feed", "error_class"),
)

last_bar_age_seconds = Gauge(
    f"{_PREFIX}last_bar_age_seconds",
    "Seconds since the most-recently received in-RTH bar, per feed label.",
    labelnames=("feed",),
)

reconnects_total = Counter(
    f"{_PREFIX}reconnects_total",
    "Count of reconnect attempts triggered by adapter disconnect events.",
)
