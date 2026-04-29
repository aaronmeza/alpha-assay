# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Prometheus metrics + HTTP exporter for the alpha_assay engine.

Every metric name in `metrics` is prefixed `alpha_assay_`. The engine
imports this module and calls the counters/gauges/histograms directly;
strategies do not import it (observability stays a sidecar pattern per
spec Section 3).
"""

from alpha_assay.observability.metrics import start_metrics_server

__all__ = ["start_metrics_server"]
