# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for the alpha-assay bus."""

from prometheus_client import Counter, Gauge, Histogram

_PREFIX = "alpha_assay_"

# Producer-side: ts_recv_ns -> ts_publish (XADD ack).
bus_publish_lag_seconds = Histogram(
    f"{_PREFIX}bus_publish_lag_seconds",
    "Producer-side publish lag from event-receive to Redis XADD ack.",
    labelnames=("stream",),
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Consumer-side: ts_event_ns -> consumer-receive.
bus_consume_lag_seconds = Histogram(
    f"{_PREFIX}bus_consume_lag_seconds",
    "Consumer-side lag from claimed event time to receive.",
    labelnames=("stream", "consumer"),
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0),
)

bus_publish_total = Counter(
    f"{_PREFIX}bus_publish_total",
    "Total messages published per stream.",
    labelnames=("stream",),
)

bus_consume_total = Counter(
    f"{_PREFIX}bus_consume_total",
    "Total messages consumed per stream + consumer.",
    labelnames=("stream", "consumer"),
)

bus_wal_pending = Gauge(
    f"{_PREFIX}bus_wal_pending",
    "Producer-side WAL records appended but not yet confirmed published.",
)

bus_wal_fsync_seconds = Histogram(
    f"{_PREFIX}bus_wal_fsync_seconds",
    "Producer-side WAL fsync duration per append.",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)

bus_wal_replayed_total = Counter(
    f"{_PREFIX}bus_wal_replayed_total",
    "Records republished from the WAL on producer restart (drain).",
    labelnames=("stream",),
)

bus_redis_degraded = Gauge(
    f"{_PREFIX}bus_redis_degraded",
    "1 if producer cannot reach Redis, else 0.",
)

feed_lock_state = Gauge(
    f"{_PREFIX}feed_lock_state",
    "1 if this ibkr-feed instance currently holds the contract lock.",
    labelnames=("contract",),
)
