# SPDX-License-Identifier: Apache-2.0
"""Real-Redis end-to-end test for the bus producer/consumer contract.

Opt-in: requires ``RUN_INTEGRATION=1`` and a reachable Redis at
``REDIS_URL`` (default ``redis://localhost:6379/0``). Skipped otherwise,
so it does not run in the default ``pytest`` / CI pass.

Uses a unique per-run stream name that is deleted on teardown, so it is
safe to run against a shared Redis (the deployed ``alphaassay-redis``)
without touching production streams.

On the deploy host (which has no project venv, so run in a throwaway container)::

    docker run --rm --network host -v /home/ameza/personal/alpha-assay:/app -w /app \\
      python:3.11-slim sh -c \\
      'pip install -q -e ".[dev]" && RUN_INTEGRATION=1 REDIS_URL=redis://127.0.0.1:6379/0 \\
         python -m pytest tests/integration/test_bus_e2e.py -v'
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
import redis as redis_pkg

from alpha_assay.bus.consumer import Consumer
from alpha_assay.bus.producer import Producer

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="set RUN_INTEGRATION=1 (and REDIS_URL) to run the real-Redis bus e2e test",
)

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_BURST_N = 100
_BURST_BUDGET_S = 2.0


@pytest.fixture
def redis_client():
    client = redis_pkg.from_url(_REDIS_URL)
    client.ping()
    yield client
    client.close()


@pytest.fixture
def stream(redis_client):
    name = f"bars.test-e2e.{uuid.uuid4().hex[:12]}"
    yield name
    redis_client.delete(name)


def test_publish_consume_roundtrip(redis_client, stream):
    payload = {
        "open": 7250.5,
        "high": 7252.0,
        "low": 7248.25,
        "close": 7251.0,
        "volume": 1234,
        "ts_minute_utc": 1_700_000_000,
    }
    seq = Producer(redis_client=redis_client, maxlen=3600).publish(stream, payload, ts_event_ns=time.time_ns())
    assert seq == 0

    consumer = Consumer(redis_client=redis_client, stream=stream, consumer_id="e2e-test", start_id="0")
    msgs = list(consumer.iter_messages(max_messages=1, block_ms=2000))

    assert len(msgs) == 1
    assert msgs[0].seq == 0
    assert msgs[0].stream == stream
    assert msgs[0].payload == payload


def test_consume_lag_below_threshold_during_burst(redis_client, stream):
    producer = Producer(redis_client=redis_client, maxlen=3600)
    consumer = Consumer(redis_client=redis_client, stream=stream, consumer_id="e2e-test", start_id="0")

    start = time.monotonic()
    for i in range(_BURST_N):
        producer.publish(stream, {"value": float(i)}, ts_event_ns=time.time_ns())
    received = list(consumer.iter_messages(max_messages=_BURST_N, block_ms=5000))
    elapsed = time.monotonic() - start

    assert len(received) == _BURST_N
    assert [m.seq for m in received] == list(range(_BURST_N))  # ordered, no gaps
    assert elapsed < _BURST_BUDGET_S, f"publish+consume of {_BURST_N} msgs took {elapsed:.3f}s (> {_BURST_BUDGET_S}s)"
    max_lag_s = max((time.time_ns() - m.ts_event_ns) / 1e9 for m in received)
    assert max_lag_s < _BURST_BUDGET_S, f"max event->consume lag {max_lag_s:.3f}s (> {_BURST_BUDGET_S}s)"
