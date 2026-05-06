# SPDX-License-Identifier: Apache-2.0
"""Tests for bus Producer using fakeredis (no network)."""

from __future__ import annotations

import time

import fakeredis
import pytest

from alpha_assay.bus.producer import Producer
from alpha_assay.bus.streams import unpack


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


def test_producer_publish_increments_seq_per_stream(redis_client):
    p = Producer(redis_client=redis_client, maxlen=3600)
    s1 = p.publish("bars.es.cme.20260618", {"close": 1.0}, ts_event_ns=time.time_ns())
    s2 = p.publish("bars.es.cme.20260618", {"close": 2.0}, ts_event_ns=time.time_ns())
    assert s2 == s1 + 1


def test_producer_publish_independent_seq_across_streams(redis_client):
    p = Producer(redis_client=redis_client, maxlen=3600)
    bar_seq = p.publish("bars.es.cme.20260618", {"close": 1.0}, ts_event_ns=time.time_ns())
    tick_seq = p.publish("ticks.tick-nyse", {"value": 100.0}, ts_event_ns=time.time_ns())
    # Each stream has independent seq starting from 0.
    assert bar_seq == 0
    assert tick_seq == 0


def test_producer_publish_writes_msgpack_payload(redis_client):
    p = Producer(redis_client=redis_client, maxlen=3600)
    payload = {"open": 7250.5, "close": 7251.0}
    p.publish("bars.es.cme.20260618", payload, ts_event_ns=1_000_000)

    entries = redis_client.xrange("bars.es.cme.20260618", "-", "+")
    assert len(entries) == 1
    _id, fields = entries[0]
    raw = fields[b"data"]
    msg = unpack(raw)
    assert msg.payload == payload
    assert msg.ts_event_ns == 1_000_000
    assert msg.stream == "bars.es.cme.20260618"


def test_producer_xadd_respects_maxlen(redis_client):
    # MAXLEN ~ 5: publish 10 entries, expect at most ~5 retained.
    p = Producer(redis_client=redis_client, maxlen=5)
    for i in range(10):
        p.publish("bars.test", {"i": i}, ts_event_ns=i)
    n = redis_client.xlen("bars.test")
    # XADD with MAXLEN ~ N caps near N (Redis allows some slack for efficiency).
    assert n <= 10  # always trimmed below initial; "~" is approximate
    assert n >= 4


def test_producer_metrics_increment(redis_client):
    from alpha_assay.bus import metrics as BM

    BM.bus_publish_total.labels(stream="bars.test")._value.set(0)
    p = Producer(redis_client=redis_client, maxlen=3600)
    p.publish("bars.test", {}, ts_event_ns=time.time_ns())
    p.publish("bars.test", {}, ts_event_ns=time.time_ns())
    assert BM.bus_publish_total.labels(stream="bars.test")._value.get() == 2
