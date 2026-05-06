# SPDX-License-Identifier: Apache-2.0
"""Tests for bus Consumer using fakeredis."""

from __future__ import annotations

import time
from threading import Thread

import fakeredis
import pytest

from alpha_assay.bus.consumer import Consumer
from alpha_assay.bus.producer import Producer


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


def test_consumer_replay_from_zero_yields_all_messages(redis_client):
    p = Producer(redis_client=redis_client)
    p.publish("bars.test", {"i": 0}, ts_event_ns=0)
    p.publish("bars.test", {"i": 1}, ts_event_ns=1)
    p.publish("bars.test", {"i": 2}, ts_event_ns=2)

    c = Consumer(redis_client=redis_client, stream="bars.test", consumer_id="rec-test", start_id="0")
    received = []
    for msg in c.iter_messages(max_messages=3):
        received.append(msg.payload["i"])
    assert received == [0, 1, 2]


def test_consumer_replay_from_dollar_skips_history(redis_client):
    p = Producer(redis_client=redis_client)
    p.publish("bars.test", {"i": 0}, ts_event_ns=0)
    p.publish("bars.test", {"i": 1}, ts_event_ns=1)

    # Consumer starts AFTER existing history.
    c = Consumer(redis_client=redis_client, stream="bars.test", consumer_id="paper", start_id="$")

    # Publish a new one in a thread; consumer should pick it up.
    def _publish_late():
        time.sleep(0.05)
        p.publish("bars.test", {"i": 99}, ts_event_ns=99)

    t = Thread(target=_publish_late, daemon=True)
    t.start()

    received = []
    for msg in c.iter_messages(max_messages=1, block_ms=500):
        received.append(msg.payload["i"])
    t.join()
    assert received == [99]


def test_consumer_metrics_increment(redis_client):
    from alpha_assay.bus import metrics as BM

    p = Producer(redis_client=redis_client)
    p.publish("bars.test-metrics", {"i": 0}, ts_event_ns=time.time_ns())

    BM.bus_consume_total.labels(stream="bars.test-metrics", consumer="rec-test")._value.set(0)
    c = Consumer(redis_client=redis_client, stream="bars.test-metrics", consumer_id="rec-test", start_id="0")
    list(c.iter_messages(max_messages=1))
    assert BM.bus_consume_total.labels(stream="bars.test-metrics", consumer="rec-test")._value.get() == 1


def test_consumer_invalid_start_id_raises(redis_client):
    with pytest.raises(ValueError, match="start_id"):
        Consumer(redis_client=redis_client, stream="bars.test", consumer_id="x", start_id="bogus")
