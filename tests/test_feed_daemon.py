# tests/test_feed_daemon.py
"""Tests for IBKRFeedDaemon publish loop with a mock IBKRAdapter and fakeredis."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

from alpha_assay.bus.consumer import Consumer
from infra.feed.feed import IBKRFeedDaemon, Subscription


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


def test_daemon_publishes_bars_to_correct_stream(redis_client, tmp_path):
    """One mock bar arriving from the adapter should appear on bars.es.cme.20260618."""

    bar_event = {
        "timestamp": "2026-05-06T13:30:00+00:00",
        "open": 7250.5,
        "high": 7252.5,
        "low": 7246.5,
        "close": 7247.25,
        "volume": 14338,
        "feed": "ES-FUT-20260618",
    }

    async def fake_subscribe_bars(spec, **kw):
        yield bar_event
        # Then block forever.
        while True:
            await asyncio.sleep(3600)

    mock_adapter = MagicMock()
    mock_adapter.connect_async = AsyncMock()
    mock_adapter.is_connected = False
    mock_adapter.subscribe_bars = fake_subscribe_bars

    sub = Subscription(
        kind="bars",
        contract={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"},
    )

    async def _run_briefly():
        daemon = IBKRFeedDaemon(
            adapter=mock_adapter,
            redis_client=redis_client,
            wal_dir=tmp_path / "wal",
        )
        task = asyncio.create_task(daemon.run_subscription(sub))
        # Let it ingest one bar, then stop.
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_briefly())

    # Verify the bar landed on the right stream.
    c = Consumer(redis_client=redis_client, stream="bars.es.cme.20260618", consumer_id="t", start_id="0")
    msgs = list(c.iter_messages(max_messages=1, block_ms=100))
    assert len(msgs) == 1
    assert msgs[0].payload["close"] == 7247.25


def test_daemon_advances_wal_watermark_on_publish(redis_client, tmp_path):
    bar_event = {
        "timestamp": "2026-05-06T13:30:00+00:00",
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 1,
        "feed": "ES-FUT-20260618",
    }

    async def fake_subscribe_bars(spec, **kw):
        yield bar_event
        while True:
            await asyncio.sleep(3600)

    mock_adapter = MagicMock()
    mock_adapter.connect_async = AsyncMock()
    mock_adapter.is_connected = False
    mock_adapter.subscribe_bars = fake_subscribe_bars

    sub = Subscription(
        kind="bars",
        contract={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"},
    )

    wal_dir = tmp_path / "wal"

    async def _run():
        daemon = IBKRFeedDaemon(
            adapter=mock_adapter,
            redis_client=redis_client,
            wal_dir=wal_dir,
        )
        task = asyncio.create_task(daemon.run_subscription(sub))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())

    # After publish, watermark should be >=0 (the seq of the published msg).
    # Per-stream WAL subdirs since the cross-stream contam fix.
    watermarks = list(wal_dir.rglob("feed-*.committed"))
    assert len(watermarks) == 1
    assert int(watermarks[0].read_text().strip()) >= 0
