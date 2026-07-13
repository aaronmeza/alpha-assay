# tests/test_feed_daemon.py
"""Tests for IBKRFeedDaemon publish loop with a mock IBKRAdapter and fakeredis."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

from alpha_assay.bus import metrics as BM
from alpha_assay.bus.consumer import Consumer
from alpha_assay.bus.streams import CURRENT_VERSION, Message
from alpha_assay.bus.streams import pack as pack_msg
from alpha_assay.bus.wal import WALAppender
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


def test_daemon_seeds_live_seq_from_existing_wal(redis_client, tmp_path):
    """Live publish seqs continue after the per-stream day-file max seq."""
    stream = "bars.es.cme.20260618"
    wal_dir = tmp_path / "wal"
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    wal = WALAppender(directory=wal_dir / stream, day=day)
    for seq in range(3):
        msg = Message(
            v=CURRENT_VERSION,
            seq=seq,
            ts_recv_ns=time.time_ns(),
            ts_event_ns=time.time_ns(),
            stream=stream,
            payload={"close": float(seq)},
        )
        wal.append(seq=seq, msg_bytes=pack_msg(msg))
        wal.advance_committed(seq)
    wal.close()

    bar_event = {
        "timestamp": "2026-05-06T13:30:00+00:00",
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 4.0,
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

    async def _run():
        """Run long enough for one live event to publish."""
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

    c = Consumer(redis_client=redis_client, stream=stream, consumer_id="t", start_id="0")
    msgs = list(c.iter_messages(max_messages=1, block_ms=100))
    assert len(msgs) == 1
    assert msgs[0].seq == 3


def test_daemon_drain_counts_and_logs_replayed_records(redis_client, tmp_path, caplog):
    """Restart drain republishes WAL records with one loud count/span log."""
    stream = "bars.es.cme.20260618"
    wal_dir = tmp_path / "wal"
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    wal = WALAppender(directory=wal_dir / stream, day=day)
    event_times = [
        datetime(2026, 5, 6, 13, 30, tzinfo=UTC),
        datetime(2026, 5, 6, 13, 31, tzinfo=UTC),
    ]
    for seq, event_time in enumerate(event_times):
        msg = Message(
            v=CURRENT_VERSION,
            seq=seq,
            ts_recv_ns=time.time_ns(),
            ts_event_ns=int(event_time.timestamp() * 1_000_000_000),
            stream=stream,
            payload={"close": float(seq)},
        )
        wal.append(seq=seq, msg_bytes=pack_msg(msg))
    wal.close()

    async def fake_subscribe_bars(spec, **kw):
        while True:
            await asyncio.sleep(3600)
            if False:
                yield {}

    mock_adapter = MagicMock()
    mock_adapter.connect_async = AsyncMock()
    mock_adapter.is_connected = False
    mock_adapter.subscribe_bars = fake_subscribe_bars

    sub = Subscription(
        kind="bars",
        contract={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"},
    )

    replay_counter = BM.bus_wal_replayed_total.labels(stream=stream)
    before = replay_counter._value.get()

    async def _run():
        """Run long enough for the WAL drain to finish."""
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

    with caplog.at_level(logging.WARNING):
        asyncio.run(_run())

    assert replay_counter._value.get() == before + 2
    assert "replaying 2 WAL records" in caplog.text
    assert stream in caplog.text
