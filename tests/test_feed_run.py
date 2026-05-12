# tests/test_feed_run.py
"""Tests for the ibkr-feed daemon supervisor (infra.feed.run).

Covers the connection watchdog: when IB Gateway drops the connection
(its nightly IBC restart, a network blip) the daemon must NOT hang in
``await queue.get()`` forever. It must notice the loss, mark itself
disconnected, and return a non-zero exit code so the container restart
policy recycles it with a fresh connection.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

from infra.feed.feed import IBKRFeedDaemon, Subscription
from infra.feed.run import _connect_with_retry, _run_subscriptions, _watch_connection


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


# --- _watch_connection ---------------------------------------------------


def test_watch_connection_returns_when_adapter_disconnects():
    adapter = MagicMock()
    adapter.is_connected = True

    async def _run():
        async def _drop_later():
            await asyncio.sleep(0.05)
            adapter.is_connected = False

        asyncio.create_task(_drop_later())
        # If the watchdog never returns this await hangs and the test
        # times out -> failure, which is the regression we are guarding.
        await asyncio.wait_for(_watch_connection(adapter, poll_seconds=0.01), timeout=2.0)

    asyncio.run(_run())


def test_watch_connection_blocks_while_connected():
    adapter = MagicMock()
    adapter.is_connected = True

    async def _run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_watch_connection(adapter, poll_seconds=0.01), timeout=0.2)

    asyncio.run(_run())


# --- _run_subscriptions --------------------------------------------------


def _bars_sub() -> Subscription:
    return Subscription(
        kind="bars",
        contract={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"},
    )


def _adapter_with_blocking_bars() -> MagicMock:
    """Mock adapter whose bar subscription yields one bar then blocks
    forever in ``await`` - the exact shape of a live IBKR subscription
    that has gone quiet because the socket died."""

    async def fake_subscribe_bars(spec, **kw):
        yield {
            "timestamp": "2026-05-06T13:30:00+00:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "feed": "ES-FUT-20260618",
        }
        while True:
            await asyncio.sleep(3600)

    adapter = MagicMock()
    adapter.connect_async = AsyncMock()
    adapter.is_connected = True
    adapter.subscribe_bars = fake_subscribe_bars
    return adapter


def test_run_subscriptions_exits_nonzero_when_connection_lost(redis_client, tmp_path):
    adapter = _adapter_with_blocking_bars()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal")
    stop = asyncio.Event()

    async def _run():
        async def _drop_later():
            await asyncio.sleep(0.1)
            adapter.is_connected = False

        asyncio.create_task(_drop_later())
        return await asyncio.wait_for(
            _run_subscriptions(daemon, adapter, [_bars_sub()], stop, watchdog_poll=0.02),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc != 0
    # The adapter gauge / counters must reflect the loss.
    adapter.disconnect.assert_called()


def test_run_subscriptions_returns_zero_on_stop(redis_client, tmp_path):
    adapter = _adapter_with_blocking_bars()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal")
    stop = asyncio.Event()

    async def _run():
        async def _stop_later():
            await asyncio.sleep(0.1)
            stop.set()

        asyncio.create_task(_stop_later())
        return await asyncio.wait_for(
            _run_subscriptions(daemon, adapter, [_bars_sub()], stop, watchdog_poll=0.02),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc == 0


def test_run_subscriptions_exits_nonzero_on_subscription_error(redis_client, tmp_path):
    async def fake_subscribe_bars(spec, **kw):
        yield {
            "timestamp": "2026-05-06T13:30:00+00:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "feed": "ES-FUT-20260618",
        }
        raise RuntimeError("ib_insync exploded")

    adapter = MagicMock()
    adapter.connect_async = AsyncMock()
    adapter.is_connected = True
    adapter.subscribe_bars = fake_subscribe_bars
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal")
    stop = asyncio.Event()

    async def _run():
        return await asyncio.wait_for(
            _run_subscriptions(daemon, adapter, [_bars_sub()], stop, watchdog_poll=0.02),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc != 0


# --- _connect_with_retry -------------------------------------------------


def test_connect_with_retry_succeeds_after_transient_failures():
    adapter = MagicMock()
    calls = {"n": 0}

    async def flaky_connect():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionRefusedError("gateway not up yet")

    adapter.connect_async = flaky_connect

    asyncio.run(_connect_with_retry(adapter, attempts=5, base_delay=0.001, max_delay=0.01))
    assert calls["n"] == 3


def test_connect_with_retry_raises_after_exhausting_attempts():
    adapter = MagicMock()
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise ConnectionRefusedError("gateway down")

    adapter.connect_async = always_fails

    with pytest.raises(ConnectionRefusedError):
        asyncio.run(_connect_with_retry(adapter, attempts=3, base_delay=0.001, max_delay=0.01))
    assert calls["n"] == 3
