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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

from infra.feed.feed import FeedManifest, IBKRFeedDaemon, Subscription
from infra.feed.run import (
    _connect_with_retry,
    _pre_open_requalify_loop,
    _resolve_front_months,
    _run_subscriptions,
    _seconds_until_next_pre_open,
    _watch_connection,
)


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


# --- _resolve_front_months --------------------------------------------------


def _es_bars_manifest() -> FeedManifest:
    """A manifest with one ES bars subscription and one breadth ticks subscription."""
    return FeedManifest(
        subscriptions=[
            Subscription(
                kind="bars",
                contract={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20250919"},
            ),
            Subscription(kind="ticks", symbol="ADD"),
        ]
    )


def _breadth_only_manifest() -> FeedManifest:
    """A manifest with no ES bars subscriptions - breadth-only deployment."""
    return FeedManifest(
        subscriptions=[
            Subscription(kind="ticks", symbol="TICK"),
            Subscription(kind="ticks", symbol="ADD"),
        ]
    )


def test_resolve_front_months_override_path(monkeypatch):
    """ES_EXPIRY set: adapter.resolve_front_month_future NOT called,
    manifest ES sub patched with override value, Redis write + gauge set."""
    monkeypatch.setenv("ES_EXPIRY", "20251219")

    fake_fut = MagicMock()
    fake_fut.lastTradeDateOrContractMonth = "20251219"
    adapter = MagicMock()
    adapter.resolve_front_month_future = AsyncMock(return_value=fake_fut)

    redis_client = fakeredis.FakeRedis()
    manifest = _es_bars_manifest()

    async def _run():
        import infra.feed.run as run_mod

        calls = []
        original_write = run_mod.write_front_month

        def capturing_write(*args, **kwargs):
            calls.append((args, kwargs))
            return original_write(*args, **kwargs)

        monkeypatch.setattr(run_mod, "write_front_month", capturing_write)
        result, _roots = await _resolve_front_months(manifest, adapter, redis_client)
        return result, calls

    result, write_calls = asyncio.run(_run())

    # resolve_front_month_future must NOT have been called
    adapter.resolve_front_month_future.assert_not_called()

    # The ES bars sub must carry the override expiry
    es_subs = [s for s in result if s.kind == "bars" and s.contract and s.contract.get("symbol") == "ES"]
    assert len(es_subs) == 1
    assert es_subs[0].contract["expiry"] == "20251219"

    # write_front_month must have been called with the override value
    assert len(write_calls) == 1
    assert write_calls[0][1].get("expiry") == "20251219" or write_calls[0][0][2] == "20251219"


def test_resolve_front_months_auto_resolve_path(monkeypatch):
    """ES_EXPIRY unset: adapter.resolve_front_month_future called once,
    resolved expiry flows into the patched sub, Redis write + gauge set."""
    monkeypatch.delenv("ES_EXPIRY", raising=False)

    resolved_expiry = "20251219"
    fake_fut = MagicMock()
    fake_fut.lastTradeDateOrContractMonth = resolved_expiry
    fake_fut.localSymbol = "ESZ5"
    adapter = MagicMock()
    adapter.resolve_front_month_future = AsyncMock(return_value=fake_fut)

    redis_client = fakeredis.FakeRedis()
    manifest = _es_bars_manifest()

    async def _run():
        import infra.feed.run as run_mod

        write_calls = []
        original_write = run_mod.write_front_month

        def capturing_write(*args, **kwargs):
            write_calls.append((args, kwargs))
            return original_write(*args, **kwargs)

        monkeypatch.setattr(run_mod, "write_front_month", capturing_write)
        result, _roots = await _resolve_front_months(manifest, adapter, redis_client)
        return result, write_calls

    result, write_calls = asyncio.run(_run())

    # resolve_front_month_future must have been called exactly once
    adapter.resolve_front_month_future.assert_awaited_once()

    # The ES bars sub must carry the resolved expiry
    es_subs = [s for s in result if s.kind == "bars" and s.contract and s.contract.get("symbol") == "ES"]
    assert len(es_subs) == 1
    assert es_subs[0].contract["expiry"] == resolved_expiry

    # write_front_month must have been called with the resolved value
    assert len(write_calls) == 1
    assert write_calls[0][1].get("expiry") == resolved_expiry or write_calls[0][0][2] == resolved_expiry


def test_resolve_front_months_skip_when_no_es_bars(monkeypatch):
    """Manifest has no ES bars subscriptions: resolve NOT called,
    write_front_month NOT called, subscriptions returned unchanged."""
    monkeypatch.delenv("ES_EXPIRY", raising=False)

    adapter = MagicMock()
    adapter.resolve_front_month_future = AsyncMock()

    redis_client = fakeredis.FakeRedis()
    manifest = _breadth_only_manifest()

    async def _run():
        import infra.feed.run as run_mod

        write_calls = []
        original_write = run_mod.write_front_month

        def capturing_write(*args, **kwargs):
            write_calls.append((args, kwargs))
            return original_write(*args, **kwargs)

        monkeypatch.setattr(run_mod, "write_front_month", capturing_write)
        result, _roots = await _resolve_front_months(manifest, adapter, redis_client)
        return result, write_calls

    result, write_calls = asyncio.run(_run())

    # resolve_front_month_future must NOT have been called
    adapter.resolve_front_month_future.assert_not_called()

    # write_front_month must NOT have been called
    assert write_calls == []

    # The subscription list must be returned unchanged
    assert result == list(manifest.subscriptions)


# --- _seconds_until_next_pre_open -----------------------------------------


def test_seconds_until_next_pre_open_before_target():
    """At 07:00 CT, next 08:00 CT is exactly 1 hour (3600 s) away."""
    from zoneinfo import ZoneInfo

    chi = ZoneInfo("America/Chicago")
    now = datetime(2026, 5, 14, 7, 0, 0, tzinfo=chi)
    assert _seconds_until_next_pre_open(now) == 3600.0


def test_seconds_until_next_pre_open_after_target():
    """At 09:00 CT, next 08:00 CT is tomorrow - exactly 23 hours away."""
    from zoneinfo import ZoneInfo

    chi = ZoneInfo("America/Chicago")
    now = datetime(2026, 5, 14, 9, 0, 0, tzinfo=chi)
    assert _seconds_until_next_pre_open(now) == 23 * 3600.0


def test_seconds_until_next_pre_open_exactly_at_target():
    """At exactly 08:00:00 CT, the target is not in the future so the
    function targets tomorrow's 08:00 - returning 24 hours."""
    from zoneinfo import ZoneInfo

    chi = ZoneInfo("America/Chicago")
    now = datetime(2026, 5, 14, 8, 0, 0, tzinfo=chi)
    assert _seconds_until_next_pre_open(now) == 24 * 3600.0


def test_seconds_until_next_pre_open_end_of_month():
    """Month-boundary: May 31 -> June 1 must not produce an invalid date."""
    from zoneinfo import ZoneInfo

    chi = ZoneInfo("America/Chicago")
    # 09:00 CT on May 31 - next 08:00 is June 1 (23 h later)
    now = datetime(2026, 5, 31, 9, 0, 0, tzinfo=chi)
    result = _seconds_until_next_pre_open(now)
    assert result == 23 * 3600.0


# --- _pre_open_requalify_loop -------------------------------------------


def test_requalify_loop_logs_unchanged_and_continues(monkeypatch, caplog):
    """When IBKR returns the same expiry that is in Redis, the loop logs
    INFO 'unchanged' and continues without writing Redis or updating the gauge."""
    import logging

    import fakeredis

    import infra.feed.run as run_mod
    from alpha_assay.data.front_month import read_front_month, write_front_month

    redis_client = fakeredis.FakeRedis()
    write_front_month(redis_client, symbol="ES", exchange="CME", expiry="20260619")

    fake_fut = MagicMock()
    fake_fut.lastTradeDateOrContractMonth = "20260619"  # same as Redis - unchanged
    fake_fut.localSymbol = "ESM6"

    adapter = MagicMock()
    adapter.resolve_front_month_future = AsyncMock(return_value=fake_fut)

    # Patch _seconds_until_next_pre_open to return 0 so the loop fires immediately.
    monkeypatch.setattr(run_mod, "_seconds_until_next_pre_open", lambda _now: 0.0)

    async def _run():
        task = asyncio.create_task(_pre_open_requalify_loop(adapter, redis_client))
        # Give the loop time to fire one iteration (sleep(0) + resolve + compare).
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    with caplog.at_level(logging.INFO, logger="alpha_assay.ibkr_feed"):
        asyncio.run(_run())

    # resolve was actually awaited (branch was exercised, not short-circuited by cancel)
    adapter.resolve_front_month_future.assert_awaited()
    # Redis key must remain unchanged
    assert read_front_month(redis_client, symbol="ES", exchange="CME") == "20260619"
    # The unchanged branch emits an INFO log with "unchanged" in the message
    assert any("unchanged" in r.message for r in caplog.records if r.levelno == logging.INFO)


def test_requalify_loop_detects_rollover(monkeypatch):
    """When IBKR returns a new expiry the loop writes the new key + sets the gauge."""
    import fakeredis

    import infra.feed.run as run_mod
    from alpha_assay.data.front_month import read_front_month, write_front_month

    redis_client = fakeredis.FakeRedis()
    write_front_month(redis_client, symbol="ES", exchange="CME", expiry="20260620")

    fake_fut = MagicMock()
    fake_fut.lastTradeDateOrContractMonth = "20260918"  # new front month
    fake_fut.localSymbol = "ESU6"

    adapter = MagicMock()
    adapter.resolve_front_month_future = AsyncMock(return_value=fake_fut)

    # Patch _seconds_until_next_pre_open to return 0 so the loop fires immediately.
    monkeypatch.setattr(run_mod, "_seconds_until_next_pre_open", lambda _now: 0.0)

    async def _run():
        task = asyncio.create_task(_pre_open_requalify_loop(adapter, redis_client))
        # Give the loop time to fire one iteration (sleep(0) + resolve + write).
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())

    adapter.resolve_front_month_future.assert_awaited()
    assert read_front_month(redis_client, symbol="ES", exchange="CME") == "20260918"


def test_requalify_loop_continues_on_ibkr_error(monkeypatch):
    """An IBKR exception on one iteration must not kill the loop - it
    continues sleeping until the next 08:00 CT."""
    import fakeredis

    import infra.feed.run as run_mod
    from alpha_assay.data.front_month import read_front_month, write_front_month

    redis_client = fakeredis.FakeRedis()
    write_front_month(redis_client, symbol="ES", exchange="CME", expiry="20260620")

    call_count = {"n": 0}

    async def flaky_resolve(**_kw):
        call_count["n"] += 1
        raise RuntimeError("IBKR timeout")

    adapter = MagicMock()
    adapter.resolve_front_month_future = flaky_resolve

    monkeypatch.setattr(run_mod, "_seconds_until_next_pre_open", lambda _now: 0.0)

    async def _run():
        task = asyncio.create_task(_pre_open_requalify_loop(adapter, redis_client))
        # Let multiple iterations fire.
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())

    # Loop must have attempted more than one iteration despite the errors.
    assert call_count["n"] > 1
    # Redis key must remain unchanged (no write on failure).
    assert read_front_month(redis_client, symbol="ES", exchange="CME") == "20260620"


# --- _resolve_front_months: multi-root (ES + NQ) ---------------------------


def _es_nq_manifest(*, nq_required: bool = False) -> FeedManifest:
    """ES (required) + NQ bars plus one breadth ticks subscription."""
    return FeedManifest(
        subscriptions=[
            Subscription(
                kind="bars",
                contract={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20250919"},
            ),
            Subscription(
                kind="bars",
                contract={"symbol": "NQ", "sec_type": "FUT", "exchange": "CME"},
                required=nq_required,
            ),
            Subscription(kind="ticks", symbol="TICK-NYSE"),
        ]
    )


def _resolver_for(expiries: dict[str, str]):
    """AsyncMock-style resolver returning a per-symbol fake Future."""

    async def _resolve(symbol: str, exchange: str, currency: str = "USD"):
        expiry = expiries[symbol]
        fut = MagicMock()
        fut.lastTradeDateOrContractMonth = expiry
        fut.localSymbol = f"{symbol}-{expiry}"
        return fut

    return _resolve


def test_resolve_front_months_resolves_each_root(monkeypatch):
    """ES and NQ each get their own ContFuture resolve, Redis key, and
    re-qualify root entry; every bars sub carries its root's expiry."""
    import fakeredis

    from alpha_assay.data.front_month import read_front_month

    monkeypatch.delenv("ES_EXPIRY", raising=False)
    monkeypatch.delenv("NQ_EXPIRY", raising=False)

    adapter = MagicMock()
    adapter.resolve_front_month_future = _resolver_for({"ES": "20260918", "NQ": "20261218"})

    redis_client = fakeredis.FakeRedis()
    manifest = _es_nq_manifest()

    result, roots = asyncio.run(_resolve_front_months(manifest, adapter, redis_client))

    by_symbol = {s.contract["symbol"]: s for s in result if s.kind == "bars"}
    assert by_symbol["ES"].contract["expiry"] == "20260918"
    assert by_symbol["NQ"].contract["expiry"] == "20261218"
    # NQ keeps its optional flag through the patch.
    assert by_symbol["NQ"].required is False

    # Each root has its own Redis metadata key.
    assert read_front_month(redis_client, symbol="ES", exchange="CME") == "20260918"
    assert read_front_month(redis_client, symbol="NQ", exchange="CME") == "20261218"

    assert roots == [("ES", "CME", "USD"), ("NQ", "CME", "USD")]


def test_resolve_front_months_nq_env_override(monkeypatch):
    """NQ_EXPIRY pins NQ without an IBKR call for NQ; ES still auto-resolves."""
    import fakeredis

    monkeypatch.delenv("ES_EXPIRY", raising=False)
    monkeypatch.setenv("NQ_EXPIRY", "20270319")

    calls: list[str] = []

    async def _resolve(symbol: str, exchange: str, currency: str = "USD"):
        calls.append(symbol)
        fut = MagicMock()
        fut.lastTradeDateOrContractMonth = "20260918"
        fut.localSymbol = f"{symbol}U6"
        return fut

    adapter = MagicMock()
    adapter.resolve_front_month_future = _resolve

    result, _roots = asyncio.run(_resolve_front_months(_es_nq_manifest(), adapter, fakeredis.FakeRedis()))

    assert calls == ["ES"], f"only ES should hit ContFuture; got {calls}"
    nq = next(s for s in result if s.kind == "bars" and s.contract["symbol"] == "NQ")
    assert nq.contract["expiry"] == "20270319"


def test_resolve_front_months_optional_root_failure_drops_subs(monkeypatch):
    """A resolve failure on an all-optional root drops its subs and keeps going."""
    import fakeredis

    from alpha_assay.data.front_month import read_front_month

    monkeypatch.delenv("ES_EXPIRY", raising=False)
    monkeypatch.delenv("NQ_EXPIRY", raising=False)

    async def _resolve(symbol: str, exchange: str, currency: str = "USD"):
        if symbol == "NQ":
            raise RuntimeError("no NQ entitlement")
        fut = MagicMock()
        fut.lastTradeDateOrContractMonth = "20260918"
        fut.localSymbol = "ESU6"
        return fut

    adapter = MagicMock()
    adapter.resolve_front_month_future = _resolve

    redis_client = fakeredis.FakeRedis()
    result, roots = asyncio.run(_resolve_front_months(_es_nq_manifest(nq_required=False), adapter, redis_client))

    symbols = [s.contract["symbol"] for s in result if s.kind == "bars"]
    assert symbols == ["ES"], f"NQ subs must be dropped on optional-root failure; got {symbols}"
    # The ticks subscription is untouched.
    assert any(s.kind == "ticks" for s in result)
    assert roots == [("ES", "CME", "USD")]
    assert read_front_month(redis_client, symbol="ES", exchange="CME") == "20260918"


def test_resolve_front_months_required_root_failure_raises(monkeypatch):
    """A resolve failure on a root with any required sub propagates (daemon exits)."""
    import fakeredis

    monkeypatch.delenv("ES_EXPIRY", raising=False)
    monkeypatch.delenv("NQ_EXPIRY", raising=False)

    async def _resolve(symbol: str, exchange: str, currency: str = "USD"):
        raise RuntimeError("IBKR down")

    adapter = MagicMock()
    adapter.resolve_front_month_future = _resolve

    with pytest.raises(RuntimeError, match="IBKR down"):
        asyncio.run(_resolve_front_months(_es_bars_manifest(), adapter, fakeredis.FakeRedis()))


# --- optional-subscription failure isolation -------------------------------


def test_run_subscriptions_survives_optional_subscription_failure(redis_client, tmp_path):
    """An optional (required: false) subscription that explodes must NOT
    exit the daemon: the required feed keeps running and the daemon only
    stops on the requested stop event (rc == 0)."""
    from alpha_assay.observability import metrics as M

    adapter = _adapter_with_blocking_bars()

    def _failing_breadth(symbol=None, exchange="NYSE", currency="USD", **kw):
        raise RuntimeError("market data entitlement missing")

    adapter.subscribe_breadth = _failing_breadth

    optional_ticks = Subscription(kind="ticks", symbol="VIX", exchange="CBOE", required=False)
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal")
    stop = asyncio.Event()

    async def _run():
        async def _stop_later():
            await asyncio.sleep(0.3)
            stop.set()

        asyncio.create_task(_stop_later())
        return await asyncio.wait_for(
            _run_subscriptions(daemon, adapter, [_bars_sub(), optional_ticks], stop, watchdog_poll=0.02),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc == 0, "optional-subscription failure must not exit the daemon"
    # The gauge reflects the dead optional feed.
    assert M.feed_subscription_up.labels(stream="ticks.vix")._value.get() == 0.0
    # The required feed's gauge is still up.
    assert M.feed_subscription_up.labels(stream="bars.es.cme.20260618")._value.get() == 1.0


def test_run_subscriptions_required_ticks_failure_still_exits(redis_client, tmp_path):
    """The same failure on a required subscription keeps the historical
    exit-and-restart behaviour."""
    adapter = _adapter_with_blocking_bars()

    def _failing_breadth(symbol=None, exchange="NYSE", currency="USD", **kw):
        raise RuntimeError("market data entitlement missing")

    adapter.subscribe_breadth = _failing_breadth

    required_ticks = Subscription(kind="ticks", symbol="TICK-NYSE")
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal")
    stop = asyncio.Event()

    async def _run():
        return await asyncio.wait_for(
            _run_subscriptions(daemon, adapter, [_bars_sub(), required_ticks], stop, watchdog_poll=0.02),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc != 0
