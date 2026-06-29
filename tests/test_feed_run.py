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

from infra.feed.feed import FeedManifest, FreshnessTracker, IBKRFeedDaemon, Subscription
from infra.feed.run import (
    EXIT_RESTART,
    _connect_with_retry,
    _env_bool,
    _in_rth,
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

    roll_restart = asyncio.Event()

    async def _run():
        task = asyncio.create_task(_pre_open_requalify_loop(adapter, redis_client, roll_restart=roll_restart))
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
    # No rollover -> no restart requested.
    assert not roll_restart.is_set()


def test_requalify_loop_detects_rollover_requests_restart(monkeypatch):
    """When IBKR returns a new expiry the loop requests a self-restart and
    returns WITHOUT pre-writing the key/gauge. Cold-start (after the restart)
    writes the new key + bakes the new stream atomically, so the key must NOT
    lead the live stream here."""
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

    roll_restart = asyncio.Event()

    async def _run():
        task = asyncio.create_task(_pre_open_requalify_loop(adapter, redis_client, roll_restart=roll_restart))
        # The loop sets roll_restart and returns on its own - no cancel needed.
        await asyncio.wait_for(roll_restart.wait(), timeout=1.0)
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(_run())

    adapter.resolve_front_month_future.assert_awaited()
    # Restart requested...
    assert roll_restart.is_set()
    # ...and the key was NOT pre-written: it stays the old expiry until cold-start
    # re-resolves and writes it (key + stream flip together, atomically).
    assert read_front_month(redis_client, symbol="ES", exchange="CME") == "20260620"


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

    roll_restart = asyncio.Event()

    async def _run():
        task = asyncio.create_task(_pre_open_requalify_loop(adapter, redis_client, roll_restart=roll_restart))
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
    # A resolve failure must NOT request a restart.
    assert not roll_restart.is_set()


def test_run_subscriptions_exits_restart_on_rollover(redis_client, tmp_path):
    """A front-month rollover (roll_restart set by a re-qualify loop) wins the
    FIRST_COMPLETED race and returns EXIT_RESTART, so the daemon cold-starts
    onto the new contract (key + stream flip atomically)."""
    adapter = _adapter_with_blocking_bars()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal")
    stop = asyncio.Event()
    roll_restart = asyncio.Event()

    async def _run():
        async def _roll_later():
            await asyncio.sleep(0.1)
            roll_restart.set()

        asyncio.create_task(_roll_later())
        return await asyncio.wait_for(
            _run_subscriptions(daemon, adapter, [_bars_sub()], stop, watchdog_poll=0.02, roll_restart=roll_restart),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc == EXIT_RESTART
    # A clean stop was NOT requested - this is a restart, not a graceful shutdown.
    assert not stop.is_set()


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


# --- data-staleness watchdog: FreshnessTracker -----------------------------
#
# The connection watchdog only catches a dropped socket. A *silent* socket -
# IBKR connected but pushing no bars/ticks - leaves the publish loop idle with
# is_connected still True (the 2026-06-17 14:06-14:24 CT stall). The staleness
# watchdog closes that gap. FreshnessTracker is the publish-boundary clock it
# reads.


def test_freshness_tracker_age_none_when_never_seen():
    tr = FreshnessTracker()
    assert tr.age("never-published") is None


def test_freshness_tracker_age_climbs_then_resets_on_mark():
    clock = {"t": 100.0}
    tr = FreshnessTracker(clock=lambda: clock["t"])

    tr.seed("bars.es.cme.20260618")
    assert tr.age("bars.es.cme.20260618") == 0.0

    clock["t"] = 130.0
    assert tr.age("bars.es.cme.20260618") == 30.0

    tr.mark("bars.es.cme.20260618")
    assert tr.age("bars.es.cme.20260618") == 0.0


def test_freshness_tracker_age_never_negative_on_clock_step_back():
    clock = {"t": 100.0}
    tr = FreshnessTracker(clock=lambda: clock["t"])
    tr.mark("s")
    clock["t"] = 90.0  # monotonic should never do this, but clamp defensively
    assert tr.age("s") == 0.0


def test_run_subscription_marks_freshness_on_publish(redis_client, tmp_path):
    """The daemon stamps the tracker at the publish boundary so the watchdog
    can see data is flowing."""
    adapter = _adapter_with_blocking_bars()
    tracker = FreshnessTracker()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal", freshness=tracker)

    async def _run():
        task = asyncio.create_task(daemon.run_subscription(_bars_sub()))
        await asyncio.sleep(0.1)  # let it publish the single bar
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    # The one published bar must have marked the stream as fresh.
    assert tracker.age("bars.es.cme.20260618") is not None


# --- data-staleness watchdog: _in_rth + _env_bool --------------------------


def _ct(y, m, d, hh, mm):
    from zoneinfo import ZoneInfo

    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo("America/Chicago"))


def test_in_rth_true_during_session():
    assert _in_rth(_ct(2026, 6, 17, 10, 0)) is True  # Wed 10:00 CT


def test_in_rth_false_before_open():
    assert _in_rth(_ct(2026, 6, 17, 7, 0)) is False


def test_in_rth_false_after_close():
    assert _in_rth(_ct(2026, 6, 17, 15, 30)) is False


def test_in_rth_open_boundary_inclusive():
    assert _in_rth(_ct(2026, 6, 17, 8, 30)) is True


def test_in_rth_close_boundary_exclusive():
    # Matches the alerter's half-open [open, close) window.
    assert _in_rth(_ct(2026, 6, 17, 15, 0)) is False


def test_in_rth_false_on_weekend():
    assert _in_rth(_ct(2026, 6, 20, 10, 0)) is False  # Saturday


def test_env_bool_default_when_unset(monkeypatch):
    monkeypatch.delenv("Y9Q_FLAG", raising=False)
    assert _env_bool("Y9Q_FLAG", default=True) is True
    assert _env_bool("Y9Q_FLAG", default=False) is False


def test_env_bool_empty_uses_default(monkeypatch):
    monkeypatch.setenv("Y9Q_FLAG", "")
    assert _env_bool("Y9Q_FLAG", default=True) is True


def test_env_bool_false_values(monkeypatch):
    for v in ("0", "false", "no", "off", "FALSE", "Off"):
        monkeypatch.setenv("Y9Q_FLAG", v)
        assert _env_bool("Y9Q_FLAG", default=True) is False, v


def test_env_bool_true_values(monkeypatch):
    for v in ("1", "true", "yes", "on", "True"):
        monkeypatch.setenv("Y9Q_FLAG", v)
        assert _env_bool("Y9Q_FLAG", default=False) is True, v


# --- data-staleness watchdog: _watch_staleness -----------------------------


# These tests drive a FAKE monotonic clock (shared by the tracker and the
# watchdog) and advance it deterministically inside the monkeypatched _in_rth -
# one fixed step per poll - so the staleness logic is governed by fake time,
# never by real wall-clock timing under CI load. Real asyncio.sleep stays tiny.


def test_watch_staleness_trips_on_stale_required_feed_during_rth(monkeypatch):
    import infra.feed.run as run_mod

    clock = {"t": 1000.0}

    def fake_in_rth(*_a, **_k):
        clock["t"] += 60.0  # 60s of fake time per poll
        return True

    monkeypatch.setattr(run_mod, "_in_rth", fake_in_rth)
    tracker = FreshnessTracker(clock=lambda: clock["t"])
    stream = "bars.es.cme.20260618"
    tracker.seed(stream)  # seeded, then never marked -> goes stale in fake time

    async def _run():
        return await asyncio.wait_for(
            run_mod._watch_staleness(
                tracker, [stream], timeout_seconds=180.0, poll_seconds=0.001, clock=lambda: clock["t"]
            ),
            timeout=2.0,
        )

    assert asyncio.run(_run()) == stream


def test_watch_staleness_does_not_trip_when_feed_fresh(monkeypatch):
    import infra.feed.run as run_mod

    clock = {"t": 1000.0}
    tracker = FreshnessTracker(clock=lambda: clock["t"])
    stream = "bars.es.cme.20260618"
    tracker.seed(stream)

    def fake_in_rth(*_a, **_k):
        clock["t"] += 60.0
        tracker.mark(stream)  # publish every poll -> age stays ~0
        return True

    monkeypatch.setattr(run_mod, "_in_rth", fake_in_rth)

    async def _run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                run_mod._watch_staleness(
                    tracker, [stream], timeout_seconds=180.0, poll_seconds=0.001, clock=lambda: clock["t"]
                ),
                timeout=0.2,
            )

    asyncio.run(_run())


def test_watch_staleness_does_not_trip_outside_rth(monkeypatch):
    import infra.feed.run as run_mod

    clock = {"t": 1000.0}

    def fake_in_rth(*_a, **_k):
        clock["t"] += 600.0  # huge fake gap, but never in RTH
        return False

    monkeypatch.setattr(run_mod, "_in_rth", fake_in_rth)
    tracker = FreshnessTracker(clock=lambda: clock["t"])
    stream = "ticks.tick-nyse"
    tracker.seed(stream)  # arbitrarily stale, but outside RTH must stay silent

    async def _run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                run_mod._watch_staleness(
                    tracker, [stream], timeout_seconds=180.0, poll_seconds=0.001, clock=lambda: clock["t"]
                ),
                timeout=0.2,
            )

    asyncio.run(_run())


def test_watch_staleness_rth_open_warmup_blocks_after_reentry(monkeypatch):
    """A feed already stale before RTH opens must still get a full
    timeout_seconds warmup AFTER RTH (re)opens before it can trip - even though
    the process-start warmup expired long ago while out of session. Proves the
    RTH-open guard is independent of the startup guard and that the per-session
    re-baseline holds the trip until at least timeout_seconds past RTH open."""
    import infra.feed.run as run_mod

    clock = {"t": 1000.0}
    tracker = FreshnessTracker(clock=lambda: clock["t"])
    stream = "bars.es.cme.20260918"
    tracker.seed(stream)  # stale from here; never marked again

    calls = {"n": 0}
    rth_open_clock = 1000.0 + 6 * 60.0  # RTH opens on the 6th poll (clock=1360)

    def fake_in_rth(*_a, **_k):
        calls["n"] += 1
        clock["t"] += 60.0  # 60s per poll
        # First 5 polls: OUT of RTH (process-start warmup expires here). Then in.
        return calls["n"] > 5

    monkeypatch.setattr(run_mod, "_in_rth", fake_in_rth)

    async def _run():
        return await asyncio.wait_for(
            run_mod._watch_staleness(
                tracker, [stream], timeout_seconds=180.0, poll_seconds=0.001, clock=lambda: clock["t"]
            ),
            timeout=2.0,
        )

    assert asyncio.run(_run()) == stream
    # The trip must not happen until at least timeout_seconds after RTH open,
    # even though the startup warmup (now - start) expired before RTH opened.
    assert clock["t"] - rth_open_clock >= 180.0


# --- data-staleness watchdog: _run_subscriptions integration ---------------


def test_run_subscriptions_staleness_restart_on_silent_required_feed(monkeypatch, redis_client, tmp_path):
    """A required feed that publishes once then goes silent while the socket
    stays connected must trip the staleness watchdog -> EXIT_RESTART, counter
    incremented, adapter disconnected for a fresh reconnect. is_connected
    stays True throughout, proving this is the staleness path, not the
    connection-loss path."""
    import infra.feed.run as run_mod
    from alpha_assay.observability import metrics as M

    monkeypatch.setattr(run_mod, "_in_rth", lambda *_a, **_k: True)

    adapter = _adapter_with_blocking_bars()  # yields one bar, then silent forever
    tracker = FreshnessTracker()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal", freshness=tracker)
    stop = asyncio.Event()
    stream = "bars.es.cme.20260618"
    before = M.feed_staleness_restarts_total.labels(stream=stream)._value.get()
    # Capture connection-loss metrics up front: the staleness path must NOT
    # touch them (the socket is live; only the data went silent).
    disc_before = M.ibkr_connection_events_total.labels(event="disconnected")._value.get()
    conn_before = M.ibkr_connected._value.get()

    async def _run():
        return await asyncio.wait_for(
            _run_subscriptions(
                daemon,
                adapter,
                [_bars_sub()],
                stop,
                watchdog_poll=0.02,
                staleness_enabled=True,
                staleness_timeout=0.05,
                staleness_poll=0.01,
                freshness=tracker,
            ),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc != 0
    assert adapter.is_connected is True, "connection never dropped; this is the staleness path"
    adapter.disconnect.assert_called()
    after = M.feed_staleness_restarts_total.labels(stream=stream)._value.get()
    assert after == before + 1
    # The staleness path must NOT falsely report a connection loss (design
    # invariant): the disconnect counter and ibkr_connected gauge are untouched.
    disc_after = M.ibkr_connection_events_total.labels(event="disconnected")._value.get()
    assert disc_after == disc_before, "staleness restart must not increment the disconnect counter"
    assert M.ibkr_connected._value.get() == conn_before, "staleness restart must not change the ibkr_connected gauge"


def test_run_subscriptions_staleness_trips_required_despite_healthy_optional(monkeypatch, redis_client, tmp_path):
    """A silent REQUIRED feed must trip even when an OPTIONAL feed keeps
    streaming - optional-feed health must never mask required-feed staleness."""
    import infra.feed.run as run_mod
    from alpha_assay.observability import metrics as M

    monkeypatch.setattr(run_mod, "_in_rth", lambda *_a, **_k: True)

    adapter = _adapter_with_blocking_bars()  # required bars: one yield, then silent

    async def streaming_breadth(symbol=None, exchange="NYSE", currency="USD", **kw):
        while True:
            yield {"value": 1.0, "symbol": symbol}
            await asyncio.sleep(0.01)

    adapter.subscribe_breadth = streaming_breadth

    tracker = FreshnessTracker()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal", freshness=tracker)
    stop = asyncio.Event()
    optional_breadth = Subscription(kind="ticks", symbol="VIX", exchange="CBOE", required=False)
    stream = "bars.es.cme.20260618"
    before = M.feed_staleness_restarts_total.labels(stream=stream)._value.get()

    async def _run():
        return await asyncio.wait_for(
            _run_subscriptions(
                daemon,
                adapter,
                [_bars_sub(), optional_breadth],
                stop,
                watchdog_poll=0.02,
                staleness_enabled=True,
                staleness_timeout=0.05,
                staleness_poll=0.01,
                freshness=tracker,
            ),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc != 0, "a stale required feed must restart even while optional feeds stream"
    assert adapter.is_connected is True
    after = M.feed_staleness_restarts_total.labels(stream=stream)._value.get()
    assert after == before + 1


def test_run_subscriptions_staleness_disabled_does_not_restart(monkeypatch, redis_client, tmp_path):
    """With the staleness watchdog disabled, a silent (connected) required feed
    does NOT restart the daemon; only the stop event ends it (rc == 0)."""
    import infra.feed.run as run_mod

    monkeypatch.setattr(run_mod, "_in_rth", lambda *_a, **_k: True)

    adapter = _adapter_with_blocking_bars()
    tracker = FreshnessTracker()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal", freshness=tracker)
    stop = asyncio.Event()

    async def _run():
        async def _stop_later():
            await asyncio.sleep(0.3)
            stop.set()

        asyncio.create_task(_stop_later())
        return await asyncio.wait_for(
            _run_subscriptions(
                daemon,
                adapter,
                [_bars_sub()],
                stop,
                watchdog_poll=0.02,
                staleness_enabled=False,
                staleness_timeout=0.05,
                staleness_poll=0.01,
                freshness=tracker,
            ),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc == 0


def _adapter_with_streaming_bars(interval: float = 0.01) -> MagicMock:
    """Mock adapter whose bar subscription keeps yielding bars forever - a
    healthy required feed that never goes stale."""

    async def fake_subscribe_bars(spec, **kw):
        while True:
            yield {
                "timestamp": "2026-06-17T13:30:00+00:00",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1,
                "feed": "ES-FUT-20260618",
            }
            await asyncio.sleep(interval)

    adapter = MagicMock()
    adapter.connect_async = AsyncMock()
    adapter.is_connected = True
    adapter.subscribe_bars = fake_subscribe_bars
    return adapter


def test_run_subscriptions_staleness_ignores_optional_feed(monkeypatch, redis_client, tmp_path):
    """A silent OPTIONAL feed must NOT trip the staleness watchdog while the
    required feed keeps flowing - the required:false degrade-only contract.
    Only the stop event ends the daemon (rc == 0)."""
    import infra.feed.run as run_mod

    monkeypatch.setattr(run_mod, "_in_rth", lambda *_a, **_k: True)

    adapter = _adapter_with_streaming_bars()  # required feed stays fresh

    async def silent_breadth(symbol=None, exchange="NYSE", currency="USD", **kw):
        yield {"value": 1.0, "symbol": symbol}  # one tick, then silent
        while True:
            await asyncio.sleep(3600)

    adapter.subscribe_breadth = silent_breadth

    tracker = FreshnessTracker()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=tmp_path / "wal", freshness=tracker)
    stop = asyncio.Event()
    optional_ticks = Subscription(kind="ticks", symbol="VIX", exchange="CBOE", required=False)

    async def _run():
        async def _stop_later():
            await asyncio.sleep(0.3)
            stop.set()

        asyncio.create_task(_stop_later())
        return await asyncio.wait_for(
            _run_subscriptions(
                daemon,
                adapter,
                [_bars_sub(), optional_ticks],
                stop,
                watchdog_poll=0.02,
                staleness_enabled=True,
                staleness_timeout=0.05,
                staleness_poll=0.01,
                freshness=tracker,
            ),
            timeout=3.0,
        )

    rc = asyncio.run(_run())
    assert rc == 0, "a stale optional feed must not restart the daemon"
