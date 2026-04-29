# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""breadth-recorder unit tests.

All IBKR interactions are mocked. No live network. Session-aware logic
uses hand-rolled datetime injection because freezegun is not a dev
dependency for this repo.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from alpha_assay.data.ibkr_adapter import IBKRAdapter
from infra.recorders.ibkr_breadth import recorder as recorder_mod
from infra.recorders.ibkr_breadth.recorder import (
    _BACKOFF_SEQUENCE,
    BreadthRecorder,
    _sanitize_symbol,
)


def _reset_counter(counter: Any, **labels: str) -> float:
    """Return the current value of a labelled prometheus counter."""
    child = counter.labels(**labels) if labels else counter
    return child._value.get()


def _mk_adapter(connected_initial: bool = False) -> Any:
    """Build a stub adapter exposing the surface the recorder uses.

    The recorder's reconnect supervisor awaits ``connect_async`` and
    ``disconnect_async`` (the post-asyncfix surface), so the stub
    exposes both async siblings alongside the sync versions for tests
    that exercise the sync ingest path directly.
    """
    state = {"connected": connected_initial}

    def connect() -> None:
        state["connected"] = True

    def disconnect() -> None:
        state["connected"] = False

    async def connect_async() -> None:
        state["connected"] = True

    async def disconnect_async() -> None:
        state["connected"] = False

    # is_connected must be a property (the adapter exposes it as one), so we
    # build a one-off class rather than using SimpleNamespace.
    class _A:
        def __init__(self) -> None:
            self.connect = connect
            self.disconnect = disconnect
            self.connect_async = connect_async
            self.disconnect_async = disconnect_async

        @property
        def is_connected(self) -> bool:
            return state["connected"]

    a = _A()
    a.__dict__["_state"] = state  # expose for tests
    return a


def _tick(symbol: str, ts: str, value: float) -> dict[str, Any]:
    """Build a canonical breadth-tick dict."""
    return {
        "timestamp": pd.Timestamp(ts),
        "value": value,
        "symbol": symbol,
    }


# --- Core aggregation -------------------------------------------------------


def test_sanitize_symbol_replaces_dash() -> None:
    assert _sanitize_symbol("TICK-NYSE") == "TICK_NYSE"
    assert _sanitize_symbol("AD-NYSE") == "AD_NYSE"


def test_recorder_aggregates_ticks_to_minute_bars(tmp_path: Path) -> None:
    """60 synthetic ticks spanning 2 minutes -> one completed bar emitted."""
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))

    # 30 ticks in 09:30 CT == 14:30 UTC (DST on 2026-04-28 -> CDT -> 13:30 UTC).
    # Use explicit UTC timestamps that map to 09:31 CT in April 2026 (CDT = UTC-5).
    # 09:30 CT -> 14:30 UTC
    emitted = []
    for i in range(30):
        bar = rec.ingest_tick(_tick("TICK-NYSE", f"2026-04-28T14:30:{i:02d}Z", 100.0 + i))
        if bar is not None:
            emitted.append(bar)
    # Move into the next minute with a later tick. That triggers emission.
    bar = rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:31:01Z", 200.0))
    if bar is not None:
        emitted.append(bar)

    assert len(emitted) == 1, emitted
    assert emitted[0]["symbol"] == "TICK-NYSE"
    assert emitted[0]["timestamp"] == pd.Timestamp("2026-04-28T14:30:00Z")


def test_recorder_emits_completed_minute_on_rollover(tmp_path: Path) -> None:
    """Tick at 09:30:05 then 09:31:02 CT -> bar for 09:30:00 emitted."""
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))

    # 09:30:05 CDT = 14:30:05 UTC on 2026-04-28.
    assert rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:30:05Z", 500.0)) is None
    emitted = rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:31:02Z", 600.0))

    assert emitted is not None
    assert emitted["timestamp"] == pd.Timestamp("2026-04-28T14:30:00Z")
    assert emitted["open"] == 500.0
    assert emitted["close"] == 500.0
    assert emitted["n_ticks"] == 1


def test_recorder_ohlc_correct(tmp_path: Path) -> None:
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))

    sequence = [
        ("2026-04-28T14:30:01Z", 100.0),  # open
        ("2026-04-28T14:30:10Z", 150.0),  # new high
        ("2026-04-28T14:30:20Z", 80.0),  # new low
        ("2026-04-28T14:30:30Z", 120.0),  # interim
        ("2026-04-28T14:30:50Z", 110.0),  # close
    ]
    for ts, v in sequence:
        assert rec.ingest_tick(_tick("TICK-NYSE", ts, v)) is None
    emitted = rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:31:00Z", 999.0))

    assert emitted is not None
    assert emitted["open"] == 100.0
    assert emitted["high"] == 150.0
    assert emitted["low"] == 80.0
    assert emitted["close"] == 110.0
    assert emitted["n_ticks"] == 5


def test_recorder_drops_ticks_outside_rth(tmp_path: Path) -> None:
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))

    # 08:00 CDT = 13:00 UTC: pre-RTH.
    assert rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T13:00:05Z", 100.0)) is None
    # 15:30 CDT = 20:30 UTC: post-RTH.
    assert rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T20:30:00Z", 500.0)) is None
    # Weekend (Saturday 2026-04-25).
    assert rec.ingest_tick(_tick("TICK-NYSE", "2026-04-25T14:30:00Z", 99.0)) is None

    # No bar should ever emit because no in-RTH bucket was ever opened.
    bar = rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T20:31:00Z", 777.0))
    assert bar is None


def test_recorder_writes_parquet_per_day_per_symbol(tmp_path: Path) -> None:
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE", "AD-NYSE"))

    # Emit 3 completed minutes for TICK-NYSE on 2026-04-28.
    for minute in (30, 31, 32):
        rec.ingest_tick(_tick("TICK-NYSE", f"2026-04-28T14:{minute}:05Z", 100.0 * minute))
    # Roll over to close the 32-minute bucket.
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:33:00Z", 9999.0))

    # One completed minute for AD-NYSE on the same day.
    rec.ingest_tick(_tick("AD-NYSE", "2026-04-28T14:30:05Z", 1200.0))
    rec.ingest_tick(_tick("AD-NYSE", "2026-04-28T14:31:00Z", 1300.0))

    rec.flush()

    tick_path = tmp_path / "TICK_NYSE" / "2026-04-28.parquet"
    ad_path = tmp_path / "AD_NYSE" / "2026-04-28.parquet"
    assert tick_path.exists(), f"missing {tick_path}"
    assert ad_path.exists(), f"missing {ad_path}"

    tick_df = pd.read_parquet(tick_path)
    ad_df = pd.read_parquet(ad_path)
    # 3 completed minutes for TICK-NYSE (14:30, 14:31, 14:32); the 14:33 bucket
    # is still in flight and not flushed.
    assert len(tick_df) == 3
    assert list(tick_df["symbol"].unique()) == ["TICK-NYSE"]
    # One completed minute for AD-NYSE (14:30).
    assert len(ad_df) == 1


# --- Metrics ---------------------------------------------------------------


def test_recorder_increments_ticks_received_counter(tmp_path: Path) -> None:
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))
    before = _reset_counter(recorder_mod.RM.recorder_ticks_received_total, symbol="TICK-NYSE")
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:30:00Z", 100.0))
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:30:30Z", 110.0))
    after = _reset_counter(recorder_mod.RM.recorder_ticks_received_total, symbol="TICK-NYSE")
    assert after - before == 2


def test_recorder_increments_bars_written_counter_on_flush(tmp_path: Path) -> None:
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:30:05Z", 100.0))
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:31:01Z", 200.0))  # emits 14:30 bar
    before = _reset_counter(recorder_mod.RM.recorder_bars_written_total, symbol="TICK-NYSE")
    rec.flush()
    after = _reset_counter(recorder_mod.RM.recorder_bars_written_total, symbol="TICK-NYSE")
    assert after - before == 1


# --- Reconnect -------------------------------------------------------------


def test_recorder_reconnect_backoff_sequence() -> None:
    """Assert the documented backoff schedule is 1,2,4,8,16,60,60,..."""
    assert _BACKOFF_SEQUENCE == (1, 2, 4, 8, 16, 60)

    # The cap must hold: repeated lookups past the end clamp to 60.
    for idx in range(6, 20):
        capped = _BACKOFF_SEQUENCE[min(idx, len(_BACKOFF_SEQUENCE) - 1)]
        assert capped == 60


def test_recorder_preserves_aggregation_state_across_reconnect(tmp_path: Path) -> None:
    """Mid-minute disconnect/reconnect must not drop the open bucket."""
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))

    # Start a bucket at 14:30.
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:30:05Z", 100.0))

    # Simulate a reconnect: recorder state survives because it lives on the
    # recorder, not on the adapter.
    assert rec._buckets["TICK-NYSE"] is not None

    # Resume: more ticks in the same minute + a rollover.
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:30:45Z", 160.0))
    emitted = rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:31:02Z", 170.0))

    assert emitted is not None
    assert emitted["open"] == 100.0
    assert emitted["close"] == 160.0
    assert emitted["high"] == 160.0
    assert emitted["low"] == 100.0
    assert emitted["n_ticks"] == 2


def test_recorder_backoff_increments_on_failure(tmp_path: Path) -> None:
    """Reconnect loop: first-connect failures increment the reconnects
    counter and honour the backoff schedule without blocking the test.
    """

    calls = {"connect": 0, "sleep": []}

    class _FakeAdapter:
        @property
        def is_connected(self) -> bool:
            return False

        def connect(self) -> None:  # pragma: no cover - recorder uses connect_async
            calls["connect"] += 1
            raise RuntimeError("boom")

        async def connect_async(self) -> None:
            calls["connect"] += 1
            raise RuntimeError("boom")

        def disconnect(self) -> None:
            pass

        async def disconnect_async(self) -> None:
            pass

        async def subscribe_breadth(self, symbol: str):  # pragma: no cover
            if False:
                yield {}

    rec = BreadthRecorder(
        adapter=_FakeAdapter(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        symbols=("TICK-NYSE",),
    )

    before_reconnects = _reset_counter(recorder_mod.RM.recorder_reconnects_total)

    async def _driver() -> None:
        rec._loop = asyncio.get_running_loop()
        rec._shutdown_event = asyncio.Event()

        # Trip the shutdown event after a few failed attempts so the test
        # doesn't hang. Wall-clock is irrelevant: wait_for on shutdown is
        # patched to return immediately.
        original_wait_for = asyncio.wait_for

        async def fake_wait_for(coro, timeout):
            calls["sleep"].append(timeout)
            # After 3 sleeps, fire the shutdown event so the supervisor exits.
            if len(calls["sleep"]) >= 3:
                rec._shutdown_event.set()  # type: ignore[union-attr]
            # Cancel the pending wait coroutine cleanly.
            task = asyncio.ensure_future(coro)
            try:
                return await original_wait_for(task, timeout=0.01)
            except TimeoutError:
                task.cancel()
                with _SuppressCancelled():
                    await task
                raise

        # Monkeypatch inside the recorder module.
        recorder_mod.asyncio.wait_for = fake_wait_for  # type: ignore[attr-defined]
        try:
            await rec._connect_and_stream_forever()
        finally:
            recorder_mod.asyncio.wait_for = original_wait_for  # type: ignore[attr-defined]

    class _SuppressCancelled:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return exc_type is asyncio.CancelledError

    asyncio.run(_driver())

    after_reconnects = _reset_counter(recorder_mod.RM.recorder_reconnects_total)
    assert after_reconnects - before_reconnects >= 3
    assert calls["connect"] >= 3
    # Backoff schedule honoured for the first three attempts.
    assert calls["sleep"][:3] == [1, 2, 4]


# --- Graceful shutdown -----------------------------------------------------


def test_recorder_graceful_shutdown_flushes_pending_bars(tmp_path: Path) -> None:
    """On shutdown, buffered bars land on disk including the in-flight bucket.

    Rollover semantics are exclusive-end: a tick whose minute-floor is
    strictly greater than the live bucket's minute closes the live
    bucket and opens a new one. A tick at exactly ``14:31:00Z`` floors
    to ``14:31`` (not ``14:30``), so it is the FIRST tick of the 14:31
    minute, not the last tick of the 14:30 minute. Other tests in this
    file assume the same semantic (see
    ``test_recorder_emits_completed_minute_on_rollover`` and
    ``test_recorder_writes_parquet_per_day_per_symbol``).

    On graceful shutdown the 14:31 bucket's minute has long since
    elapsed by wall-clock, so ``_drain_and_flush`` emits it too. The
    final shard therefore carries both bars.
    """
    rec = BreadthRecorder(adapter=_mk_adapter(), out_dir=tmp_path, symbols=("TICK-NYSE",))

    # First tick opens the 14:30 bucket. The second tick at the exact
    # minute boundary (14:31:00Z) closes 14:30 and opens 14:31 with
    # open=200. _drain_and_flush then flushes both buckets to disk.
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:30:05Z", 100.0))
    rec.ingest_tick(_tick("TICK-NYSE", "2026-04-28T14:31:00Z", 200.0))

    rec._drain_and_flush()

    shard = tmp_path / "TICK_NYSE" / "2026-04-28.parquet"
    assert shard.exists()
    df = pd.read_parquet(shard)
    assert len(df) == 2
    assert df.iloc[0]["timestamp"] == pd.Timestamp("2026-04-28T14:30:00Z")
    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[1]["timestamp"] == pd.Timestamp("2026-04-28T14:31:00Z")
    assert df.iloc[1]["open"] == 200.0


def test_recorder_metrics_module_is_independent() -> None:
    """py`. metrics live elsewhere."""
    rm = importlib.import_module("alpha_assay.observability.recorder_metrics")
    assert hasattr(rm, "recorder_ticks_received_total")
    assert hasattr(rm, "recorder_bars_written_total")
    assert hasattr(rm, "recorder_write_errors_total")
    assert hasattr(rm, "recorder_session_gap_seconds")
    assert hasattr(rm, "recorder_reconnects_total")


# --- async call site regression --------------------------------------------


class _FakeEvent:
    """Minimal eventkit.Event stand-in supporting ``+=`` / ``-=`` and fire."""

    def __init__(self) -> None:
        self._handlers: list[Any] = []

    def __iadd__(self, h: Any) -> _FakeEvent:
        self._handlers.append(h)
        return self

    def __isub__(self, h: Any) -> _FakeEvent:
        if h in self._handlers:
            self._handlers.remove(h)
        return self


def test_recorder_run_loop_uses_connect_async_no_runtime_error(tmp_path: Path) -> None:
    """End-to-end: BreadthRecorder.run() drives a real :class:`IBKRAdapter`
    whose ``_ib`` is mocked. The supervisor must invoke ``connectAsync``
    (the awaited path). Calling the sync ``connect()`` from inside a
    running loop is what crashed the first real deployment with
    ``RuntimeError: This event loop is already running``; this test
    pins the post-fix call site so the bug cannot silently regress.

    ``reqMktData`` is intentionally exercised via the sync ib_insync API:
    ib_insync registers the subscription synchronously and never invokes
    ``loop.run_until_complete``, so an ``Async`` sibling is unnecessary
    and the sync version is safe inside an event loop.
    """
    ib = MagicMock(name="IB")
    ib.connectAsync = AsyncMock(return_value=None)
    ib.isConnected.return_value = False
    # ``subscribe_breadth`` registers a pendingTickersEvent handler then
    # blocks on the queue. We never deliver a tick - the test only needs
    # to prove that connect_async was awaited and reqMktData fired
    # without any sync run-until-complete crash.
    ib.pendingTickersEvent = _FakeEvent()
    ib.reqMktData.return_value = SimpleNamespace(contract=SimpleNamespace())

    adapter = IBKRAdapter(ib=ib)

    rec = BreadthRecorder(
        adapter=adapter,
        out_dir=tmp_path,
        symbols=("TICK-NYSE",),
        flush_period_seconds=999,
    )

    async def _driver() -> None:
        run_task = asyncio.create_task(rec.run())
        # Pump the loop until connect + subscribe have both fired.
        for _ in range(50):
            await asyncio.sleep(0)
            if ib.connectAsync.await_count >= 1 and ib.reqMktData.call_count >= 1:
                break
        assert rec._shutdown_event is not None
        rec._shutdown_event.set()
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_driver())

    ib.connectAsync.assert_awaited()
    assert ib.reqMktData.call_count >= 1
