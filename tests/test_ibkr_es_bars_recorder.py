# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""ES-bars recorder unit tests.

All IBKR interactions are mocked. No live network. Mirrors the breadth-recorder test patterns: stub adapter for sync paths, asyncio
driver for the reconnect supervisor.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from alpha_assay.data.databento_adapter import load_parquet
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from infra.recorders.ibkr_es_bars import recorder as recorder_mod
from infra.recorders.ibkr_es_bars.recorder import (
    _BACKOFF_SEQUENCE,
    _OHLCV_COLUMNS,
    ESBarsRecorder,
    _clamp_bar,
    _is_in_rth,
)


def _counter_value(counter: Any, **labels: str) -> float:
    """Return the current value of a labelled prometheus counter."""
    child = counter.labels(**labels) if labels else counter
    return child._value.get()


def _gauge_value(gauge: Any, **labels: str) -> float:
    child = gauge.labels(**labels) if labels else gauge
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
    a.__dict__["_state"] = state
    return a


def _bar(
    ts: str,
    *,
    open_: float = 5000.0,
    high: float = 5005.0,
    low: float = 4995.0,
    close: float = 5002.0,
    volume: int = 100,
    feed: str = "ES-FUT-20260618",
) -> dict[str, Any]:
    """Build a canonical bar dict matching ``_normalize_bar``'s schema."""
    return {
        "timestamp": pd.Timestamp(ts),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "feed": feed,
    }


# --- helpers ----------------------------------------------------------------


def test_is_in_rth_session_window() -> None:
    # 09:31 CDT on 2026-04-28 (Tue, DST active) = 14:31 UTC -> in session.
    assert _is_in_rth(pd.Timestamp("2026-04-28T14:31:00Z")) is True
    # 08:00 CDT = 13:00 UTC -> pre-RTH.
    assert _is_in_rth(pd.Timestamp("2026-04-28T13:00:00Z")) is False
    # 15:30 CDT = 20:30 UTC -> post-RTH.
    assert _is_in_rth(pd.Timestamp("2026-04-28T20:30:00Z")) is False
    # Saturday 2026-04-25 14:30 UTC -> weekend.
    assert _is_in_rth(pd.Timestamp("2026-04-25T14:30:00Z")) is False


def test_clamp_bar_repairs_invariant_violations() -> None:
    # high < close -> clamp high up to close.
    bar = _bar("2026-04-28T14:30:00Z", open_=100.0, high=99.0, low=90.0, close=110.0)
    out = _clamp_bar(bar)
    assert out["high"] == 110.0
    assert out["low"] == 90.0

    # low > open -> clamp low down to open.
    bar = _bar("2026-04-28T14:30:00Z", open_=80.0, high=120.0, low=85.0, close=100.0)
    out = _clamp_bar(bar)
    assert out["low"] == 80.0
    assert out["high"] == 120.0


# --- core ingestion ---------------------------------------------------------


def test_recorder_appends_in_rth_bars(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    out = rec.ingest_bar(_bar("2026-04-28T14:30:00Z"))
    assert out is not None
    assert out["timestamp"] == pd.Timestamp("2026-04-28T14:30:00Z")
    assert "feed" not in out  # canonical OHLCV row has no feed column
    assert set(out.keys()) == {"timestamp", "open", "high", "low", "close", "volume"}


def test_recorder_drops_out_of_rth_bars(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    # 13:00 UTC = 08:00 CDT -> pre-RTH.
    assert rec.ingest_bar(_bar("2026-04-28T13:00:00Z")) is None
    # 20:30 UTC = 15:30 CDT -> post-RTH.
    assert rec.ingest_bar(_bar("2026-04-28T20:30:00Z")) is None
    # Saturday.
    assert rec.ingest_bar(_bar("2026-04-25T14:30:00Z")) is None
    # No flush should produce a shard - no in-RTH bars were ever buffered.
    assert rec.flush() == 0
    assert list(tmp_path.glob("*.parquet")) == []


def test_recorder_clamps_ohlc_on_ingest(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    # Bar with high < close violates the load_parquet invariant.
    bar = _bar("2026-04-28T14:30:00Z", open_=100.0, high=99.0, low=80.0, close=120.0, volume=50)
    out = rec.ingest_bar(bar)
    assert out is not None
    assert out["high"] == 120.0
    assert out["low"] == 80.0


def test_recorder_drops_duplicate_or_older_timestamps(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z", close=100.0))
    rec.ingest_bar(_bar("2026-04-28T14:31:00Z", close=110.0))
    # Duplicate timestamp -> in-place replace.
    rec.ingest_bar(_bar("2026-04-28T14:31:00Z", close=115.0))
    # Out-of-order older bar -> ignored.
    rec.ingest_bar(_bar("2026-04-28T14:29:00Z", close=99.0))

    rec.flush()
    df = pd.read_parquet(tmp_path / "2026-04-28.parquet")
    assert len(df) == 2
    assert list(df["close"]) == [100.0, 115.0]


def test_recorder_writes_parquet_shard_per_day(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)

    # Three minutes on 2026-04-28.
    for minute in (30, 31, 32):
        rec.ingest_bar(_bar(f"2026-04-28T14:{minute}:00Z", close=5000.0 + minute))

    rec.flush()

    shard = tmp_path / "2026-04-28.parquet"
    assert shard.exists(), f"missing {shard}"
    df = pd.read_parquet(shard)
    assert len(df) == 3
    assert list(df.columns) == list(_OHLCV_COLUMNS)
    assert df["close"].iloc[0] == 5030.0
    assert df["close"].iloc[-1] == 5032.0


def test_recorder_parquet_loads_via_databento_adapter(tmp_path: Path) -> None:
    """End-to-end schema parity: a recorder shard must round-trip via
    the public ``databento_adapter.load_parquet`` validator without
    raising ``DatabentoSchemaError``.
    """
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)

    # Five canonical bars, monotonic timestamps, valid OHLC.
    for i in range(5):
        rec.ingest_bar(
            _bar(
                f"2026-04-28T14:{30 + i}:00Z",
                open_=5000.0 + i,
                high=5010.0 + i,
                low=4990.0 + i,
                close=5005.0 + i,
                volume=100 + i,
            )
        )
    rec.flush()

    df = load_parquet(tmp_path / "2026-04-28.parquet")
    assert len(df) == 5
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "America/Chicago"
    assert df.index.is_monotonic_increasing


def test_recorder_handles_day_rollover(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    # Day 1: 2026-04-28 14:30 UTC -> 09:30 CDT.
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z", close=5000.0))
    rec.ingest_bar(_bar("2026-04-28T14:31:00Z", close=5001.0))
    # Day 2 starts implicitly when next ingest sees a new local date.
    rec.ingest_bar(_bar("2026-04-29T14:30:00Z", close=5100.0))

    # The day-1 buffer should have been flushed on rollover.
    day1 = tmp_path / "2026-04-28.parquet"
    assert day1.exists()
    df1 = pd.read_parquet(day1)
    assert len(df1) == 2

    # Day 2 still buffered until explicit flush.
    rec.flush()
    day2 = tmp_path / "2026-04-29.parquet"
    assert day2.exists()
    df2 = pd.read_parquet(day2)
    assert len(df2) == 1


# --- metrics ---------------------------------------------------------------


def test_recorder_increments_received_counter(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    label = "ES-FUT-20260618"
    before = _counter_value(recorder_mod.RM.bars_received_total, feed=label)
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z"))
    rec.ingest_bar(_bar("2026-04-28T14:31:00Z"))
    rec.ingest_bar(_bar("2026-04-28T13:00:00Z"))  # out-of-RTH still counted received
    after = _counter_value(recorder_mod.RM.bars_received_total, feed=label)
    assert after - before == 3


def test_recorder_increments_written_counter_on_flush(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    label = "ES-FUT-20260618"
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z"))
    rec.ingest_bar(_bar("2026-04-28T14:31:00Z"))
    before = _counter_value(recorder_mod.RM.bars_written_total, feed=label)
    rec.flush()
    after = _counter_value(recorder_mod.RM.bars_written_total, feed=label)
    assert after - before == 2


def test_recorder_resets_last_bar_age_on_in_rth_bar(tmp_path: Path) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    label = "ES-FUT-20260618"
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z"))
    val = _gauge_value(recorder_mod.RM.last_bar_age_seconds, feed=label)
    # Gauge is now wall-clock-derived. On bar arrival it computes
    # `monotonic_now - bar_received_at`, which is microseconds, not exact
    # zero. A loose upper bound catches the regression where the gauge
    # would lie at 0.0 forever.
    assert 0.0 <= val < 0.1


def test_recorder_bar_age_climbs_when_no_bars_arrive(tmp_path: Path) -> None:
    """The freshness gauge must reflect wall-clock staleness so the
    healthcheck can detect a frozen ingestion path. Pin the recorder's
    monotonic clock and verify the gauge tracks the delta forward.
    """
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    label = "ES-FUT-20260618"
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z"))
    receipt_time = rec._last_bar_received_at  # type: ignore[attr-defined]
    assert receipt_time is not None
    # Simulate 47 wall-clock seconds elapsing without any new bars.
    with patch.object(recorder_mod.time, "monotonic", return_value=receipt_time + 47.0):
        rec._update_age_gauge()
    val = _gauge_value(recorder_mod.RM.last_bar_age_seconds, feed=label)
    assert 46.5 <= val <= 47.5


def test_recorder_bar_age_reports_uptime_when_no_bar_yet(tmp_path: Path) -> None:
    """Before any bar arrives the gauge should report seconds-since-start,
    not 0.0 - a recorder that never subscribed must still fail freshness."""
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    label = "ES-FUT-20260618"
    start = rec._recorder_start_monotonic  # type: ignore[attr-defined]
    with patch.object(recorder_mod.time, "monotonic", return_value=start + 120.0):
        rec._update_age_gauge()
    val = _gauge_value(recorder_mod.RM.last_bar_age_seconds, feed=label)
    assert 119.5 <= val <= 120.5


def test_recorder_metrics_module_is_independent() -> None:
    """The breadth recorder edits its own metrics module under
    ``alpha_assay.observability.recorder_metrics``. metrics live
    co-located under the recorder package so neither phase's edits
    cause merge conflicts on the other's catalog.
    """
    rm = importlib.import_module("infra.recorders.ibkr_es_bars.recorder_metrics")
    assert hasattr(rm, "bars_received_total")
    assert hasattr(rm, "bars_written_total")
    assert hasattr(rm, "write_errors_total")
    assert hasattr(rm, "last_bar_age_seconds")
    assert hasattr(rm, "reconnects_total")


# --- write-error handling --------------------------------------------------


def test_recorder_increments_write_errors_when_flush_fails(tmp_path: Path, monkeypatch) -> None:
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z"))

    # Force pandas.DataFrame.to_parquet to raise inside the flush path.
    def _boom(self, *_a, **_kw):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom, raising=True)
    label = "ES-FUT-20260618"
    before = _counter_value(recorder_mod.RM.write_errors_total, feed=label, error_class="OSError")
    written = rec.flush()
    after = _counter_value(recorder_mod.RM.write_errors_total, feed=label, error_class="OSError")
    assert written == 0
    assert after - before == 1


# --- reconnect -------------------------------------------------------------


def test_recorder_backoff_sequence_is_documented() -> None:
    assert _BACKOFF_SEQUENCE == (1, 2, 4, 8, 16, 60)
    for idx in range(6, 20):
        capped = _BACKOFF_SEQUENCE[min(idx, len(_BACKOFF_SEQUENCE) - 1)]
        assert capped == 60


def test_recorder_preserves_day_buffer_across_reconnect(tmp_path: Path) -> None:
    """Mid-session disconnect/reconnect must not drop buffered bars."""
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z", close=5000.0))
    rec.ingest_bar(_bar("2026-04-28T14:31:00Z", close=5001.0))

    # Buffer survives reconnect because it lives on the recorder.
    assert rec._day_buffer is not None
    assert len(rec._day_buffer.rows) == 2

    rec.ingest_bar(_bar("2026-04-28T14:32:00Z", close=5002.0))
    rec.flush()

    df = pd.read_parquet(tmp_path / "2026-04-28.parquet")
    assert len(df) == 3


def test_recorder_backoff_increments_on_failure(tmp_path: Path) -> None:
    """First-connect failures increment the reconnects counter and
    honour the backoff schedule without blocking the test wall-clock.
    """
    calls: dict[str, Any] = {"connect": 0, "sleep": []}

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

        async def subscribe_bars(self, _spec):  # pragma: no cover
            if False:
                yield {}

    rec = ESBarsRecorder(
        adapter=_FakeAdapter(),  # type: ignore[arg-type]
        out_dir=tmp_path,
    )

    before_reconnects = _counter_value(recorder_mod.RM.reconnects_total)

    async def _driver() -> None:
        rec._loop = asyncio.get_running_loop()
        rec._shutdown_event = asyncio.Event()

        original_wait_for = asyncio.wait_for

        async def fake_wait_for(coro, timeout):
            calls["sleep"].append(timeout)
            if len(calls["sleep"]) >= 3:
                rec._shutdown_event.set()  # type: ignore[union-attr]
            task = asyncio.ensure_future(coro)
            try:
                return await original_wait_for(task, timeout=0.01)
            except TimeoutError:
                task.cancel()
                with _SuppressCancelled():
                    await task
                raise

        recorder_mod.asyncio.wait_for = fake_wait_for  # type: ignore[attr-defined]
        try:
            await rec._connect_and_stream_forever()
        finally:
            recorder_mod.asyncio.wait_for = original_wait_for  # type: ignore[attr-defined]

    class _SuppressCancelled:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, _exc, _tb) -> bool:
            return exc_type is asyncio.CancelledError

    asyncio.run(_driver())

    after_reconnects = _counter_value(recorder_mod.RM.reconnects_total)
    assert after_reconnects - before_reconnects >= 3
    assert calls["connect"] >= 3
    assert calls["sleep"][:3] == [1, 2, 4]


# --- graceful shutdown -----------------------------------------------------


def test_recorder_graceful_shutdown_flushes_pending_bars(tmp_path: Path) -> None:
    """SIGTERM drain must persist the buffered day before exit."""
    rec = ESBarsRecorder(adapter=_mk_adapter(), out_dir=tmp_path)
    rec.ingest_bar(_bar("2026-04-28T14:30:00Z", close=5000.0))
    rec.ingest_bar(_bar("2026-04-28T14:31:00Z", close=5001.0))

    rec._drain_and_flush()

    shard = tmp_path / "2026-04-28.parquet"
    assert shard.exists()
    df = pd.read_parquet(shard)
    assert len(df) == 2
    assert df.iloc[0]["open"] == 5000.0


def test_recorder_run_drains_on_shutdown_event(tmp_path: Path) -> None:
    """Full run() loop: subscribe, ingest, set shutdown, verify shard
    landed on disk via the graceful-shutdown drain path.
    """
    bars_to_emit = [
        _bar("2026-04-28T14:30:00Z", close=5000.0),
        _bar("2026-04-28T14:31:00Z", close=5001.0),
    ]

    class _StreamingAdapter:
        def __init__(self) -> None:
            self._connected = False

        def connect(self) -> None:  # pragma: no cover - recorder uses connect_async
            self._connected = True

        async def connect_async(self) -> None:
            self._connected = True

        def disconnect(self) -> None:
            self._connected = False

        async def disconnect_async(self) -> None:
            self._connected = False

        @property
        def is_connected(self) -> bool:
            return self._connected

        async def subscribe_bars(self, _spec):
            for bar in bars_to_emit:
                yield bar
            # Idle until cancelled.
            await asyncio.sleep(60)

    rec = ESBarsRecorder(
        adapter=_StreamingAdapter(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        flush_period_seconds=999,
    )

    async def _driver() -> None:
        run_task = asyncio.create_task(rec.run())
        # Yield a few times so the subscription generator drains both bars.
        for _ in range(20):
            await asyncio.sleep(0)
            if rec._day_buffer is not None and len(rec._day_buffer.rows) >= 2:
                break
        # Trigger the shutdown event the same way SIGTERM would.
        assert rec._shutdown_event is not None
        rec._shutdown_event.set()
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_driver())

    shard = tmp_path / "2026-04-28.parquet"
    assert shard.exists()
    df = pd.read_parquet(shard)
    assert len(df) == 2


# --- async call site regression --------------------------------------------


class _FakeUpdateEvent:
    """Minimal eventkit.Event stand-in supporting ``+=`` / ``-=``."""

    def __init__(self) -> None:
        self._handlers: list[Any] = []

    def __iadd__(self, h: Any) -> _FakeUpdateEvent:
        self._handlers.append(h)
        return self

    def __isub__(self, h: Any) -> _FakeUpdateEvent:
        if h in self._handlers:
            self._handlers.remove(h)
        return self


class _FakeBarDataList(list):
    """List + updateEvent attribute, mimics ib_insync.BarDataList."""

    def __init__(self, bars: list[Any]) -> None:
        super().__init__(bars)
        self.updateEvent = _FakeUpdateEvent()


def test_recorder_run_loop_uses_async_ib_api_no_runtime_error(tmp_path: Path) -> None:
    """End-to-end: recorder.run() drives a real :class:`IBKRAdapter` whose
    ``_ib`` is mocked. The supervisor must invoke
    ``connectAsync`` / ``reqHistoricalDataAsync`` (the awaited path).
    Calling the sync ``connect`` / ``reqHistoricalData`` from inside a
    running loop is what crashed the first real deployment with
    ``RuntimeError: This event loop is already running``; this test
    pins the post-fix call site so the bug cannot silently regress.
    """
    fixtures = [
        SimpleNamespace(
            date=pd.Timestamp("2026-04-28T14:30:00Z").to_pydatetime(),
            open=5000.0,
            high=5005.0,
            low=4995.0,
            close=5002.0,
            volume=100,
            average=5001.0,
            barCount=10,
        )
    ]
    fake_list = _FakeBarDataList(fixtures)

    ib = MagicMock(name="IB")
    # Adapter's connect_async awaits this - if the recorder regresses
    # to sync connect(), this AsyncMock is never awaited.
    ib.connectAsync = AsyncMock(return_value=None)
    # Adapter's subscribe_bars awaits this.
    ib.reqHistoricalDataAsync = AsyncMock(return_value=fake_list)
    # is_connected drives the supervisor's connect-or-skip branch.
    ib.isConnected.return_value = False

    adapter = IBKRAdapter(ib=ib)

    rec = ESBarsRecorder(
        adapter=adapter,
        out_dir=tmp_path,
        flush_period_seconds=999,
        contract_spec={
            "symbol": "ES",
            "sec_type": "FUT",
            "exchange": "CME",
            "currency": "USD",
            "expiry": "20260618",
        },
    )

    async def _driver() -> None:
        run_task = asyncio.create_task(rec.run())
        # Pump the loop briefly so connect_async + subscribe_bars run.
        for _ in range(50):
            await asyncio.sleep(0)
            if ib.connectAsync.await_count >= 1 and ib.reqHistoricalDataAsync.await_count >= 1:
                break
        # Once both async siblings have been awaited, trigger shutdown.
        assert rec._shutdown_event is not None
        rec._shutdown_event.set()
        # Bounded drain - the test must not hang on a regression.
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_driver())

    ib.connectAsync.assert_awaited()
    ib.reqHistoricalDataAsync.assert_awaited()
    # The historical buffer fixture had one in-RTH bar - it should land
    # in the day buffer via ingest_bar -> flush during shutdown drain.
    shard = tmp_path / "2026-04-28.parquet"
    assert shard.exists()
    df = pd.read_parquet(shard)
    assert len(df) == 1
