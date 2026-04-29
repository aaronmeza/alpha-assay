# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Regression tests for the IBKRAdapter async/sync mismatch.

Background
----------

Before this fix, ``IBKRAdapter.subscribe_bars`` was declared ``async def``
but invoked ``self._ib.reqHistoricalData(...)``, which is the synchronous
ib_insync API. ``ib_insync`` implements those sync wrappers via
``util.run() -> loop.run_until_complete()``. Calling them from inside an
already-running asyncio event loop raises::

    RuntimeError: This event loop is already running

That is exactly what crashed the ES-bars recorder and the breadth recorder on first deployment against a real IB Gateway: both
recorders run an async ``run()`` loop that called ``adapter.connect()``
and iterated ``adapter.subscribe_bars()`` from inside the loop.

The unit tests at the time substituted the ``_ib`` member with a
``MagicMock``, so the real ``util.run`` path was never exercised and the
bug landed on main.

These tests pin the async call site directly. They:

1. Run :meth:`IBKRAdapter.subscribe_bars` inside ``asyncio.run`` with an
   :class:`unittest.mock.AsyncMock` standing in for
   ``ib_insync.IB.reqHistoricalDataAsync``, asserting that one canonical
   bar yields without raising and that the awaited path is taken (not
   the sync sibling).
2. Pin :meth:`IBKRAdapter.connect_async` to ``ib_insync.IB.connectAsync``
   in the same way - sync ``connect`` is the legacy entrypoint kept for
   ``IBKRExecAdapter`` and ``scripts/paper_dryrun.py`` which call it
   from outside any event loop.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from alpha_assay.data.ibkr_adapter import IBKRAdapter


class _FakeEvent:
    """Minimal eventkit.Event stand-in supporting ``+=`` / ``-=``."""

    def __init__(self) -> None:
        self._handlers: list[Any] = []

    def __iadd__(self, h: Any) -> _FakeEvent:
        self._handlers.append(h)
        return self

    def __isub__(self, h: Any) -> _FakeEvent:
        if h in self._handlers:
            self._handlers.remove(h)
        return self


class _FakeBarDataList(list):
    """Stand-in for ib_insync.BarDataList: list with an updateEvent attr."""

    def __init__(self, bars: list[Any]) -> None:
        super().__init__(bars)
        self.updateEvent = _FakeEvent()


def _bar(ts: str, *, close: float = 5000.0) -> Any:
    """Build a single SimpleNamespace bar that quacks like ib_insync.BarData."""
    return SimpleNamespace(
        date=pd.Timestamp(ts).to_pydatetime(),
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        volume=100,
        average=close,
        barCount=10,
    )


def test_subscribe_bars_awaits_async_ib_api_no_runtime_error() -> None:
    """The async call site must not invoke the sync ``reqHistoricalData``.

    If the implementation regresses to the sync API, the AsyncMock
    sibling would never be awaited and the synchronous one would be
    invoked instead - which is exactly the path that calls
    ``loop.run_until_complete`` and crashes a real recorder.
    """
    fake_list = _FakeBarDataList([_bar("2026-04-28T14:30:00Z", close=5005.0)])

    ib = MagicMock(name="IB")
    ib.reqHistoricalDataAsync = AsyncMock(return_value=fake_list)

    adapter = IBKRAdapter(ib=ib)
    spec = {
        "symbol": "ES",
        "sec_type": "FUT",
        "exchange": "CME",
        "currency": "USD",
        "expiry": "202606",
    }

    async def _drain_one() -> dict[str, Any]:
        gen = adapter.subscribe_bars(spec)
        try:
            return await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        finally:
            await gen.aclose()

    ev = asyncio.run(_drain_one())

    # Assert the AsyncMock was awaited - this is what proves the call
    # site uses the async API, not the sync one.
    ib.reqHistoricalDataAsync.assert_awaited_once()
    # The sync sibling must NOT be touched. (MagicMock returns a child
    # MagicMock on attribute access, so we can't just assert
    # not_called - guard it explicitly via the mock's configure_mock.)
    assert "reqHistoricalData" not in [c[0] for c in ib.method_calls]

    assert ev["close"] == 5005.0
    assert ev["feed"] == "ES-FUT-202606"


def test_connect_async_awaits_ib_connectAsync() -> None:
    """``IBKRAdapter.connect_async`` must await ib_insync's ``connectAsync``.

    Sync ``connect`` is preserved for sync callers (executor,
    paper_dryrun); async callers (recorders) must use
    ``connect_async`` to avoid the nested ``run_until_complete`` crash.
    """
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = False
    ib.connectAsync = AsyncMock(return_value=None)

    adapter = IBKRAdapter(ib=ib, host="10.0.0.5", port=4002, client_id=42)

    asyncio.run(adapter.connect_async())

    ib.connectAsync.assert_awaited_once()
    kwargs = ib.connectAsync.await_args.kwargs
    assert kwargs["host"] == "10.0.0.5"
    assert kwargs["port"] == 4002
    assert kwargs["clientId"] == 42
    assert kwargs["readonly"] is True


def test_connect_async_propagates_failure_and_increments_error_counter() -> None:
    """Connection failures from the async path must mirror the sync path:
    increment the error counter and re-raise.
    """
    from alpha_assay.observability import metrics as M

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = False
    ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError("boom"))

    adapter = IBKRAdapter(ib=ib)

    before = M.ibkr_connection_events_total.labels(event="error")._value.get()
    raised = False
    try:
        asyncio.run(adapter.connect_async())
    except ConnectionRefusedError:
        raised = True
    assert raised is True

    after = M.ibkr_connection_events_total.labels(event="error")._value.get()
    assert after == before + 1


def test_disconnect_async_is_safe_from_running_loop() -> None:
    """``disconnect_async`` should be callable inside ``asyncio.run`` without
    raising. ib_insync >= 0.9.86 has no ``disconnectAsync``, so the
    async sibling forwards to the sync ``disconnect`` which itself does
    not call ``run_until_complete``.
    """
    ib = MagicMock(name="IB")
    # First call: connected -> disconnect should fire. Second call: idempotent.
    ib.isConnected.side_effect = [True, False]

    adapter = IBKRAdapter(ib=ib)

    async def _flow() -> None:
        await adapter.disconnect_async()
        await adapter.disconnect_async()

    asyncio.run(_flow())
    assert ib.disconnect.call_count == 1


def test_subscribe_breadth_does_not_require_async_ib_api() -> None:
    """``reqMktData`` is a fire-and-forget subscription registration in
    ib_insync; it does not call ``run_until_complete`` and therefore is
    safe to call from inside a running event loop without an ``Async``
    sibling. This test pins that contract: the adapter calls the sync
    method even from within ``asyncio.run``.
    """
    ib = MagicMock(name="IB")
    ib.pendingTickersEvent = _FakeEvent()
    fake_ticker = SimpleNamespace(contract=SimpleNamespace())
    ib.reqMktData.return_value = fake_ticker

    adapter = IBKRAdapter(ib=ib)

    async def _kick() -> None:
        gen = adapter.subscribe_breadth(symbol="TICK-NYSE")
        # Step the generator just enough to trigger the reqMktData call.
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        await gen.aclose()

    asyncio.run(_kick())

    assert ib.reqMktData.call_count == 1
