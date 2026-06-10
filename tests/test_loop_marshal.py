# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for the ib_insync loop-affinity marshaling layer.

The defect this layer fixes: exec-adapter calls are made from
bus-consumer worker threads while the ib_insync event loop runs
elsewhere, and ib_insync's IB/Client objects are not thread-safe
(``Client.sendMsg`` resolves the THREAD-CURRENT loop and writes to the
asyncio transport, which must happen on the loop owning the
connection). The invariants verified here:

- ``IBLoopThread.call`` from any non-owner thread executes the function
  ON the owning loop's thread, with that loop installed as the
  thread-current loop (so ``ib_insync.util.getLoop()`` resolves to it).
- Callers block for the result and fail loudly (``ExecMarshalTimeout``)
  on timeout.
- Re-entrant calls from the loop thread execute inline (no
  self-marshal deadlock).
- ``FillDispatcher`` delivers fill events on its own thread, preserving
  order, so a fill callback that takes a strategy lock can never wedge
  the IB loop - the lock-inversion scenario is exercised end-to-end.

No ib_insync objects here; the layer is generic over callables.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from alpha_assay.exec.loop_marshal import (
    ExecMarshalTimeout,
    FillDispatcher,
    IBLoopThread,
)


@pytest.fixture()
def loop_thread():
    lt = IBLoopThread(name="test-ib-loop")
    lt.start()
    yield lt
    lt.stop()


# ---------------------------------------------------------------------
# IBLoopThread: loop ownership + marshaling
# ---------------------------------------------------------------------


def test_call_executes_on_owner_loop_thread(loop_thread):
    """A call from the test (non-owner) thread runs on the loop thread,
    with the owned loop installed as that thread's current loop - the
    exact property ib_insync's util.getLoop() depends on."""
    seen: dict = {}

    def probe():
        seen["thread"] = threading.current_thread()
        seen["running_loop"] = asyncio.get_running_loop()
        # What ib_insync's util.getLoop() resolves on this thread.
        seen["policy_loop"] = asyncio.get_event_loop_policy().get_event_loop()
        return 42

    result = loop_thread.call(probe)

    assert result == 42
    assert seen["thread"] is not threading.current_thread()
    assert seen["thread"].name == "test-ib-loop"
    assert seen["running_loop"] is loop_thread.loop
    assert seen["policy_loop"] is loop_thread.loop


def test_call_from_worker_thread_lands_on_owner_loop(loop_thread):
    """Same property when the caller is itself a worker thread (the
    bus-consumer shape: run_in_executor threads driving exec calls)."""
    seen: dict = {}
    results: list = []

    def probe():
        seen["thread"] = threading.current_thread()
        return "ok"

    def worker():
        results.append(loop_thread.call(probe))

    t = threading.Thread(target=worker, name="bus-worker")
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert results == ["ok"]
    assert seen["thread"].name == "test-ib-loop"


def test_call_returns_kwargs_result_and_propagates_exceptions(loop_thread):
    assert loop_thread.call(lambda a, b=0: a + b, 1, b=2) == 3

    with pytest.raises(ValueError, match="boom"):
        loop_thread.call(_raise_value_error)


def _raise_value_error():
    raise ValueError("boom")


def test_call_timeout_raises_exec_marshal_timeout(loop_thread):
    """A wedged loop must surface to the caller as a loud, typed error
    within the timeout - never an indefinite block."""
    with pytest.raises(ExecMarshalTimeout, match="did not complete"):
        loop_thread.call(time.sleep, 0.5, timeout=0.05)
    # Let the wedging sleep finish so stop() can join cleanly.
    time.sleep(0.6)


def test_call_reentrant_from_loop_thread_executes_inline(loop_thread):
    """A callback already running on the loop thread (e.g. a fill
    handler that decides to place an order) must not marshal-and-block
    onto its own loop - that is a self-deadlock. call() detects owner-
    thread re-entry and executes directly."""

    def inner():
        assert loop_thread.is_owner_thread()
        return "inner-ran"

    def outer():
        # We are ON the loop thread now; a blocking re-marshal would
        # deadlock until the outer timeout. Inline execution returns
        # immediately.
        return loop_thread.call(inner, timeout=1.0)

    assert loop_thread.call(outer, timeout=2.0) == "inner-ran"


def test_run_coro_executes_on_loop_and_returns(loop_thread):
    async def coro():
        await asyncio.sleep(0)
        return asyncio.get_running_loop()

    assert loop_thread.run_coro(coro()) is loop_thread.loop


def test_run_coro_from_loop_thread_raises(loop_thread):
    async def coro():  # pragma: no cover - never awaited
        return None

    def on_loop():
        c = coro()
        try:
            with pytest.raises(RuntimeError, match="deadlock"):
                loop_thread.run_coro(c)
        finally:
            c.close()

    loop_thread.call(on_loop)


def test_call_before_start_raises():
    lt = IBLoopThread(name="never-started")
    with pytest.raises(RuntimeError, match="not started"):
        lt.call(lambda: None)


def test_start_twice_raises_and_stop_is_idempotent():
    lt = IBLoopThread(name="lifecycle")
    lt.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            lt.start()
    finally:
        lt.stop()
    lt.stop()  # second stop is a no-op


def test_stop_from_loop_thread_rejected(loop_thread):
    def try_stop():
        with pytest.raises(RuntimeError, match="loop thread itself"):
            loop_thread.stop()

    loop_thread.call(try_stop)


# ---------------------------------------------------------------------
# FillDispatcher: delivery off the IB loop, in order
# ---------------------------------------------------------------------


def test_fill_dispatcher_delivers_on_own_thread_in_order():
    delivered: list = []
    done = threading.Event()

    def deliver(evt):
        delivered.append((evt["i"], threading.current_thread()))
        if evt["i"] == 9:
            done.set()

    dispatcher = FillDispatcher(deliver, name="test-fill-dispatch")
    dispatcher.start()
    try:
        for i in range(10):
            dispatcher.submit({"i": i})
        assert done.wait(timeout=5)
    finally:
        dispatcher.stop()

    assert [i for i, _ in delivered] == list(range(10))
    threads = {t for _, t in delivered}
    assert len(threads) == 1
    (worker,) = threads
    assert worker is not threading.current_thread()
    assert worker.name == "test-fill-dispatch"


def test_fill_dispatcher_survives_delivery_exception():
    delivered: list = []
    done = threading.Event()

    def deliver(evt):
        if evt["i"] == 0:
            raise RuntimeError("first event explodes")
        delivered.append(evt["i"])
        done.set()

    dispatcher = FillDispatcher(deliver)
    dispatcher.start()
    try:
        dispatcher.submit({"i": 0})
        dispatcher.submit({"i": 1})
        assert done.wait(timeout=5)
    finally:
        dispatcher.stop()

    assert delivered == [1]


def test_fill_dispatcher_stop_drains_queue():
    delivered: list = []
    dispatcher = FillDispatcher(lambda evt: delivered.append(evt["i"]))
    dispatcher.start()
    for i in range(5):
        dispatcher.submit({"i": i})
    dispatcher.stop()
    assert delivered == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------
# The deadlock scenario, end to end
# ---------------------------------------------------------------------


def test_fill_path_is_deadlock_free_under_lock_inversion(loop_thread):
    """The exact production hazard, reproduced structurally:

    1. A bus-consumer thread takes the strategy runner's lock and, still
       holding it, blocks on an exec call marshaled to the IB loop.
    2. While that marshaled call runs ON the IB loop, a fill fires
       (ib_insync fires filledEvent on the loop thread) whose delivery
       callback needs the SAME lock.

    If the fill callback ran inline on the IB loop it would block on the
    lock, the loop could never finish the marshaled call, and the bus
    thread would never release the lock: classic lock-inversion
    deadlock. With FillDispatcher, the loop thread only enqueues; the
    dispatcher thread blocks on the lock instead, the marshaled call
    completes, the lock is released, and the fill is delivered.
    """
    runner_lock = threading.RLock()
    fill_delivered = threading.Event()
    fill_threads: list = []

    def deliver(evt):
        # Mirrors PaperStrategyRunner.handle_fill: takes the runner lock.
        with runner_lock:
            fill_threads.append(threading.current_thread())
            fill_delivered.set()

    dispatcher = FillDispatcher(deliver, name="fill-dispatch-inversion")
    dispatcher.start()

    def exec_call_on_loop():
        # Mirrors placeOrder processing: while on the IB loop, the fill
        # event fires. Inline delivery would deadlock right here.
        assert loop_thread.is_owner_thread()
        dispatcher.submit({"order_id": 1})
        return "placed"

    results: list = []

    def bus_worker():
        with runner_lock:  # _on_row holds the lock across the exec call
            results.append(loop_thread.call(exec_call_on_loop, timeout=5.0))
            # Hold the lock a beat longer so the dispatcher is provably
            # blocked on it while we still hold it.
            time.sleep(0.1)

    t = threading.Thread(target=bus_worker, name="bus-worker")
    t.start()
    t.join(timeout=5)

    try:
        assert not t.is_alive(), "bus worker deadlocked"
        assert results == ["placed"]
        assert fill_delivered.wait(timeout=5), "fill never delivered"
        assert fill_threads[0].name == "fill-dispatch-inversion"
        assert fill_threads[0] is not loop_thread._thread
    finally:
        dispatcher.stop()
