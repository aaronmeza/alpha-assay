# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Loop-affinity marshaling for ib_insync interactions.

ib_insync's ``IB`` / ``Client`` objects are NOT thread-safe, and they are
loop-affine in a specific way: ``util.getLoop()`` resolves the
*thread-current* event loop (``asyncio.get_event_loop_policy()
.get_event_loop()``), not the running one. Three consequences drive this
module's design (verified against ib_insync 0.9.86 source):

1. ``Connection.connectAsync`` binds the TCP transport to ``getLoop()``
   at connect time - the loop current on the thread that executes the
   connect owns the transport forever after.
2. ``Client.sendMsg`` (the ``placeOrder`` / ``cancelOrder`` write path)
   calls ``getLoop()`` for ``loop.time()`` / ``loop.call_at`` throttling.
   Invoked from a thread with no current loop (any non-main worker
   thread on Python 3.12+) it raises ``RuntimeError``; invoked from a
   thread with a *different* current loop it schedules throttle
   callbacks on the wrong loop and writes to the transport from a
   foreign thread.
3. Incoming messages (order status, fills) are only processed while the
   owning loop is actually running, and ib_insync events fire on that
   loop's thread.

``IBLoopThread`` therefore owns a dedicated event loop on a dedicated
daemon thread: the loop is installed as that thread's current loop (so
every ``getLoop()`` inside ib_insync resolves to it), it runs forever
(so fills are delivered), and ``call()`` marshals arbitrary synchronous
ib_insync interactions onto it from any thread, blocking the caller for
the result with a hard timeout. There is no ``placeOrderAsync`` in
ib_insync - the sync calls are non-blocking client-side sends - so
wrapping them in a coroutine executed on the owning loop via
``asyncio.run_coroutine_threadsafe`` is the canonical mechanism.

``FillDispatcher`` hands fill events from the IB loop thread to a
dedicated consumer thread. This is a deadlock guard, not a nicety: a
bus-consumer thread may hold the strategy runner's lock while blocking
on a marshaled exec call; if fills were delivered inline on the IB loop
thread, the fill callback would block on that same runner lock and the
loop could never execute the marshaled call - lock-inversion deadlock.
With the dispatcher, the IB loop thread only ever enqueues and stays
free to serve marshaled calls. A single dispatcher thread preserves
fill ordering.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

_LOG = logging.getLogger(__name__)

# Marshaled exec calls are non-blocking client-side sends (placeOrder,
# cancelOrder) or local state reads (positions, openTrades); they only
# fail to return promptly if the IB loop itself is wedged. 10s is a
# generous bound that still fails loudly within a bar interval.
DEFAULT_CALL_TIMEOUT_SECONDS = 10.0

# Connect performs a real TCP handshake + API startup; ib_insync's own
# connect timeout applies underneath, this is the outer backstop.
CONNECT_TIMEOUT_SECONDS = 30.0

_THREAD_START_TIMEOUT_SECONDS = 5.0
_STOP_JOIN_TIMEOUT_SECONDS = 5.0


class ExecMarshalTimeout(RuntimeError):
    """A call marshaled onto the IB loop did not complete in time.

    Raised to the *calling* thread. The in-flight future is cancelled
    best-effort, but a synchronous ib_insync call that already started
    on the loop cannot be interrupted - a timeout here means the IB
    loop is wedged and the process should be treated as unhealthy.
    """


class IBLoopThread:
    """Owns a dedicated asyncio event loop on a daemon thread.

    Lifecycle: ``start()`` spawns the thread, installs a fresh loop as
    that thread's current loop, and runs it forever; ``stop()`` stops
    the loop and joins the thread. All ib_insync interaction for a
    connection created on this loop MUST go through ``call()`` /
    ``run_coro()``.
    """

    def __init__(self, *, name: str = "ib-loop") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    # --- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Spawn the loop thread and block until the loop is running."""
        if self._thread is not None:
            raise RuntimeError(f"IBLoopThread {self._name!r} already started")
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=_THREAD_START_TIMEOUT_SECONDS):
            raise RuntimeError(f"IBLoopThread {self._name!r} failed to start within {_THREAD_START_TIMEOUT_SECONDS}s")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        # ib_insync resolves its loop via util.getLoop() ==
        # get_event_loop_policy().get_event_loop(), i.e. the THREAD-
        # CURRENT loop. set_event_loop makes every getLoop() call inside
        # ib_insync code running on this thread resolve to our loop, so
        # the transport, throttle timers, and futures all bind here.
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._started.set()
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def stop(self, *, timeout: float = _STOP_JOIN_TIMEOUT_SECONDS) -> None:
        """Stop the loop and join the thread. Idempotent."""
        thread, loop = self._thread, self._loop
        if thread is None or loop is None or not thread.is_alive():
            return
        if self.is_owner_thread():
            raise RuntimeError("IBLoopThread.stop() must not be called from the loop thread itself")
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout)
        if thread.is_alive():
            # Daemon thread: process exit will reap it. Loud log so a
            # wedged loop is visible in operator output.
            _LOG.error("IBLoopThread %r did not stop within %.1fs; leaking daemon thread", self._name, timeout)

    # --- introspection ----------------------------------------------------

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError(f"IBLoopThread {self._name!r} not started")
        return self._loop

    def is_owner_thread(self) -> bool:
        """True when the calling thread IS the loop thread."""
        return self._thread is not None and threading.current_thread() is self._thread

    # --- marshaling -------------------------------------------------------

    def call(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> Any:
        """Run ``fn(*args, **kwargs)`` on the loop thread; return its result.

        Blocks the calling thread until the call completes or ``timeout``
        elapses (then raises :class:`ExecMarshalTimeout`). Re-entrant:
        when invoked FROM the loop thread (e.g. a future fill callback
        that decides to place an order), the function is executed
        directly instead of marshaled - marshaling from the loop thread
        to itself and blocking would deadlock.
        """
        if self.is_owner_thread():
            return fn(*args, **kwargs)

        # Raise the not-started error BEFORE creating the coroutine so a
        # misconfigured caller doesn't also leak an un-awaited coroutine.
        _ = self.loop

        async def _invoke() -> Any:
            return fn(*args, **kwargs)

        label = getattr(fn, "__qualname__", None) or repr(fn)
        return self._await(_invoke(), timeout=timeout, label=label)

    def run_coro(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        label: str = "coroutine",
    ) -> Any:
        """Run a coroutine on the loop thread; block for its result.

        For genuinely-async ib_insync entry points (``connectAsync``).
        Must not be called from the loop thread (would deadlock).
        """
        if self.is_owner_thread():
            raise RuntimeError("run_coro() must not be called from the loop thread (deadlock)")
        return self._await(coro, timeout=timeout, label=label)

    def _await(self, coro: Coroutine[Any, Any, Any], *, timeout: float, label: str) -> Any:
        loop = self.loop
        cfut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return cfut.result(timeout)
        except FuturesTimeoutError:
            cfut.cancel()
            raise ExecMarshalTimeout(
                f"{label} did not complete within {timeout:.1f}s on IB loop {self._name!r}; "
                "the loop may be wedged - treat the process as unhealthy"
            ) from None


_STOP = object()  # FillDispatcher shutdown sentinel


class FillDispatcher:
    """Delivers fill events on a dedicated thread, off the IB loop.

    ``submit()`` is called from the IB loop thread (ib_insync event
    handlers) and never blocks; ``deliver`` runs on this dispatcher's
    own thread so callbacks may take strategy locks and perform I/O
    without ever wedging the IB loop. Single-threaded delivery
    preserves event order. Exceptions in ``deliver`` are logged and the
    thread keeps running.
    """

    def __init__(self, deliver: Callable[[dict[str, Any]], None], *, name: str = "ib-fill-dispatch") -> None:
        self._deliver = deliver
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def submit(self, evt: dict[str, Any]) -> None:
        """Enqueue one fill event. Non-blocking; safe from any thread."""
        self._queue.put(evt)

    def stop(self, *, timeout: float = _STOP_JOIN_TIMEOUT_SECONDS) -> None:
        """Drain queued events, then stop the thread. Idempotent."""
        if not self._started:
            return
        self._queue.put(_STOP)
        self._thread.join(timeout)
        if self._thread.is_alive():
            _LOG.error("FillDispatcher did not stop within %.1fs; leaking daemon thread", timeout)

    def is_dispatch_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self._deliver(item)
            except Exception:
                _LOG.exception("fill delivery raised; event dropped: %r", item)


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_CALL_TIMEOUT_SECONDS",
    "ExecMarshalTimeout",
    "FillDispatcher",
    "IBLoopThread",
]
