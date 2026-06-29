# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""paper_dryrun exits EXIT_RESTART when the IBKR order-path connection drops.

The 10-day silent outage (alphaassay-0n6): a peer-closed socket (the nightly IB
Gateway / IBC restart) left the order path dead, but the process had no watchdog,
so it never exited and Docker never reconnected it - and the ibkr_connected gauge
stayed stuck at 1, so no alert could fire. _async_main must now:
  - exit EXIT_RESTART when the connection is lost (so `restart: unless-stopped`
    reconnects on a fresh cold start), and
  - mark the gauge down (the gauge no longer lies while the order path is dead),
while still exiting EXIT_OK on a clean stop.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from pathlib import Path

import fakeredis
from prometheus_client import REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "paper_dryrun.py"
_MODULE_NAME = "paper_dryrun"


def _load_script_module():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    # ib_insync/eventkit call the deprecated asyncio.get_event_loop() at import.
    # If an earlier asyncio.run()-based test (e.g. the watchdog units) tore down
    # the main-thread loop, that import-time call raises - ensure a loop first so
    # first-loading this script module is order-independent.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class _FakeAdapter:
    """Minimal adapter surface the watchdog + heartbeat read."""

    def __init__(self, *, connected: bool) -> None:
        self.is_connected = connected


def _cfg(module):
    return module.DryrunConfig(
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_client_id=1,
        ibkr_account="",
        metrics_port=8000,
        es_expiry="20260918",
        duration_seconds=0,
        bus_redis_url="redis://localhost:6379/0",
    )


def _idle_consume(_consumer, _strategy, stop_event, *_args):
    """Stand-in for the bus-consumer loops: idle until stop, independent of any
    fakeredis blocking quirks so the test isolates the watchdog race."""
    while not stop_event.is_set():
        time.sleep(0.01)


def _gauge():
    return REGISTRY.get_sample_value("alpha_assay_ibkr_connected")


def _run_async_main(module, adapter, stop_event_thread, *, trip):
    redis_client = fakeredis.FakeRedis()
    bus_consumers = module._build_bus_consumers(_cfg(module), redis_client)
    strategy = module.AlwaysFlatStrategy(exec_adapter=object(), trade_log=None)

    async def _run():
        asyncio.create_task(_trip_later())
        return await asyncio.wait_for(
            module._async_main(
                _cfg(module),
                adapter,
                strategy,
                stop_event_thread,
                bus_consumers=bus_consumers,
            ),
            timeout=8.0,
        )

    async def _trip_later():
        await asyncio.sleep(0.05)
        trip()

    return asyncio.run(_run())


def test_async_main_exits_restart_and_marks_gauge_down_on_connection_loss(monkeypatch):
    # Given a connected order path, When the IBKR connection drops mid-run, Then
    # _async_main returns EXIT_RESTART and the gauge reads 0.
    module = _load_script_module()
    monkeypatch.setattr(module, "WATCHDOG_POLL_SECONDS", 0.02)
    monkeypatch.setattr(module, "_consume_bars_from_bus_sync", _idle_consume)
    monkeypatch.setattr(module, "_consume_breadth_from_bus_sync", _idle_consume)

    module.M.ibkr_connected.set(1)
    adapter = _FakeAdapter(connected=True)
    stop_event_thread = threading.Event()

    rc = _run_async_main(module, adapter, stop_event_thread, trip=lambda: setattr(adapter, "is_connected", False))

    assert rc == module.EXIT_RESTART
    assert _gauge() == 0.0


def test_async_main_exits_ok_on_clean_stop(monkeypatch):
    # Given a healthy connection, When a clean stop is requested, Then _async_main
    # returns EXIT_OK (the watchdog must not turn a graceful shutdown into a restart).
    module = _load_script_module()
    monkeypatch.setattr(module, "WATCHDOG_POLL_SECONDS", 0.02)
    monkeypatch.setattr(module, "_consume_bars_from_bus_sync", _idle_consume)
    monkeypatch.setattr(module, "_consume_breadth_from_bus_sync", _idle_consume)

    adapter = _FakeAdapter(connected=True)
    stop_event_thread = threading.Event()

    rc = _run_async_main(module, adapter, stop_event_thread, trip=stop_event_thread.set)

    assert rc == module.EXIT_OK


def test_async_main_clean_stop_wins_concurrent_connection_loss(monkeypatch):
    # Finding F: a stop requested via the threading event must win even when the
    # connection is already lost and the threading->async bridge has not yet fired,
    # so a clean shutdown is never misreported as a restart.
    module = _load_script_module()
    monkeypatch.setattr(module, "WATCHDOG_POLL_SECONDS", 0.02)
    monkeypatch.setattr(module, "_consume_bars_from_bus_sync", _idle_consume)
    monkeypatch.setattr(module, "_consume_breadth_from_bus_sync", _idle_consume)

    adapter = _FakeAdapter(connected=False)  # connection already lost -> wd_task ready at once
    stop_event_thread = threading.Event()
    stop_event_thread.set()  # stop requested concurrently

    redis_client = fakeredis.FakeRedis()
    bus_consumers = module._build_bus_consumers(_cfg(module), redis_client)
    strategy = module.AlwaysFlatStrategy(exec_adapter=object(), trade_log=None)
    rc = asyncio.run(
        asyncio.wait_for(
            module._async_main(_cfg(module), adapter, strategy, stop_event_thread, bus_consumers=bus_consumers),
            timeout=8.0,
        )
    )
    assert rc == module.EXIT_OK


def test_async_main_direct_mode_does_not_arm_watchdog(monkeypatch):
    # Finding G: legacy direct-IBKR mode (no bus_consumers) must NOT arm the
    # watchdog - a gateway-down cold start serves metrics and waits for a clean
    # stop rather than restart-looping. If the watchdog were armed, a disconnected
    # adapter would return EXIT_RESTART immediately; we assert EXIT_OK on stop.
    module = _load_script_module()
    monkeypatch.setattr(module, "WATCHDOG_POLL_SECONDS", 0.02)

    async def _idle_async(*_a, **_k):
        while True:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(module, "_consume_bars", _idle_async)
    monkeypatch.setattr(module, "_consume_breadth", _idle_async)

    adapter = _FakeAdapter(connected=False)  # gateway down at cold start
    stop_event_thread = threading.Event()
    strategy = module.AlwaysFlatStrategy(exec_adapter=object(), trade_log=None)

    async def _run():
        async def _stop_later():
            await asyncio.sleep(0.1)
            stop_event_thread.set()

        asyncio.create_task(_stop_later())
        return await asyncio.wait_for(
            module._async_main(_cfg(module), adapter, strategy, stop_event_thread),  # no bus_consumers -> direct mode
            timeout=8.0,
        )

    assert asyncio.run(_run()) == module.EXIT_OK


def test_connect_exec_with_retry_succeeds_after_transient_failures(monkeypatch):
    # Finding E: the cold-start retry must actually be exercised - succeeds once a
    # later attempt connects (gateway came back mid-IBC-cycle).
    module = _load_script_module()
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    adapter = _FakeAdapter(connected=False)
    calls = {"n": 0}

    class _Exec:
        def connect(self):
            calls["n"] += 1
            if calls["n"] >= 3:
                adapter.is_connected = True
            else:
                raise ConnectionError("gateway down")

    assert module._connect_exec_with_retry(_Exec(), adapter) is True
    assert calls["n"] == 3


def test_connect_exec_with_retry_returns_false_when_exhausted(monkeypatch):
    # All attempts fail -> returns False (caller continues metrics-only; the
    # watchdog/Docker loop recovers when the gateway returns).
    module = _load_script_module()
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    adapter = _FakeAdapter(connected=False)

    class _Exec:
        def connect(self):
            raise ConnectionError("gateway down")

    assert module._connect_exec_with_retry(_Exec(), adapter) is False
