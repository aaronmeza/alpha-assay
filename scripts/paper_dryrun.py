# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""always-flat IBKR paper dry-run entrypoint.

This script supersedes the ``scripts/paper_trader_stub.py``
heartbeat-only placeholder once IBKR creds are wired up. It:

- Connects to a real IBKR paper account via ``IBKRAdapter`` (read path,
  ) configured with ``read_only=True``.
- Wraps the read adapter in ``IBKRExecAdapter`` with ``mode=PAPER`` and
  ``dry_run=True``. The wrap is defense-in-depth: the always-flat
  strategy below NEVER decides to enter, so ``place_bracket_order`` is
  never invoked even before ``dry_run`` would gate it.
- Subscribes to ES futures 1-min bars (``CME``, RTH-only) AND
  ``TICK-NYSE`` breadth via the read adapter.
- Increments ``alpha_assay_bars_processed_total{feed="es"}`` per bar
  and ``alpha_assay_bars_processed_total{feed="tick_nyse"}`` per
  breadth tick.
- Exposes a Prometheus ``/metrics`` endpoint on ``METRICS_PORT``.
- Emits a stdout heartbeat every 30s with bar / tick counts and IBKR
  connection state.
- Drains gracefully on SIGTERM / SIGINT within
  ``DRAIN_TIMEOUT_SECONDS`` (default 20s).

The script is the always-flat paper dry-run; it replaces
``paper_trader_stub.py`` heartbeat-only stub when IBKR creds are
wired up. It NEVER submits orders. Unit invariants live in
``tests/test_paper_dryrun_unit.py``; deployment-host verification
lives in ``tests/integration/test_e2e_paper_dryrun.py`` (opt-in via
``RUN_LIVE_E2E=1``).

Environment
-----------

``IBKR_HOST``                Default ``127.0.0.1``.
``IBKR_PORT``                Default ``4002`` (IB Gateway paper
                             headless). The four-port matrix:
                             4002 = Gateway paper, 4001 = Gateway live,
                             7497 = TWS paper, 7496 = TWS live. the deployment host
                             runs Gateway headless on 4002.
``IBKR_CLIENT_ID``           Default ``1``. Must be unique per IBKR
                             connection; the breadth recorder defaults
                             to 21 to avoid collision.
``IBKR_ACCOUNT``             Optional. Empty string defers to the
                             default account associated with the
                             connection.
``METRICS_PORT``             Default ``8000``. The the deployment host compose binds
                             host ``18000 -> container 8000``.
``DRYRUN_DURATION_SECONDS``  Default ``0`` (run until SIGTERM /
                             SIGINT). Set to a positive int for
                             time-bounded runs (used by the integration
                             test).
``ES_EXPIRY``                Front-month contract code, ``YYYYMMDD``.
                             Hardcoded fallback ``20260618`` (ESM6,
                             E-mini S&P June 2026; verified via
                             ContFuture qualify on 2026-04-28). The
                             short YYYYMM form is rejected by IBKR with
                             "No security definition has been found".
                             Update on each quarterly roll; this is a
                             documented staleness risk.

The script is invoked directly by the the deployment host ``paper-trader`` compose
service. No CLI parser; everything is env-driven so the compose file
stays the canonical config surface.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from types import FrameType
from typing import Any

from prometheus_client import start_http_server

from alpha_assay.data.ibkr_adapter import IBKRAdapter
from alpha_assay.exec.ibkr import ExecMode, IBKRExecAdapter
from alpha_assay.observability import metrics as M

HEARTBEAT_INTERVAL_SECONDS = 30
DRAIN_TIMEOUT_SECONDS = 20
DEFAULT_ES_EXPIRY = "20260618"  # ESM6 (June 2026 E-mini S&P); documented staleness risk on roll.

_LOG = logging.getLogger("alpha_assay.paper_dryrun")


@dataclass(frozen=True)
class DryrunConfig:
    """All env-resolved configuration for the dry-run.

    Built by :func:`load_config_from_env`. Treat as immutable; the
    main loop does not mutate any of these fields after startup.
    """

    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int
    ibkr_account: str
    metrics_port: int
    es_expiry: str
    duration_seconds: int


def load_config_from_env() -> DryrunConfig:
    """Resolve all dry-run configuration from environment variables.

    See module docstring for the per-variable defaults and meaning.
    """
    return DryrunConfig(
        ibkr_host=os.environ.get("IBKR_HOST", "127.0.0.1"),
        ibkr_port=int(os.environ.get("IBKR_PORT", "4002")),
        ibkr_client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")),
        ibkr_account=os.environ.get("IBKR_ACCOUNT", ""),
        metrics_port=int(os.environ.get("METRICS_PORT", "8000")),
        es_expiry=os.environ.get("ES_EXPIRY", DEFAULT_ES_EXPIRY),
        duration_seconds=int(os.environ.get("DRYRUN_DURATION_SECONDS", "0")),
    )


def es_contract_spec(cfg: DryrunConfig) -> dict[str, Any]:
    """Build the canonical ES futures contract spec for IBKRAdapter.

    See ``alpha_assay.data.ibkr_adapter._build_contract`` for the
    schema; ``expiry`` maps to ``lastTradeDateOrContractMonth`` for
    FUT contracts.
    """
    return {
        "symbol": "ES",
        "sec_type": "FUT",
        "exchange": "CME",
        "currency": "USD",
        "expiry": cfg.es_expiry,
    }


class AlwaysFlatStrategy:
    """The always-flat dry-run strategy.

    Increments observability counters per bar / per breadth tick but
    never enters a position. ``decide`` always returns 0. The
    ``exec_adapter`` is captured for completeness (so the compose-stack
    contract is identical to the real paper-trader) but
    ``place_bracket_order`` is NEVER invoked from this class.

    The ``feed_label`` argument to ``on_bar`` and ``on_breadth_tick``
    is the Prometheus label value (NOT the raw IBKR feed name): we use
    short, stable labels (``"es"``, ``"tick_nyse"``) so dashboards and
    alerts can target a known set without depending on contract roll.
    """

    def __init__(self, *, exec_adapter: IBKRExecAdapter | object) -> None:
        self._exec_adapter = exec_adapter
        self.bars_seen = 0
        self.ticks_seen = 0
        self.disconnect_count = 0

    def decide(self, _bar: dict[str, Any]) -> int:
        """Return 0 unconditionally. The always-flat invariant."""
        return 0

    def on_bar(self, bar: dict[str, Any], *, feed_label: str) -> None:
        """Handle one ES bar event.

        Increments the per-feed bar counter and consults ``decide``
        only as a sanity check that the always-flat invariant holds.
        """
        self.bars_seen += 1
        M.bars_processed_total.labels(feed=feed_label).inc()
        # Pure paranoia: assert the always-flat invariant inline. If
        # someone subclasses AlwaysFlatStrategy and breaks decide(),
        # this will surface in the heartbeat logs without ever
        # reaching place_bracket_order.
        sig = self.decide(bar)
        if sig != 0:
            _LOG.error("always-flat invariant violation: decide returned %s; refusing to act", sig)

    def on_breadth_tick(self, _tick: dict[str, Any], *, feed_label: str) -> None:
        """Handle one breadth tick. Increments the per-feed counter."""
        self.ticks_seen += 1
        M.bars_processed_total.labels(feed=feed_label).inc()

    def on_disconnect(self) -> None:
        """Record a disconnect event. Idempotent / non-raising so the
        heartbeat loop can keep running while the adapter reconnects.
        """
        self.disconnect_count += 1
        _LOG.warning(
            "ibkr disconnect observed; count=%d (heartbeat continues, reconnect deferred to " "ib_insync)",
            self.disconnect_count,
        )


def build_adapters(cfg: DryrunConfig) -> tuple[IBKRAdapter, IBKRExecAdapter]:
    """Construct the read + exec adapters from a resolved config.

    The exec adapter uses ``mode=PAPER`` and ``dry_run=True``; both
    are belt-and-suspenders since the always-flat strategy never
    submits an order.
    """
    adapter = IBKRAdapter(
        host=cfg.ibkr_host,
        port=cfg.ibkr_port,
        client_id=cfg.ibkr_client_id,
        account=cfg.ibkr_account or None,
        read_only=True,
    )
    exec_adapter = IBKRExecAdapter(
        adapter=adapter,
        mode=ExecMode.PAPER,
        dry_run=True,
    )
    return adapter, exec_adapter


# ----------------------------------------------------------------------


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        _LOG.info("paper-dryrun received signal %d; requesting stop", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


async def _consume_bars(
    adapter: IBKRAdapter,
    spec: dict[str, Any],
    strategy: AlwaysFlatStrategy,
    stop_event: asyncio.Event,
) -> None:
    """Async-iterate ES bars and feed them to the strategy until the
    stop event fires."""
    feed_label = "es"
    try:
        gen = adapter.subscribe_bars(spec, bar_size="1 min", what_to_show="TRADES")
        async for bar in gen:
            if stop_event.is_set():
                await gen.aclose()
                return
            strategy.on_bar(bar, feed_label=feed_label)
    except asyncio.CancelledError:
        raise
    except Exception:
        _LOG.exception("ES bar subscription crashed; signaling disconnect")
        strategy.on_disconnect()


async def _consume_breadth(
    adapter: IBKRAdapter,
    symbol: str,
    strategy: AlwaysFlatStrategy,
    stop_event: asyncio.Event,
) -> None:
    feed_label = "tick_nyse"
    try:
        gen = adapter.subscribe_breadth(symbol=symbol)
        async for tick in gen:
            if stop_event.is_set():
                await gen.aclose()
                return
            strategy.on_breadth_tick(tick, feed_label=feed_label)
    except asyncio.CancelledError:
        raise
    except Exception:
        _LOG.exception("breadth subscription crashed; signaling disconnect")
        strategy.on_disconnect()


async def _heartbeat_loop(
    strategy: AlwaysFlatStrategy,
    adapter: IBKRAdapter,
    stop_event: asyncio.Event,
) -> None:
    """One-line stdout heartbeat every HEARTBEAT_INTERVAL_SECONDS."""
    while not stop_event.is_set():
        # Sleep in 1s slices so SIGTERM is responsive.
        for _ in range(HEARTBEAT_INTERVAL_SECONDS):
            if stop_event.is_set():
                return
            await asyncio.sleep(1)
        connected = "yes" if adapter.is_connected else "no"
        print(
            "paper-dryrun heartbeat: "
            f"bars={strategy.bars_seen} ticks={strategy.ticks_seen} "
            f"disconnects={strategy.disconnect_count} ibkr_connected={connected}",
            flush=True,
        )


async def _async_main(
    cfg: DryrunConfig,
    adapter: IBKRAdapter,
    strategy: AlwaysFlatStrategy,
    stop_event_thread: threading.Event,
) -> int:
    """Async portion of the dry-run loop."""
    spec = es_contract_spec(cfg)
    stop_event = asyncio.Event()

    async def _bridge_stop() -> None:
        # Bridge the threading.Event (set by signal handlers in the
        # main thread) into the asyncio.Event used by the consumer
        # tasks. Polling at 200ms keeps shutdown latency comfortably
        # under the 20s drain budget.
        while not stop_event_thread.is_set():
            await asyncio.sleep(0.2)
        stop_event.set()

    bars_task = asyncio.create_task(_consume_bars(adapter, spec, strategy, stop_event))
    breadth_task = asyncio.create_task(_consume_breadth(adapter, "TICK-NYSE", strategy, stop_event))
    heartbeat_task = asyncio.create_task(_heartbeat_loop(strategy, adapter, stop_event))
    bridge_task = asyncio.create_task(_bridge_stop())

    deadline_task: asyncio.Task[None] | None = None
    if cfg.duration_seconds > 0:

        async def _deadline() -> None:
            await asyncio.sleep(cfg.duration_seconds)
            stop_event.set()

        deadline_task = asyncio.create_task(_deadline())

    try:
        # Block until stop signaled.
        await stop_event.wait()
    finally:
        for t in (bars_task, breadth_task, heartbeat_task, bridge_task):
            t.cancel()
        if deadline_task is not None:
            deadline_task.cancel()
        # Drain with a hard deadline so SIGTERM honors DRAIN_TIMEOUT_SECONDS.
        pending = [bars_task, breadth_task, heartbeat_task, bridge_task]
        if deadline_task is not None:
            pending.append(deadline_task)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=DRAIN_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            _LOG.warning("drain timeout exceeded; some tasks may have leaked")

    return 0


def run(cfg: DryrunConfig) -> int:
    """Synchronous entrypoint. Sets up logging, metrics, signals,
    constructs adapters, and runs the async loop.

    Returns the process exit code (always 0 on clean drain).
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOG.info(
        "paper-dryrun starting: ibkr=%s:%d client_id=%d metrics_port=%d es_expiry=%s " "duration_seconds=%d",
        cfg.ibkr_host,
        cfg.ibkr_port,
        cfg.ibkr_client_id,
        cfg.metrics_port,
        cfg.es_expiry,
        cfg.duration_seconds,
    )
    if cfg.es_expiry == DEFAULT_ES_EXPIRY:
        _LOG.warning(
            "ES_EXPIRY not set; using hardcoded fallback %s. Update on quarterly roll.",
            DEFAULT_ES_EXPIRY,
        )

    # Start metrics endpoint BEFORE attempting the IBKR connect so
    # health probes succeed even if IBKR is unreachable.
    start_http_server(cfg.metrics_port)

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    adapter, exec_adapter = build_adapters(cfg)
    strategy = AlwaysFlatStrategy(exec_adapter=exec_adapter)

    # Best-effort connect. If the connection fails (no Gateway up) we
    # still serve /metrics so health probes can detect the issue and
    # the operator can SSH in. The strategy still runs; it just sees
    # zero events until the gateway comes back.
    try:
        adapter.connect()
    except Exception:
        _LOG.exception("initial ibkr connect failed; continuing with metrics-only loop")
        strategy.on_disconnect()

    try:
        return asyncio.run(_async_main(cfg, adapter, strategy, stop_event))
    finally:
        try:
            adapter.disconnect()
        except Exception:
            _LOG.exception("error during ibkr disconnect; ignoring")


def main() -> int:
    cfg = load_config_from_env()
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
