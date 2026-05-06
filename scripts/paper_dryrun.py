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
``BUS_REDIS_URL``            Optional. Redis URL for bus-consumer mode
                             (e.g. ``redis://localhost:6379/0``). When
                             set, bars + breadth are read from the bus
                             instead of subscribing directly to IBKR.
                             Leave unset for direct-IBKR mode
                             (backward-compatible default).
``RUNS_DIR``                 Optional. Directory for trade-record
                             parquet output. Default
                             ``/app/runs/paper-live``. Set to a temp
                             path in tests/CI. Empty string disables
                             trade logging.
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
from pathlib import Path
from types import FrameType
from typing import Any

import pandas as pd
from prometheus_client import start_http_server

from alpha_assay.bus.consumer import Consumer
from alpha_assay.bus.streams import stream_name_for_bars, stream_name_for_ticks
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from alpha_assay.exec.ibkr import ExecMode, IBKRExecAdapter
from alpha_assay.exec.trade_log import TradeLog, TradeRecord
from alpha_assay.observability import metrics as M

HEARTBEAT_INTERVAL_SECONDS = 30
DRAIN_TIMEOUT_SECONDS = 20
DEFAULT_ES_EXPIRY = "20260618"  # ESM6 (June 2026 E-mini S&P); documented staleness risk on roll.

# Default runs directory inside the container. Override via RUNS_DIR env var.
# Outside container (local dev / CI) set RUNS_DIR to a temp path.
DEFAULT_RUNS_DIR = "/app/runs/paper-live"

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
    # Bus-consumer mode: if set, reads bars/ticks from the Redis bus
    # instead of subscribing directly to IBKR. Format: redis://host:port/db
    # or redis://user:password@host:port/db. Leave empty for direct-IBKR mode.
    bus_redis_url: str = ""
    # Root directory for trade-record output. Set to a temp path in tests.
    # Empty string disables trade logging (no parquet written).
    runs_dir: str = DEFAULT_RUNS_DIR


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
        bus_redis_url=os.environ.get("BUS_REDIS_URL", ""),
        runs_dir=os.environ.get("RUNS_DIR", DEFAULT_RUNS_DIR),
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

    The optional ``trade_log`` argument wires in per-trade parquet
    emission. When ``decide`` returns a non-zero signal (which this
    class never does, but subclasses may), ``_maybe_emit_trade`` writes
    a :class:`~alpha_assay.exec.trade_log.TradeRecord` to the log.
    """

    def __init__(
        self,
        *,
        exec_adapter: IBKRExecAdapter | object,
        trade_log: TradeLog | None = None,
    ) -> None:
        self._exec_adapter = exec_adapter
        self._trade_log = trade_log
        self.bars_seen = 0
        self.ticks_seen = 0
        self.disconnect_count = 0
        # Running mock account balance for P&L tracking.
        self._account_balance = 100_000.0

    def decide(self, _bar: dict[str, Any]) -> int:
        """Return 0 unconditionally. The always-flat invariant."""
        return 0

    def _maybe_emit_trade(self, bar: dict[str, Any], signal: int) -> None:
        """Emit a TradeRecord if a trade_log is configured and signal != 0.

        Called from ``on_bar`` after ``decide``. This path is never
        exercised by ``AlwaysFlatStrategy`` itself (``decide`` always
        returns 0), but is available for subclasses that override
        ``decide`` with real signal logic.
        """
        if self._trade_log is None or signal == 0:
            return
        close = float(bar.get("close", 0.0))
        # Mock fill: assume market-on-open, fill at close + 0.25 tick slippage.
        tick_slip = 0.25
        fill = close + tick_slip if signal > 0 else close - tick_slip
        # Mock ATR-based stop/target (placeholder - 2-point hard stop, 4-point target).
        atr_stop = 2.0
        atr_target = 4.0
        stop = fill - atr_stop if signal > 0 else fill + atr_stop
        target = fill + atr_target if signal > 0 else fill - atr_target
        # P&L is zero at entry; will be updated on exit (future extension).
        record = TradeRecord(
            timestamp=(
                pd.Timestamp(bar.get("timestamp", "")).tz_convert("UTC")
                if isinstance(bar.get("timestamp"), (pd.Timestamp,))
                else pd.Timestamp(str(bar.get("timestamp", "")), tz="UTC")
            ),
            signal_type="long_entry" if signal > 0 else "short_entry",
            entry_price=close,
            stop=stop,
            target=target,
            mock_fill_price=fill,
            mock_pnl_dollars=0.0,
            account_balance_after=self._account_balance,
        )
        self._trade_log.write(record)
        _LOG.info(
            "trade emitted: signal=%d fill=%.2f stop=%.2f target=%.2f",
            signal,
            fill,
            stop,
            target,
        )

    def on_bar(self, bar: dict[str, Any], *, feed_label: str) -> None:
        """Handle one ES bar event.

        Increments the per-feed bar counter and consults ``decide``
        only as a sanity check that the always-flat invariant holds.
        For subclasses that override ``decide``, signals are passed
        through to ``_maybe_emit_trade``.
        """
        self.bars_seen += 1
        M.bars_processed_total.labels(feed=feed_label).inc()
        sig = self.decide(bar)
        if sig != 0:
            # In this class, this branch is a paranoia guard (always-flat invariant).
            # Subclasses that override decide() with real signal logic will reach here.
            _LOG.error("always-flat invariant violation: decide returned %s; refusing to act", sig)
        self._maybe_emit_trade(bar, sig)

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
# Bus-consumer helpers
# ----------------------------------------------------------------------


def _build_bus_consumers(
    cfg: DryrunConfig,
    redis_client: Any,
) -> tuple[Consumer, Consumer]:
    """Build bar + breadth bus consumers for the given config.

    Returns ``(bars_consumer, breadth_consumer)``. Both use
    ``start_id="$"`` so they consume only new messages (latest-only
    mode for the paper-trader - we don't replay historical bars).
    """
    spec = es_contract_spec(cfg)
    bars_stream = stream_name_for_bars(spec)
    breadth_stream = stream_name_for_ticks("TICK-NYSE")
    bars_consumer = Consumer(
        redis_client=redis_client,
        stream=bars_stream,
        consumer_id="paper-trader-bars",
        start_id="$",
    )
    breadth_consumer = Consumer(
        redis_client=redis_client,
        stream=breadth_stream,
        consumer_id="paper-trader-breadth",
        start_id="$",
    )
    return bars_consumer, breadth_consumer


def _consume_bars_from_bus_sync(
    consumer: Consumer,
    strategy: AlwaysFlatStrategy,
    stop_event: threading.Event,
) -> None:
    """Synchronous bus-consumer loop for ES bars.

    Runs in an executor thread. iter_messages returns on block timeout
    (no messages within block_ms), so wrap in an outer while loop that
    keeps re-entering iter_messages until stop_event is set.
    """
    feed_label = "es"
    while not stop_event.is_set():
        for msg in consumer.iter_messages(block_ms=1000):
            if stop_event.is_set():
                return
            try:
                bar = {
                    "timestamp": pd.Timestamp(msg.payload["ts_minute_utc"], unit="s", tz="UTC"),
                    "open": msg.payload["open"],
                    "high": msg.payload["high"],
                    "low": msg.payload["low"],
                    "close": msg.payload["close"],
                    "volume": msg.payload["volume"],
                }
            except KeyError:
                _LOG.error("paper-trader bars: skipping malformed payload=%r", msg.payload)
                continue
            strategy.on_bar(bar, feed_label=feed_label)


def _consume_breadth_from_bus_sync(
    consumer: Consumer,
    strategy: AlwaysFlatStrategy,
    stop_event: threading.Event,
) -> None:
    """Synchronous bus-consumer loop for TICK-NYSE breadth."""
    feed_label = "tick_nyse"
    while not stop_event.is_set():
        for msg in consumer.iter_messages(block_ms=1000):
            if stop_event.is_set():
                return
            try:
                tick = {
                    "timestamp": pd.Timestamp(msg.ts_event_ns, unit="ns", tz="UTC"),
                    "value": msg.payload["value"],
                    "symbol": msg.payload.get("symbol", "TICK-NYSE"),
                }
            except KeyError:
                _LOG.error("paper-trader breadth: skipping malformed payload=%r", msg.payload)
                continue
            strategy.on_breadth_tick(tick, feed_label=feed_label)


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
    adapter: IBKRAdapter | None,
    stop_event: asyncio.Event,
) -> None:
    """One-line stdout heartbeat every HEARTBEAT_INTERVAL_SECONDS.

    ``adapter`` may be None in bus-consumer mode; IBKR connectivity is
    reported as ``n/a`` in that case since the paper-trader is no longer
    a direct IBKR subscriber.
    """
    while not stop_event.is_set():
        # Sleep in 1s slices so SIGTERM is responsive.
        for _ in range(HEARTBEAT_INTERVAL_SECONDS):
            if stop_event.is_set():
                return
            await asyncio.sleep(1)
        connected = "n/a(bus)" if adapter is None else ("yes" if adapter.is_connected else "no")
        print(
            "paper-dryrun heartbeat: "
            f"bars={strategy.bars_seen} ticks={strategy.ticks_seen} "
            f"disconnects={strategy.disconnect_count} ibkr_connected={connected}",
            flush=True,
        )


async def _async_main(
    cfg: DryrunConfig,
    adapter: IBKRAdapter | None,
    strategy: AlwaysFlatStrategy,
    stop_event_thread: threading.Event,
    bus_consumers: tuple[Consumer, Consumer] | None = None,
) -> int:
    """Async portion of the dry-run loop.

    If ``bus_consumers`` is provided the bar + breadth streams are read
    from the bus (executor threads). Otherwise the direct-IBKR path is
    used (``adapter`` must not be None in that case).
    """
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

    if bus_consumers is not None:
        # Bus-consumer mode: run sync loops in executor threads.
        # run_in_executor returns a Future (already awaitable); wrap with
        # ensure_future so it composes with asyncio.wait below.
        bars_consumer, breadth_consumer = bus_consumers
        loop = asyncio.get_running_loop()

        bars_task = asyncio.ensure_future(
            loop.run_in_executor(
                None,
                _consume_bars_from_bus_sync,
                bars_consumer,
                strategy,
                stop_event_thread,
            )
        )
        breadth_task = asyncio.ensure_future(
            loop.run_in_executor(
                None,
                _consume_breadth_from_bus_sync,
                breadth_consumer,
                strategy,
                stop_event_thread,
            )
        )
        # Heartbeat loop needs a mock adapter-like object for connectivity reporting.
        # In bus mode, IBKR connectivity is not directly observable - use None sentinel.
        heartbeat_task = asyncio.create_task(_heartbeat_loop(strategy, None, stop_event))
    else:
        assert adapter is not None, "adapter required for direct-IBKR mode"
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

        # Flush trade log on shutdown so in-memory records are persisted.
        if strategy._trade_log is not None:
            try:
                strategy._trade_log.flush()
                _LOG.info("trade log flushed on shutdown")
            except Exception:
                _LOG.exception("trade log flush failed on shutdown; records may be lost")

    return 0


def run(cfg: DryrunConfig) -> int:
    """Synchronous entrypoint. Sets up logging, metrics, signals,
    constructs adapters, and runs the async loop.

    Returns the process exit code (always 0 on clean drain).

    Bus-consumer mode is active when ``cfg.bus_redis_url`` is set.
    Direct-IBKR mode is used otherwise (backward-compatible default).
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOG.info(
        "paper-dryrun starting: ibkr=%s:%d client_id=%d metrics_port=%d es_expiry=%s "
        "duration_seconds=%d bus_mode=%s",
        cfg.ibkr_host,
        cfg.ibkr_port,
        cfg.ibkr_client_id,
        cfg.metrics_port,
        cfg.es_expiry,
        cfg.duration_seconds,
        "yes" if cfg.bus_redis_url else "no",
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

    # Construct TradeLog if RUNS_DIR is configured.
    trade_log: TradeLog | None = None
    if cfg.runs_dir:
        trade_log = TradeLog(out_dir=Path(cfg.runs_dir))
        _LOG.info("trade log enabled at %s/trades.parquet", cfg.runs_dir)

    if cfg.bus_redis_url:
        # Bus-consumer mode: no IBKR connection needed.
        import redis as redis_pkg

        redis_client = redis_pkg.from_url(cfg.bus_redis_url)
        # Build a stub exec adapter (never used in always-flat, but satisfies the
        # constructor contract so the compose stack remains uniform).
        exec_adapter_stub = object()
        strategy = AlwaysFlatStrategy(exec_adapter=exec_adapter_stub, trade_log=trade_log)
        bus_consumers = _build_bus_consumers(cfg, redis_client)
        _LOG.info(
            "bus-consumer mode active; bars stream=%s breadth stream=%s",
            bus_consumers[0]._stream,
            bus_consumers[1]._stream,
        )
        return asyncio.run(_async_main(cfg, None, strategy, stop_event, bus_consumers=bus_consumers))

    # Direct-IBKR mode (backward-compatible default).
    adapter, exec_adapter = build_adapters(cfg)
    strategy = AlwaysFlatStrategy(exec_adapter=exec_adapter, trade_log=trade_log)

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
