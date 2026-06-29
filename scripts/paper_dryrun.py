# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""IBKR paper-trader entrypoint (always-flat by default).

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
wired up. By default it NEVER submits orders. Unit invariants live in
``tests/test_paper_dryrun_unit.py``; deployment-host verification
lives in ``tests/integration/test_e2e_paper_dryrun.py`` (opt-in via
``RUN_LIVE_E2E=1``).

Strategy mode (opt-in)
----------------------

When ``PAPER_STRATEGY`` is set (bus-consumer mode only), the script
hosts a real `BaseStrategy` subclass via
``alpha_assay.paper.PaperStrategyRunner``: the joined ES + breadth
minute frame is built incrementally from the bus, signals are executed
as PAPER bracket orders through ``IBKRExecAdapter``, and realized P&L
is written to the trade log on every round-trip exit. Live mode stays
triple-gated behind the three-lock interlock regardless; this flag only
enables *paper* submission. When ``PAPER_STRATEGY`` is unset the
behavior is exactly the always-flat dry-run described above.

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
``ES_EXPIRY``                Emergency override for the front-month
                             contract code, ``YYYYMMDD``. When set,
                             this value is used directly and Redis is
                             not consulted. When unset (normal
                             operation), the resolved expiry is read
                             from the Redis metadata key written by
                             ibkr-feed at startup. If neither is
                             available at runtime the process fails
                             closed with ``FrontMonthMissingError``.
``PAPER_STRATEGY``           Optional. ``package.module:ClassName`` of a
                             ``BaseStrategy`` subclass to host. Unset =
                             always-flat dry-run. Requires
                             ``BUS_REDIS_URL`` (the joined frame needs
                             both breadth streams, which only the bus
                             carries).
``PAPER_STRATEGY_CONFIG``    Path to the strategy's YAML config
                             (``alpha_assay.config.loader`` schema).
                             Required when ``PAPER_STRATEGY`` is set.
``PAPER_STARTING_BALANCE``   Reference balance for risk-based sizing in
                             strategy mode. Default ``100000``.

The script is invoked directly by the the deployment host ``paper-trader`` compose
service. No CLI parser; everything is env-driven so the compose file
stays the canonical config surface.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis as redis_pkg

import pandas as pd
from prometheus_client import start_http_server

from alpha_assay.bus.consumer import Consumer
from alpha_assay.bus.streams import stream_name_for_bars, stream_name_for_ticks
from alpha_assay.data.front_month import (
    FrontMonthMissingError,
    read_front_month,
    validate_yyyymmdd,
)
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from alpha_assay.exec.ibkr import ExecMode, IBKRExecAdapter, build_exec_adapter
from alpha_assay.exec.loop_marshal import IBLoopThread
from alpha_assay.exec.trade_log import TradeLog, TradeRecord
from alpha_assay.exec.watchdog import EXIT_OK, EXIT_RESTART, watch_connection
from alpha_assay.observability import metrics as M
from alpha_assay.paper.runner import PaperStrategyRunner, load_paper_strategy

HEARTBEAT_INTERVAL_SECONDS = 30
DRAIN_TIMEOUT_SECONDS = 20
# Connection watchdog poll cadence (strategy mode order path). A peer-closed
# socket - the nightly IB Gateway / IBC restart - leaves ib_insync disconnected
# with no auto-reconnect, so the watchdog exits EXIT_RESTART and Docker
# (restart: unless-stopped) brings the process back on a fresh connect. Mirrors
# the ibkr-feed producer; see alpha_assay.exec.watchdog.
WATCHDOG_POLL_SECONDS = 5
# Cold-start connect retry: a freshly (re)started container can land mid-IBC
# cycle while IB Gateway is briefly down. Retry with backoff before giving up
# so the watchdog/Docker loop is not churned during the nightly restart window.
CONNECT_RETRY_ATTEMPTS = 5
CONNECT_RETRY_BASE_DELAY = 2.0
CONNECT_RETRY_MAX_DELAY = 30.0

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
    # None when ES_EXPIRY env var is unset; resolved to a concrete expiry
    # via _resolve_es_expiry() after the Redis client is available.
    es_expiry: str | None
    duration_seconds: int
    # Bus-consumer mode: if set, reads bars/ticks from the Redis bus
    # instead of subscribing directly to IBKR. Format: redis://host:port/db
    # or redis://user:password@host:port/db. Leave empty for direct-IBKR mode.
    bus_redis_url: str = ""
    # Root directory for trade-record output. Set to a temp path in tests.
    # Empty string disables trade logging (no parquet written).
    runs_dir: str = DEFAULT_RUNS_DIR
    # Strategy mode: "package.module:ClassName" of a BaseStrategy subclass.
    # Empty string = always-flat dry-run (backward-compatible default).
    paper_strategy: str = ""
    # YAML config path for the strategy (required when paper_strategy set).
    paper_strategy_config: str = ""
    # Reference balance for risk-based sizing in strategy mode.
    paper_starting_balance: float = 100_000.0


def load_config_from_env() -> DryrunConfig:
    """Resolve all dry-run configuration from environment variables.

    ``es_expiry`` is set to the ``ES_EXPIRY`` env var when present (emergency
    override), or ``None`` when unset. The ``None`` case means the caller must
    call ``_resolve_es_expiry(redis_client)`` after a Redis connection is
    available to populate the field via ``dataclasses.replace``.

    See module docstring for the per-variable defaults and meaning.
    """
    env_expiry = os.environ.get("ES_EXPIRY", "").strip() or None
    return DryrunConfig(
        ibkr_host=os.environ.get("IBKR_HOST", "127.0.0.1"),
        ibkr_port=int(os.environ.get("IBKR_PORT", "4002")),
        ibkr_client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")),
        ibkr_account=os.environ.get("IBKR_ACCOUNT", ""),
        metrics_port=int(os.environ.get("METRICS_PORT", "8000")),
        es_expiry=env_expiry,
        duration_seconds=int(os.environ.get("DRYRUN_DURATION_SECONDS", "0")),
        bus_redis_url=os.environ.get("BUS_REDIS_URL", ""),
        runs_dir=os.environ.get("RUNS_DIR", DEFAULT_RUNS_DIR),
        paper_strategy=os.environ.get("PAPER_STRATEGY", "").strip(),
        paper_strategy_config=os.environ.get("PAPER_STRATEGY_CONFIG", "").strip(),
        paper_starting_balance=float(os.environ.get("PAPER_STARTING_BALANCE", "100000")),
    )


def _resolve_es_expiry_with_wait(
    redis_client: redis_pkg.Redis | None,
    *,
    max_wait_seconds: float = 60.0,
    poll_interval_seconds: float = 5.0,
) -> tuple[str, str]:
    """Resolve ES expiry, polling Redis for up to max_wait_seconds.

    Env-override path is checked first (instant, validated). If env is
    unset and Redis is provided, poll the key for up to max_wait_seconds
    before giving up. Handles the cold-start race where the consumer boots
    before ibkr-feed has written the metadata key.

    Returns:
        Tuple of (expiry_value, source_label). Log source for operator
        traceability.

    Raises:
        InvalidExpiryError: env override is set but not a valid YYYYMMDD.
        FrontMonthMissingError: neither env nor Redis key available after
            max_wait_seconds.
    """
    override = os.environ.get("ES_EXPIRY", "").strip()
    if override:
        # Validate before using - bad env value must fail loudly.
        validated = validate_yyyymmdd(override, source="ES_EXPIRY env")
        return validated, "ES_EXPIRY env"
    if redis_client is None:
        raise FrontMonthMissingError(
            "ES_EXPIRY env var is unset and no Redis client (BUS_REDIS_URL) - "
            "cannot resolve ES front-month. Set ES_EXPIRY to a YYYYMMDD value, "
            "or set BUS_REDIS_URL and ensure ibkr-feed has published the "
            "front-month metadata key."
        )
    deadline = time.monotonic() + max_wait_seconds
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            # read_front_month validates the stored value internally (Fix A).
            value = read_front_month(redis_client, symbol="ES", exchange="CME")
            return value, "Redis metadata key"
        except FrontMonthMissingError as e:
            last_err = e
            time.sleep(poll_interval_seconds)
    raise FrontMonthMissingError(
        f"ES front-month not in Redis after {max_wait_seconds:.0f}s wait; "
        f"ibkr-feed may not have started or failed to resolve. Last error: {last_err}"
    )


def _resolve_es_expiry(
    redis_client: redis_pkg.Redis | None,
) -> tuple[str, str]:
    """Resolve the ES expiry for this session.

    Delegates to _resolve_es_expiry_with_wait with default timeouts.
    Kept for backward-compat with callers that use this name directly.
    """
    return _resolve_es_expiry_with_wait(redis_client)


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


def _mark_ibkr_disconnected(adapter: IBKRAdapter) -> None:
    """Reflect a watchdog-detected connection loss in the metrics.

    Mirrors ``infra.feed.run._mark_disconnected``: ``adapter.disconnect()`` is a
    no-op once ``is_connected`` is already False (the peer closed the socket -
    the nightly IB Gateway restart), so it never moves the gauge. Setting it
    explicitly is what stops ``alpha_assay_ibkr_connected`` from being stuck at 1
    while the order path is actually dead - the silent-failure that made the 10-day
    paper-trader outage invisible to every alert. The actual socket teardown is
    handled by the caller's ``finally`` (``exec_adapter.disconnect()``).
    """
    M.ibkr_connected.set(0)
    M.ibkr_connection_events_total.labels(event="disconnected").inc()


def _connect_exec_with_retry(exec_adapter: IBKRExecAdapter, adapter: IBKRAdapter) -> bool:
    """Connect the exec adapter, retrying with exponential backoff.

    A freshly (re)started container can land mid-IBC-cycle while IB Gateway is
    briefly down. Rather than connect once and immediately trip the watchdog into
    a Docker restart loop, give the gateway a short window to come back. Returns
    True once connected; False if every attempt failed (the caller continues in a
    metrics-only state and the watchdog/Docker loop recovers when the gateway
    returns). Mirrors ``infra.feed.run._connect_with_retry``.
    """
    for i in range(CONNECT_RETRY_ATTEMPTS):
        try:
            exec_adapter.connect()
        except Exception:  # noqa: BLE001 - retry on any connect failure
            _LOG.warning("ibkr connect attempt %d/%d failed", i + 1, CONNECT_RETRY_ATTEMPTS, exc_info=True)
        if adapter.is_connected:
            return True
        if i < CONNECT_RETRY_ATTEMPTS - 1:
            delay = min(CONNECT_RETRY_MAX_DELAY, CONNECT_RETRY_BASE_DELAY * (2**i))
            _LOG.warning("retrying ibkr connect in %.0fs", delay)
            time.sleep(delay)
    return False


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


def _build_ad_consumer(cfg: DryrunConfig, redis_client: Any) -> Consumer:
    """Build the AD-NYSE breadth consumer (strategy mode only).

    The always-flat dry-run consumes TICK-NYSE alone (heartbeat parity
    with the original direct-IBKR subscription set); a hosted strategy
    needs the ADD column too, so strategy mode adds this stream.
    """
    return Consumer(
        redis_client=redis_client,
        stream=stream_name_for_ticks("AD-NYSE"),
        consumer_id="paper-trader-breadth-ad",
        start_id="$",
    )


def _consume_bars_from_bus_sync(
    consumer: Consumer,
    strategy: AlwaysFlatStrategy | PaperStrategyRunner,
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
            bar = {
                "timestamp": pd.Timestamp(msg.payload["ts_minute_utc"], unit="s", tz="UTC"),
                "open": msg.payload["open"],
                "high": msg.payload["high"],
                "low": msg.payload["low"],
                "close": msg.payload["close"],
                "volume": msg.payload["volume"],
            }
            strategy.on_bar(bar, feed_label=feed_label)


def _consume_breadth_from_bus_sync(
    consumer: Consumer,
    strategy: AlwaysFlatStrategy | PaperStrategyRunner,
    stop_event: threading.Event,
    feed_label: str = "tick_nyse",
    default_symbol: str = "TICK-NYSE",
) -> None:
    """Synchronous bus-consumer loop for one breadth tick stream."""
    while not stop_event.is_set():
        for msg in consumer.iter_messages(block_ms=1000):
            if stop_event.is_set():
                return
            tick = {
                "timestamp": pd.Timestamp(msg.ts_event_ns, unit="ns", tz="UTC"),
                "value": msg.payload["value"],
                "symbol": msg.payload.get("symbol", default_symbol),
            }
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
    strategy: AlwaysFlatStrategy | PaperStrategyRunner,
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
    strategy: AlwaysFlatStrategy | PaperStrategyRunner,
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
    strategy: AlwaysFlatStrategy | PaperStrategyRunner,
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
    strategy: AlwaysFlatStrategy | PaperStrategyRunner,
    stop_event_thread: threading.Event,
    bus_consumers: tuple[Consumer, Consumer] | None = None,
    ad_breadth_consumer: Consumer | None = None,
) -> int:
    """Async portion of the dry-run loop.

    If ``bus_consumers`` is provided the bar + breadth streams are read
    from the bus (executor threads). Otherwise the direct-IBKR path is
    used (``adapter`` must not be None in that case).
    ``ad_breadth_consumer`` adds the AD-NYSE stream in strategy mode;
    None preserves the always-flat consumer set exactly.
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
        ad_task: asyncio.Future[None] | None = None
        if ad_breadth_consumer is not None:
            ad_task = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    _consume_breadth_from_bus_sync,
                    ad_breadth_consumer,
                    strategy,
                    stop_event_thread,
                    "ad_nyse",
                    "AD-NYSE",
                )
            )
        # Heartbeat connectivity: in always-flat bus mode there is no IBKR
        # connection at all (adapter is None -> "n/a(bus)"); in strategy
        # mode the exec-side adapter is passed through so the heartbeat
        # reports real order-path connectivity.
        heartbeat_task = asyncio.create_task(_heartbeat_loop(strategy, adapter, stop_event))
    else:
        assert adapter is not None, "adapter required for direct-IBKR mode"
        ad_task = None
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

    # Connection watchdog: scoped to STRATEGY mode (adapter is not None AND bus
    # mode), i.e. the order path - its whole purpose. On a peer-closed socket (the
    # nightly IB Gateway / IBC restart) ib_insync does not auto-reconnect, so the
    # watchdog exits EXIT_RESTART and Docker (restart: unless-stopped) reconnects
    # on a fresh start - mirroring ibkr-feed. Deliberately NOT armed in legacy
    # direct-IBKR mode (bus_consumers is None): that path intentionally serves
    # metrics-only and waits for the gateway rather than restart-looping, and a
    # gateway-down cold start would otherwise trip the watchdog immediately.
    # Always-flat bus mode (adapter=None) holds no IBKR connection at all.
    watch_enabled = adapter is not None and bus_consumers is not None
    wd_task: asyncio.Task[None] | None = None
    if watch_enabled:
        wd_task = asyncio.create_task(
            watch_connection(adapter, poll_seconds=WATCHDOG_POLL_SECONDS),
            name="ibkr-watchdog",
        )
    stop_wait = asyncio.create_task(stop_event.wait(), name="stop-wait")

    exit_code = EXIT_OK
    try:
        # Block until a stop is requested OR (strategy mode) the IBKR
        # connection is lost.
        race: list[asyncio.Future[Any]] = [stop_wait]
        if wd_task is not None:
            race.append(wd_task)
        done, _ = await asyncio.wait(race, return_when=asyncio.FIRST_COMPLETED)
        # Stop wins ties: asyncio.wait(FIRST_COMPLETED) can return both when a
        # clean SIGTERM/SIGINT and a connection loss become ready in the same loop
        # turn. Honour the clean stop (EXIT_OK) in that case, matching
        # infra/feed/run.py; only a connection loss with no stop requested restarts.
        # stop_event_thread is the source-of-truth stop signal (set by the signal
        # handler); checking it as well closes the ~200ms window where the
        # threading->asyncio bridge has not yet set stop_event, during which a
        # concurrent connection loss would otherwise win and force a needless restart.
        if stop_wait in done or stop_event_thread.is_set():
            exit_code = EXIT_OK
        elif wd_task is not None and wd_task in done:
            exit_code = EXIT_RESTART
            _LOG.error(
                "IBKR connection lost; exiting EXIT_RESTART so the container "
                "restart policy reconnects on a fresh start"
            )
            _mark_ibkr_disconnected(adapter)
            # Wake the asyncio loops AND the bus-consumer threads for a clean drain.
            # The bus-consumer threads poll stop_event_thread between iter_messages
            # calls (block_ms=1000), so they observe it and exit within ~1s - the
            # asyncio.run default-executor shutdown then does not hang. (A wedged
            # consumer thread remaining a teardown hazard is tracked for the shared
            # resilience-base hardening in alphaassay-e84.)
            stop_event.set()
            stop_event_thread.set()
    finally:
        for t in (bars_task, breadth_task, heartbeat_task, bridge_task, stop_wait):
            t.cancel()
        if wd_task is not None:
            wd_task.cancel()
        if ad_task is not None:
            ad_task.cancel()
        if deadline_task is not None:
            deadline_task.cancel()
        # Drain with a hard deadline so SIGTERM honors DRAIN_TIMEOUT_SECONDS.
        pending = [bars_task, breadth_task, heartbeat_task, bridge_task, stop_wait]
        if wd_task is not None:
            pending.append(wd_task)
        if ad_task is not None:
            pending.append(ad_task)
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

    return exit_code


def run(cfg: DryrunConfig) -> int:
    """Synchronous entrypoint. Sets up logging, metrics, signals,
    constructs adapters, and runs the async loop.

    Returns the process exit code (always 0 on clean drain).

    Bus-consumer mode is active when ``cfg.bus_redis_url`` is set.
    Direct-IBKR mode is used otherwise (backward-compatible default).
    """
    if cfg.paper_strategy and not cfg.bus_redis_url:
        # Fail closed before any side effects: the joined frame needs the
        # AD-NYSE stream, which only the bus carries. Silently downgrading
        # an intended strategy run to always-flat would mask an operator
        # config error.
        raise RuntimeError(
            "PAPER_STRATEGY requires bus-consumer mode (set BUS_REDIS_URL); "
            "direct-IBKR mode only supports the always-flat dry-run"
        )
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOG.info(
        "paper-dryrun starting: ibkr=%s:%d client_id=%d metrics_port=%d " "duration_seconds=%d bus_mode=%s",
        cfg.ibkr_host,
        cfg.ibkr_port,
        cfg.ibkr_client_id,
        cfg.metrics_port,
        cfg.duration_seconds,
        "yes" if cfg.bus_redis_url else "no",
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
        # Bus-consumer mode: market data comes from Redis Streams.
        import redis as redis_pkg

        redis_client = redis_pkg.from_url(cfg.bus_redis_url)
        # Resolve the front-month expiry now that we have a Redis client.
        expiry, source = _resolve_es_expiry(redis_client)
        cfg = dataclasses.replace(cfg, es_expiry=expiry)
        _LOG.info("resolved ES expiry from %s: %s", source, expiry)
        bus_consumers = _build_bus_consumers(cfg, redis_client)
        _LOG.info(
            "bus-consumer mode active; bars stream=%s breadth stream=%s",
            bus_consumers[0]._stream,
            bus_consumers[1]._stream,
        )

        loaded = load_paper_strategy(dict(os.environ))
        if loaded is None:
            # Always-flat dry-run (backward-compatible default): no IBKR
            # connection at all. Build a stub exec adapter (never used in
            # always-flat, but satisfies the constructor contract so the
            # compose stack remains uniform).
            exec_adapter_stub = object()
            strategy = AlwaysFlatStrategy(exec_adapter=exec_adapter_stub, trade_log=trade_log)
            return asyncio.run(_async_main(cfg, None, strategy, stop_event, bus_consumers=bus_consumers))

        # Strategy mode: market data still comes from the bus, but an IBKR
        # connection is required for the ORDER path (paper submission +
        # fills + startup reconciliation). read_only=False because the
        # exec adapter submits paper orders; live stays triple-gated
        # behind build_exec_adapter's three-lock check.
        _LOG.warning(
            "strategy mode: hosting %s with config %s (PAPER order submission enabled)",
            cfg.paper_strategy,
            cfg.paper_strategy_config,
        )
        adapter = IBKRAdapter(
            host=cfg.ibkr_host,
            port=cfg.ibkr_port,
            client_id=cfg.ibkr_client_id,
            account=cfg.ibkr_account or None,
            read_only=False,
        )
        # The IB connection lives on a dedicated event-loop thread.
        # ib_insync's IB/Client are not thread-safe and are loop-affine,
        # and the order path is driven from bus-consumer worker threads
        # (run_in_executor) plus this main thread (connect + reconcile,
        # both BEFORE asyncio.run creates its own loop). IBLoopThread
        # owns the connection's loop and runs it forever so order-status
        # and fill messages are delivered; every exec call below
        # marshals onto it and blocks with a hard timeout (see
        # alpha_assay.exec.loop_marshal).
        ib_loop = IBLoopThread(name="ib-exec-loop")
        ib_loop.start()
        exec_adapter = build_exec_adapter(adapter=adapter, dry_run=False, loop_owner=ib_loop)
        runner = PaperStrategyRunner(
            strategy=loaded.strategy,
            config=loaded.config,
            exec_adapter=exec_adapter,
            contract_spec=es_contract_spec(cfg),
            trade_log=trade_log,
            starting_balance=cfg.paper_starting_balance,
        )
        exec_adapter.on_fill(runner.handle_fill)
        # Retry the cold-start connect with backoff: a freshly (re)started
        # container can land mid-IBC-cycle while IB Gateway is briefly down, and
        # exiting immediately would churn the watchdog/Docker restart loop.
        if not _connect_exec_with_retry(exec_adapter, adapter):
            _LOG.error("initial ibkr connect failed after retries; orders will fail until the gateway returns")
            runner.on_disconnect()
        if adapter.is_connected:
            qty, cancelled = runner.startup_reconcile()
            _LOG.info("startup reconcile done: position_qty=%g orders_cancelled=%d", qty, cancelled)
        else:
            _LOG.error("skipping startup reconcile: IBKR not connected; broker state is UNVERIFIED")
        ad_consumer = _build_ad_consumer(cfg, redis_client)
        _LOG.info("strategy mode adds breadth stream=%s", ad_consumer._stream)
        try:
            return asyncio.run(
                _async_main(
                    cfg,
                    adapter,
                    runner,
                    stop_event,
                    bus_consumers=bus_consumers,
                    ad_breadth_consumer=ad_consumer,
                )
            )
        finally:
            try:
                # Marshaled onto the IB loop thread; also drains and
                # stops the fill dispatcher. Must precede ib_loop.stop()
                # so the disconnect executes on a still-running loop.
                exec_adapter.disconnect()
            except Exception:
                _LOG.exception("error during ibkr disconnect; ignoring")
            ib_loop.stop()

    # Direct-IBKR mode (backward-compatible default).
    # ES_EXPIRY env var must be set when running without a bus Redis connection;
    # _resolve_es_expiry raises FrontMonthMissingError (fails closed) if neither is available.
    expiry, source = _resolve_es_expiry(None)
    cfg = dataclasses.replace(cfg, es_expiry=expiry)
    _LOG.info("resolved ES expiry from %s: %s", source, expiry)
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
