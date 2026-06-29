# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Env-selected strategy host for the paper trader.

`PaperStrategyRunner` exposes the same event surface as the always-flat
dry-run strategy (`on_bar` / `on_breadth_tick` / `on_disconnect`) so the
bus-consumer loops in `scripts/paper_dryrun.py` drive either one
unchanged. The runner:

- builds the canonical joined minute frame incrementally
  (`alpha_assay.paper.frame`),
- calls `BaseStrategy.generate_signals` on every completed row and acts
  only on a non-zero signal on the frame's last row (the strategy's own
  dedup semantics - e.g. first-per-day gating - flow through untouched),
- enforces the session window, hard risk caps, and a max of ONE
  concurrent position before submitting a paper bracket via
  `IBKRExecAdapter.place_bracket_order`,
- correlates entry/stop/target fills into a round trip and writes the
  realized P&L to the `TradeLog` on exit (the dashboard equity curve
  reads `account_balance_after` from those records),
- flattens at the session-close buffer and on the strategy's time-stop,
- reconciles broker state on startup (conservative policy: cancel
  resting orders and market-flatten any open position, loudly).

Strategy selection is env-driven:

``PAPER_STRATEGY``           ``package.module:ClassName`` of a
                             `BaseStrategy` subclass. Unset = always-flat
                             dry-run (backward-compatible default).
``PAPER_STRATEGY_CONFIG``    Path to the strategy's YAML config (the
                             `alpha_assay.config.loader` schema). Required
                             when ``PAPER_STRATEGY`` is set - a strategy
                             without risk caps and session bounds must
                             not trade.
``PAPER_STARTING_BALANCE``   Reference account balance for risk-based
                             sizing (default ``100000``). The sizer uses
                             a fixed reference, not live equity.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from alpha_assay.config.loader import AlphaAssayConfig, load_config
from alpha_assay.exec.ibkr import OrderPlan
from alpha_assay.exec.trade_log import TradeLog, TradeRecord
from alpha_assay.filters.session_mask import CLOSE_CT_MINUTES, session_mask
from alpha_assay.observability import metrics as M
from alpha_assay.paper.frame import MinuteCloseAggregator, SessionFrameBuilder
from alpha_assay.risk.caps import RiskCaps, RiskCapViolation
from alpha_assay.risk.sizing import PositionSizer
from alpha_assay.strategy.base import BaseStrategy, Signal

_LOG = logging.getLogger(__name__)

# Breadth bus symbol -> joined-frame column.
BREADTH_SYMBOL_COLUMNS = {
    "TICK-NYSE": "TICK",
    "AD-NYSE": "ADD",
}

# Instrument-symbol prefix -> dollar point value. Mirrors the contract
# multipliers in `alpha_assay.engine.instrument_factory` (kept as data
# here so the paper container does not need the nautilus extra).
# Micro contracts must sort before their parent prefix.
_POINT_VALUES: tuple[tuple[str, float], ...] = (
    ("MES", 5.0),
    ("MNQ", 2.0),
    ("ES", 50.0),
)


def point_value_for(instrument: str) -> float:
    """Dollar value of one point for `instrument` (prefix match)."""
    for prefix, value in _POINT_VALUES:
        if instrument.startswith(prefix):
            return value
    raise ValueError(f"unknown instrument {instrument!r}; expected prefix in {[p for p, _ in _POINT_VALUES]}")


@dataclass(frozen=True)
class LoadedStrategy:
    """A strategy instance plus its validated config, ready to host."""

    strategy: BaseStrategy
    config: AlphaAssayConfig


def load_paper_strategy(env: dict[str, str]) -> LoadedStrategy | None:
    """Resolve PAPER_STRATEGY / PAPER_STRATEGY_CONFIG from `env`.

    Returns None when ``PAPER_STRATEGY`` is unset (always-flat default).
    Raises on a half-configured or unimportable strategy - fail closed
    rather than silently falling back when an operator clearly intended
    a strategy to run.
    """
    class_path = (env.get("PAPER_STRATEGY") or "").strip()
    if not class_path:
        return None
    config_path = (env.get("PAPER_STRATEGY_CONFIG") or "").strip()
    if not config_path:
        raise ValueError("PAPER_STRATEGY is set but PAPER_STRATEGY_CONFIG is not; refusing to run without risk caps")
    config = load_config(config_path)
    if class_path != config.strategy.class_:
        _LOG.warning(
            "PAPER_STRATEGY %s overrides strategy.class %s from %s",
            class_path,
            config.strategy.class_,
            config_path,
        )
    module_name, _, class_name = class_path.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"PAPER_STRATEGY must look like 'package.module:ClassName', got {class_path!r}")
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)
    strategy = strategy_cls(config.strategy.params)
    if config.execution.mode != "paper":
        _LOG.warning(
            "strategy config execution.mode=%s; the paper runner submits PAPER orders regardless", config.execution.mode
        )
    return LoadedStrategy(strategy=strategy, config=config)


@dataclass
class _OpenPosition:
    """State of the single in-flight round trip."""

    direction: int  # +1 long, -1 short
    contracts: int
    plan: OrderPlan
    signal_ts: pd.Timestamp  # frame timestamp (America/Chicago)
    signal_type: str  # "long_entry" / "short_entry"
    entry_ref_price: float  # last close at signal time (bracket anchor)
    time_stop: timedelta | None
    entry_qty: float = 0.0
    entry_avg_px: float = 0.0
    exit_qty: float = 0.0
    exit_avg_px: float = 0.0
    flatten_order_id: int | None = None
    flatten_reason: str | None = None  # "flat_close" / "time_stop"


class PaperStrategyRunner:
    """Hosts a `BaseStrategy` against the live bus in paper mode.

    Duck-type compatible with the always-flat strategy's event surface
    (`on_bar`, `on_breadth_tick`, `on_disconnect`, plus the
    `bars_seen` / `ticks_seen` / `disconnect_count` heartbeat counters
    and the `_trade_log` shutdown-flush attribute).

    The exec adapter only needs `place_bracket_order`, `cancel_order`,
    `place_market_order`, `position_quantity` and `cancel_open_orders`;
    tests substitute a fake.
    """

    def __init__(
        self,
        *,
        strategy: BaseStrategy,
        config: AlphaAssayConfig,
        exec_adapter: Any,
        contract_spec: dict[str, Any],
        trade_log: TradeLog | None = None,
        starting_balance: float = 100_000.0,
    ) -> None:
        self._strategy = strategy
        self._strategy_name = type(strategy).__name__
        self._config = config
        self._exec = exec_adapter
        self._contract_spec = contract_spec
        self._trade_log = trade_log
        self._point_value = point_value_for(config.execution.instrument)
        self._risk_caps = RiskCaps(
            max_stop_pts=config.risk_caps.max_stop_pts,
            min_target_pts=config.risk_caps.min_target_pts,
            min_target_to_stop_ratio=config.risk_caps.min_target_to_stop_ratio,
        )
        self._sizer = PositionSizer(
            account_balance=starting_balance,
            instrument_multiplier=self._point_value,
            risk_per_trade_pct=config.execution.risk_per_trade_pct,
            max_contracts=config.execution.max_contracts,
        )
        self._balance = starting_balance
        self._realized_total = 0.0
        self._builder = SessionFrameBuilder()
        self._aggregators = {symbol: MinuteCloseAggregator(symbol) for symbol in BREADTH_SYMBOL_COLUMNS}
        self._position: _OpenPosition | None = None
        # RLock: a synchronous exec fake may invoke handle_fill from
        # within place_bracket_order, which runs under this lock.
        #
        # Threading contract for the live paper path: bus-consumer
        # worker threads drive on_bar/on_breadth_tick, which take this
        # lock and may block on exec calls marshaled onto the IB loop
        # thread (exec/loop_marshal.IBLoopThread). handle_fill arrives
        # on the exec adapter's FillDispatcher thread - NEVER the IB
        # loop thread - so taking this lock here can never wedge the
        # loop that serves those marshaled calls (lock-inversion
        # deadlock). Invariant: handle_fill must stay free of exec-
        # adapter calls; if one is ever added it only blocks the
        # dispatcher thread (safe), but it delays subsequent fills -
        # prefer deferring broker actions to the bar-driven paths.
        self._lock = threading.RLock()

        # Heartbeat counters (same contract as the always-flat strategy).
        self.bars_seen = 0
        self.ticks_seen = 0
        self.disconnect_count = 0
        # Signals that survived every gate and produced an order, in
        # firing order. Equivalence tests compare this against
        # generate_signals over the end-state frame.
        self.fired_signals: list[Signal] = []

        # Front-month roll backstop: the expiry the bus subscription is
        # bound to vs the producer's resolved front month. While they
        # diverge (a roll the consumer has not yet rebound across) the
        # runner REFUSES to submit orders - it must never trade on bars
        # from a rolled-off contract. The paper-trader harness keeps these
        # fresh from the front-month key on each consumer poll.
        self._consumed_expiry: str | None = contract_spec.get("expiry")
        self._resolved_expiry: str | None = contract_spec.get("expiry")

        M.position_contracts.set(0)
        M.realized_pnl_dollars.set(0.0)

    # --- introspection ----------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """Current joined minute frame (read-only view for tests/ops)."""
        return self._builder.frame

    @property
    def account_balance(self) -> float:
        return self._balance

    @property
    def open_position(self) -> _OpenPosition | None:
        return self._position

    @property
    def front_month_diverged(self) -> bool:
        """True while the consumed expiry differs from the resolved front month.

        The runner refuses to open new positions while this holds (see
        :meth:`_on_row`): trading on a rolled-off contract is exactly the
        failure this fix prevents.
        """
        return (
            self._consumed_expiry is not None
            and self._resolved_expiry is not None
            and self._consumed_expiry != self._resolved_expiry
        )

    def update_front_month(
        self,
        *,
        consumed_expiry: str | None = None,
        resolved_expiry: str | None = None,
    ) -> None:
        """Refresh the consumed / resolved front-month expiries.

        Called by the paper-trader harness on each consumer poll:
        ``resolved_expiry`` from the front-month key, ``consumed_expiry``
        once the bus subscription has rebound to a new contract. A
        divergence between the two halts new entries until the rebind
        catches up.

        When ``consumed_expiry`` advances (a roll cutover), the ORDER
        contract is re-pointed too: ``_contract_spec`` is rebuilt for the
        new expiry so ``place_bracket_order`` / ``cancel_open_orders`` /
        ``position_quantity`` / ``startup_reconcile`` all act on the LIVE
        contract. Without this the runner would generate signals from
        new-contract bars but submit orders against the rolled-off
        contract.

        Flat-at-roll invariant: the roll happens pre-open (the producer
        re-qualifies at 08:00 CT, before the 08:30 RTH open) and the
        strategy flattens into the close, so the runner is flat at the
        cutover. Re-pointing the order contract while a position is OPEN
        would orphan the resting bracket on the old contract, so that
        case is refused loudly (divergence is left in place, halting new
        entries) rather than silently re-pointed - a manual-intervention
        signal, never expected in normal operation.
        """
        if resolved_expiry is not None:
            self._resolved_expiry = resolved_expiry
        if consumed_expiry is not None and consumed_expiry != self._consumed_expiry:
            if self._position is not None:
                _LOG.error(
                    "front-month cutover to %s while a position is OPEN on contract %s; NOT re-pointing "
                    "the order contract (would orphan the resting bracket) and NOT clearing divergence - "
                    "the open position is managed on its original contract and new entries stay halted; "
                    "manual intervention required",
                    consumed_expiry,
                    self._contract_spec.get("expiry"),
                )
                return
            self._contract_spec = {**self._contract_spec, "expiry": consumed_expiry}
            self._consumed_expiry = consumed_expiry
            _LOG.warning(
                "front-month cutover: order contract re-pointed to expiry %s (flat at roll)",
                consumed_expiry,
            )
        elif consumed_expiry is not None:
            self._consumed_expiry = consumed_expiry

    # --- bus event surface --------------------------------------------------

    def on_bar(self, bar: dict[str, Any], *, feed_label: str) -> None:
        """Ingest one completed ES 1-min bar from the bus."""
        self.bars_seen += 1
        M.bars_processed_total.labels(feed=feed_label).inc()
        with self._lock:
            for ts in self._builder.add_es_bar(bar):
                self._on_row(ts)

    def on_breadth_tick(self, tick: dict[str, Any], *, feed_label: str) -> None:
        """Ingest one raw breadth tick; aggregates to per-minute closes."""
        self.ticks_seen += 1
        M.bars_processed_total.labels(feed=feed_label).inc()
        symbol = str(tick.get("symbol", ""))
        column = BREADTH_SYMBOL_COLUMNS.get(symbol)
        if column is None:
            _LOG.warning("breadth tick for unknown symbol %r; ignoring", symbol)
            return
        with self._lock:
            emitted = self._aggregators[symbol].ingest(pd.Timestamp(tick["timestamp"]), float(tick["value"]))
            if emitted is None:
                return
            minute, close = emitted
            for ts in self._builder.add_breadth_close(column, minute, close):
                self._on_row(ts)

    def on_disconnect(self) -> None:
        self.disconnect_count += 1
        _LOG.warning("ibkr disconnect observed; count=%d", self.disconnect_count)

    # --- startup reconciliation ---------------------------------------------

    def startup_reconcile(self) -> tuple[float, int]:
        """Conservative restart policy: flatten anything already open.

        A restart mid-trade means the runner has no entry context for the
        broker-side position, so adopting it would mean managing a trade
        it cannot price. Cancel resting orders for the instrument and
        market-flatten any open position, logging loudly so the operator
        sees it. Returns ``(position_qty_found, orders_cancelled)``.
        """
        with self._lock:
            cancelled = int(self._exec.cancel_open_orders(self._contract_spec))
            qty = float(self._exec.position_quantity(self._contract_spec))
            if cancelled:
                _LOG.error("startup reconcile: cancelled %d resting order(s) on %s", cancelled, self._contract_spec)
            if qty != 0:
                side = "SELL" if qty > 0 else "BUY"
                _LOG.error(
                    "startup reconcile: found open position of %+g on %s; submitting %s market flatten",
                    qty,
                    self._contract_spec,
                    side,
                )
                self._exec.place_market_order(
                    contract_spec=self._contract_spec,
                    side=side,
                    quantity=abs(int(qty)),
                )
            return qty, cancelled

    # --- fills (90e.3: realized P&L) -----------------------------------------

    def handle_fill(self, evt: dict[str, Any]) -> None:
        """Correlate a canonical fill event into the open round trip.

        Entry fills accumulate the average fill price. Exit fills (stop,
        target, or flatten order) accumulate until the position quantity
        is covered, then the round trip is finalized: realized P&L is
        computed, the account balance updated, and a completed-trade
        record written to the trade log.
        """
        with self._lock:
            pos = self._position
            if pos is None:
                _LOG.warning("fill with no open position (startup flatten or stale): %s", evt)
                return
            order_id = int(evt.get("order_id", 0))
            qty = float(evt.get("quantity", 0.0))
            price = float(evt.get("price", 0.0))
            if order_id == pos.plan.parent_id:
                pos.entry_avg_px = _accumulate_avg(pos.entry_avg_px, pos.entry_qty, price, qty)
                pos.entry_qty += qty
                M.orders_filled_total.labels(type="entry", status="filled").inc()
                M.fill_slippage_points.labels(type="entry").observe((price - pos.entry_ref_price) * pos.direction)
                _LOG.info("entry fill: qty=%g px=%.2f (avg %.2f)", qty, price, pos.entry_avg_px)
                return
            outcome: str | None = None
            if order_id == pos.plan.stop_id:
                outcome = "stop"
            elif order_id == pos.plan.target_id:
                outcome = "target"
            elif pos.flatten_order_id is not None and order_id == pos.flatten_order_id:
                outcome = pos.flatten_reason or "flat_close"
            if outcome is None:
                _LOG.warning("fill for unknown order_id=%d; ignoring: %s", order_id, evt)
                return
            pos.exit_avg_px = _accumulate_avg(pos.exit_avg_px, pos.exit_qty, price, qty)
            pos.exit_qty += qty
            M.orders_filled_total.labels(type=outcome, status="filled").inc()
            if pos.exit_qty >= pos.contracts:
                self._finalize_round_trip(pos, outcome, evt.get("timestamp"))

    # --- internals -----------------------------------------------------------

    def _on_row(self, ts: pd.Timestamp) -> None:
        """Evaluate the strategy on a newly completed frame row."""
        frame = self._builder.frame
        self._manage_open_position(ts)
        t0 = time.perf_counter()
        signals = self._strategy.generate_signals(frame)
        M.signal_eval_seconds.labels(strategy=self._strategy_name).observe(time.perf_counter() - t0)
        direction = int(signals.iloc[-1])
        if direction == 0:
            return
        M.signals_generated_total.labels(strategy=self._strategy_name, direction=str(direction)).inc()
        if self.front_month_diverged:
            # Backstop: the bus subscription is on a rolled-off contract
            # (the producer has moved to a new front month and the consumer
            # has not yet rebound). Refuse to enter on stale-contract bars.
            self._filtered("front_month_diverged", "stale_contract")
            _LOG.error(
                "front-month diverged (consuming %s, resolved %s); refusing to trade until rebind",
                self._consumed_expiry,
                self._resolved_expiry,
            )
            return
        if not self._in_session(ts):
            self._filtered("session_mask", "outside_window")
            return
        if self._position is not None:
            self._filtered("max_positions", "position_open")
            return
        signal = Signal(timestamp=ts, direction=direction)
        exit_params = self._strategy.get_exit_params(signal, frame)
        try:
            self._risk_caps.validate_exit_params(exit_params)
        except RiskCapViolation as exc:
            self._filtered("risk_caps", "risk_cap")
            _LOG.warning("risk cap violation; signal dropped: %s", exc)
            return
        contracts = self._sizer.compute_contracts(exit_params.stop_points)
        side = "BUY" if direction > 0 else "SELL"
        entry_ref = float(frame["close"].iloc[-1])
        try:
            plan = self._exec.place_bracket_order(
                contract_spec=self._contract_spec,
                side=side,
                quantity=contracts,
                entry_type="MARKET",
                stop_points=exit_params.stop_points,
                target_points=exit_params.target_points,
                limit_price=entry_ref,
            )
        except Exception:
            _LOG.exception("bracket submission failed for signal at %s; staying flat", ts)
            return
        self._position = _OpenPosition(
            direction=direction,
            contracts=contracts,
            plan=plan,
            signal_ts=ts,
            signal_type="long_entry" if direction > 0 else "short_entry",
            entry_ref_price=entry_ref,
            time_stop=exit_params.time_stop,
        )
        self.fired_signals.append(signal)
        M.signals_fired_total.labels(strategy=self._strategy_name, direction=str(direction)).inc()
        M.position_contracts.set(direction * contracts)
        M.bar_to_order_seconds.observe(0.0)
        _LOG.info(
            "signal fired: %s %d contract(s) at ref %.2f (stop %.2f / target %.2f)",
            side,
            contracts,
            entry_ref,
            plan.stop_price,
            plan.target_price,
        )

    def _manage_open_position(self, ts: pd.Timestamp) -> None:
        """Flat-into-close and time-stop checks, driven by bar time."""
        pos = self._position
        if pos is None or pos.flatten_order_id is not None:
            return
        minutes = ts.hour * 60 + ts.minute
        cutoff = CLOSE_CT_MINUTES - self._config.session.minutes_before_close
        if minutes >= cutoff:
            self._flatten(pos, "flat_close")
            return
        if pos.time_stop is not None and (ts - pos.signal_ts) >= pos.time_stop:
            self._flatten(pos, "time_stop")

    def _flatten(self, pos: _OpenPosition, reason: str) -> None:
        """Cancel resting children and market-flatten the open position."""
        _LOG.warning("flattening open position (%s): %+d x %d", reason, pos.direction, pos.contracts)
        for order_id in (pos.plan.stop_id, pos.plan.target_id):
            try:
                self._exec.cancel_order(order_id)
            except Exception:
                _LOG.exception("cancel_order(%d) failed during %s flatten", order_id, reason)
        side = "SELL" if pos.direction > 0 else "BUY"
        try:
            pos.flatten_order_id = int(
                self._exec.place_market_order(
                    contract_spec=self._contract_spec,
                    side=side,
                    quantity=pos.contracts,
                )
            )
            pos.flatten_reason = reason
        except Exception:
            _LOG.exception("market flatten submission failed (%s); brackets may still be resting", reason)

    def _finalize_round_trip(self, pos: _OpenPosition, outcome: str, exit_ts: Any) -> None:
        """Compute realized P&L and persist the completed round trip."""
        entry_px = pos.entry_avg_px if pos.entry_qty > 0 else pos.entry_ref_price
        if pos.entry_qty <= 0:
            _LOG.warning("exit fill before any entry fill observed; using entry reference price %.2f", entry_px)
        points = (pos.exit_avg_px - entry_px) * pos.direction
        realized = points * pos.contracts * self._point_value
        self._balance += realized
        self._realized_total += realized
        ts = pd.Timestamp(exit_ts) if exit_ts is not None else pd.Timestamp.now(tz="UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        M.trades_total.labels(outcome=outcome).inc()
        M.trade_pnl_points.labels(outcome=outcome).observe(points)
        M.trade_duration_seconds.observe(max(0.0, (ts - pos.signal_ts.tz_convert("UTC")).total_seconds()))
        M.realized_pnl_dollars.set(self._realized_total)
        M.position_contracts.set(0)

        if self._trade_log is not None:
            record = TradeRecord(
                timestamp=ts,
                signal_type=pos.signal_type,
                entry_price=pos.entry_ref_price,
                stop=pos.plan.stop_price,
                target=pos.plan.target_price,
                mock_fill_price=entry_px,
                mock_pnl_dollars=realized,
                account_balance_after=self._balance,
            )
            self._trade_log.write(record)
            # Flush per round trip: a handful of trades per session, and
            # the dashboard reads the parquet while the process runs.
            try:
                self._trade_log.flush()
            except Exception:
                _LOG.exception("trade log flush failed; record buffered for shutdown flush")
        _LOG.info(
            "round trip closed (%s): %s %d x %.2f -> %.2f = %+.2f USD (balance %.2f)",
            outcome,
            pos.signal_type,
            pos.contracts,
            entry_px,
            pos.exit_avg_px,
            realized,
            self._balance,
        )
        self._position = None

    def _in_session(self, ts: pd.Timestamp) -> bool:
        mask = session_mask(
            pd.DatetimeIndex([ts]),
            minutes_after_open=self._config.session.minutes_after_open,
            minutes_before_close=self._config.session.minutes_before_close,
        )
        return bool(mask.iloc[0])

    def _filtered(self, filter_name: str, reason: str) -> None:
        M.signals_filtered_total.labels(
            strategy=self._strategy_name,
            filter_name=filter_name,
            reason=reason,
        ).inc()


def _accumulate_avg(avg: float, qty: float, fill_px: float, fill_qty: float) -> float:
    """Quantity-weighted running average fill price."""
    total = qty + fill_qty
    if total <= 0:
        return fill_px
    return (avg * qty + fill_px * fill_qty) / total


__all__ = [
    "BREADTH_SYMBOL_COLUMNS",
    "LoadedStrategy",
    "PaperStrategyRunner",
    "load_paper_strategy",
    "point_value_for",
]
