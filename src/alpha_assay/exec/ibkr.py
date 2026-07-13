# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""IBKR order-submission adapter (WRITE path, ).

Composes an ``IBKRAdapter`` (read path, ) and exposes:

- ``place_bracket_order`` -- parent entry + stop-loss child +
  take-profit child submitted via ``ib_insync.IB.placeOrder``.
- ``cancel_order`` -- request cancellation for a given order id.
- ``on_fill`` -- register a callback that receives fills in the
  canonical AlphaAssay schema.

Triple-locked live-mode guard
-----------------------------

Paper execution is the default. Paper mode still submits real orders --
the IBKR paper account executes them -- it just does not route to the
live broker. Flipping to live requires ALL THREE locks to be engaged:

1. Env var ``ALPHA_ASSAY_LIVE=1``
2. CLI flag ``--live`` on the ``alpha_assay`` CLI
3. Checklist file present on disk at
   ``~/.alpha_assay/go_live_checklist_signed`` (overridable via env
   ``ALPHA_ASSAY_CHECKLIST_PATH``)

If ANY of the three is missing the adapter falls back to PAPER and
emits a WARN log naming the missing locks. The three-lock check lives
in ``check_locks`` which is a pure function: it never modifies
``os.environ`` and never writes to disk -- test independence relies on
this.

Checklist file format
---------------------

only checks existence -- contents are not parsed here. A separate
ship-out tool populates the file with a human-signed manifest listing
approved configs, cap sizes, gateway versions, etc. Signing the
checklist is a deliberate manual step.

Observability
-------------

``alpha_assay_exec_mode{mode="paper|live"}`` is set to 1 for the active
mode and 0 for the other. ``alpha_assay_live_lock_state{lock="..."}``
reflects each of the three lock signals. ``orders_submitted_total`` is
incremented per child with ``type=parent|stop|target``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from alpha_assay.exec.loop_marshal import (
    CONNECT_TIMEOUT_SECONDS,
    DEFAULT_CALL_TIMEOUT_SECONDS,
    FillDispatcher,
    IBLoopThread,
)
from alpha_assay.observability import metrics as M

if TYPE_CHECKING:
    from alpha_assay.data.ibkr_adapter import IBKRAdapter

_LOG = logging.getLogger(__name__)

_DEFAULT_CHECKLIST_SUFFIX = Path(".alpha_assay") / "go_live_checklist_signed"
# ib_insync 0.9.86 keeps this as a local in wrapper.py:Wrapper.error,
# not as a public attribute: warningCodes = {110, 165, 202, 399,
# 404, 434, 492, 10167}. Keep this mirror paired with the source
# because those notices must not be treated as terminal order rejects.
_IB_INSYNC_WARNING_CODES = {110, 165, 202, 399, 404, 434, 492, 10167}
_BENIGN_NOTICE_CODES = _IB_INSYNC_WARNING_CODES | {10349}


class ExecMode(Enum):
    """Execution mode. PAPER is the default; LIVE requires all 3 locks."""

    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True)
class LiveModeLocks:
    """Snapshot of the three live-mode lock signals.

    Attributes
    ----------
    env_set:
        ``ALPHA_ASSAY_LIVE`` environment variable is set to ``"1"``.
    cli_flag:
        The caller passed ``--live`` on the CLI (plumbed through from
        the CLI layer to ``check_locks``).
    checklist_signed:
        The checklist file exists at the configured path.
    """

    env_set: bool
    cli_flag: bool
    checklist_signed: bool

    def all_engaged(self) -> bool:
        """Return True only if all three locks are True."""
        return self.env_set and self.cli_flag and self.checklist_signed

    def missing(self) -> list[str]:
        """Return the names of the locks that are NOT engaged."""
        missing: list[str] = []
        if not self.env_set:
            missing.append("env")
        if not self.cli_flag:
            missing.append("cli")
        if not self.checklist_signed:
            missing.append("checklist")
        return missing


@dataclass
class OrderPlan:
    """Result of ``place_bracket_order``.

    Holds the three IBKR order ids plus the computed prices. When
    ``dry_run`` is True, no orders were submitted but the plan is still
    returned so callers can inspect the pricing.
    """

    parent_id: int
    stop_id: int
    target_id: int
    side: str
    quantity: int
    entry_price: float
    stop_price: float
    target_price: float
    dry_run: bool = False


def _resolve_checklist_path(env: dict[str, str], checklist_path: Path | None) -> Path:
    """Resolve the checklist path from explicit arg / env / default."""
    if checklist_path is not None:
        return checklist_path
    env_override = env.get("ALPHA_ASSAY_CHECKLIST_PATH")
    if env_override:
        return Path(env_override)
    return Path.home() / _DEFAULT_CHECKLIST_SUFFIX


def check_locks(
    env: dict[str, str] | None = None,
    cli_flag: bool = False,
    checklist_path: Path | None = None,
) -> LiveModeLocks:
    """Evaluate the three live-mode locks.

    Pure function: does NOT mutate ``os.environ`` or write to disk.
    Only reads the env dict and ``Path.exists()``.

    Parameters
    ----------
    env:
        Environment-variable mapping. Defaults to a snapshot of
        ``os.environ`` if None is passed.
    cli_flag:
        True if ``--live`` was passed on the CLI.
    checklist_path:
        Explicit path to the signed-checklist file. If None, falls back
        to ``ALPHA_ASSAY_CHECKLIST_PATH`` env var, then to the default
        ``~/.alpha_assay/go_live_checklist_signed``.
    """
    env_map = dict(os.environ) if env is None else dict(env)
    env_set = env_map.get("ALPHA_ASSAY_LIVE") == "1"
    resolved = _resolve_checklist_path(env_map, checklist_path)
    checklist_signed = resolved.exists()

    locks = LiveModeLocks(
        env_set=env_set,
        cli_flag=bool(cli_flag),
        checklist_signed=checklist_signed,
    )
    _update_lock_gauges(locks)
    return locks


def _update_lock_gauges(locks: LiveModeLocks) -> None:
    M.live_lock_state.labels(lock="env").set(1 if locks.env_set else 0)
    M.live_lock_state.labels(lock="cli").set(1 if locks.cli_flag else 0)
    M.live_lock_state.labels(lock="checklist").set(1 if locks.checklist_signed else 0)


def _update_exec_mode_gauge(mode: ExecMode) -> None:
    M.exec_mode.labels(mode="paper").set(1 if mode is ExecMode.PAPER else 0)
    M.exec_mode.labels(mode="live").set(1 if mode is ExecMode.LIVE else 0)


class IBKRExecAdapter:
    """IBKR order-submission adapter.

    Parameters
    ----------
    adapter:
        The read-side ``IBKRAdapter``. The exec adapter reuses its
        underlying ``ib_insync.IB`` client so connect / disconnect
        lifecycles stay single-sourced.
    mode:
        ``ExecMode.PAPER`` (default) or ``ExecMode.LIVE``. Constructed
        by ``build_exec_adapter`` after the three-lock check.
    dry_run:
        If True, ``place_bracket_order`` builds the OrderPlan with
        real ids allocated from the client but skips the
        ``placeOrder`` calls. Useful for unit tests and 's
        paper-dryrun deploy sanity check.
    loop_owner:
        Optional :class:`~alpha_assay.exec.loop_marshal.IBLoopThread`.
        When provided, EVERY ib_insync interaction (connect, orders,
        cancels, position/open-order reads) is marshaled onto that
        dedicated loop thread - ib_insync's IB/Client are not
        thread-safe and are loop-affine (``util.getLoop()`` resolves
        the thread-current loop; the transport and throttle timers bind
        to it). Callers block on the marshaled result with a hard
        timeout and fail loudly on expiry. Fill events, which ib_insync
        fires on the loop thread, are handed to a dedicated dispatcher
        thread so a fill callback can never wedge the IB loop (see
        ``loop_marshal.FillDispatcher``). When None (default), calls
        execute inline on the calling thread - correct only when the
        caller already owns the connection's loop (tests with fakes,
        dry-run paths that never touch ib_insync).
    """

    def __init__(
        self,
        *,
        adapter: IBKRAdapter,
        mode: ExecMode = ExecMode.PAPER,
        dry_run: bool = False,
        loop_owner: IBLoopThread | None = None,
    ) -> None:
        self._adapter = adapter
        self._ib = adapter._ib  # reuse the same client; see docstring
        self.mode = mode
        self.dry_run = dry_run
        self._loop_owner = loop_owner
        self._fill_dispatcher: FillDispatcher | None = None
        self._fill_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._order_status_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._trades_by_order_id: dict[int, Any] = {}
        _update_exec_mode_gauge(mode)

    # --- marshaling -------------------------------------------------------

    def _execute(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` on the connection-owning loop thread when one is
        configured; inline otherwise. Blocks for the result."""
        if self._loop_owner is None:
            return fn(*args, **kwargs)
        return self._loop_owner.call(fn, *args, timeout=DEFAULT_CALL_TIMEOUT_SECONDS, **kwargs)

    # --- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Connect to TWS / Gateway via the underlying IB client.

        Logs the active mode at WARNING so it is visible even in quiet
        log configurations. Paper still submits orders; the IBKR paper
        account handles execution.

        With a ``loop_owner``, the connect coroutine runs ON the owned
        loop thread so ib_insync binds the transport to that loop (the
        loop runs forever, so order-status and fill messages are
        actually delivered). The fill dispatcher is started here too.
        """
        _LOG.warning("IBKRExecAdapter connect: mode=%s", self.mode.value.upper())
        if self._loop_owner is not None:
            if self._fill_dispatcher is None:
                self._fill_dispatcher = FillDispatcher(self._dispatch_dispatcher_event)
                self._fill_dispatcher.start()
            if not self._adapter.is_connected:
                self._loop_owner.run_coro(
                    self._adapter.connect_async(),
                    timeout=CONNECT_TIMEOUT_SECONDS,
                    label="IBKRAdapter.connect_async",
                )
            return
        if not self._adapter.is_connected:
            # Re-use the read adapter's connection path so connection
            # counters and the ibkr_connected gauge stay single-source.
            self._adapter.connect()

    def disconnect(self) -> None:
        if self._loop_owner is not None:
            try:
                self._execute(self._adapter.disconnect)
            finally:
                if self._fill_dispatcher is not None:
                    self._fill_dispatcher.stop()
                    self._fill_dispatcher = None
            return
        self._adapter.disconnect()

    # --- orders ---------------------------------------------------------

    def place_bracket_order(
        self,
        *,
        contract_spec: dict[str, Any],
        side: str,
        quantity: int,
        entry_type: str,
        stop_points: float,
        target_points: float,
        limit_price: float | None = None,
    ) -> OrderPlan:
        """Submit a parent + stop + target bracket.

        The parent order is either MARKET or LIMIT. The stop child is a
        STOP order at ``entry +/- stop_points``; the target child is a
        LIMIT order at ``entry +/- target_points``. Children have the
        reverse action (BUY becomes SELL and vice versa).

        Returns the ``OrderPlan`` with the computed prices and the
        three order ids. If ``self.dry_run`` is True, the plan is
        built but ``placeOrder`` is never called.

        Safe to call from any thread when a ``loop_owner`` is
        configured: the body executes on the connection-owning loop
        thread and this method blocks for the result.
        """
        return self._execute(
            self._place_bracket_order_on_loop,
            contract_spec=contract_spec,
            side=side,
            quantity=quantity,
            entry_type=entry_type,
            stop_points=stop_points,
            target_points=target_points,
            limit_price=limit_price,
        )

    def _place_bracket_order_on_loop(
        self,
        *,
        contract_spec: dict[str, Any],
        side: str,
        quantity: int,
        entry_type: str,
        stop_points: float,
        target_points: float,
        limit_price: float | None = None,
    ) -> OrderPlan:
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if entry_type not in ("MARKET", "LIMIT"):
            raise ValueError(f"entry_type must be MARKET or LIMIT, got {entry_type!r}")
        if entry_type == "LIMIT" and limit_price is None:
            raise ValueError("LIMIT entry requires limit_price")
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

        # Deferred import: ib_insync pulls eventkit which calls
        # asyncio.get_event_loop() at import time and trips on Python 3.13
        # when no loop has been created yet. Importing inside the live
        # order path keeps `check_locks` and other non-order code paths
        # free of that dependency.
        from ib_insync import LimitOrder, MarketOrder, StopOrder

        from alpha_assay.data.ibkr_adapter import _build_contract

        contract = _build_contract(contract_spec)
        reverse = "SELL" if side == "BUY" else "BUY"
        sign = 1.0 if side == "BUY" else -1.0
        # Entry price ref: LIMIT uses the passed limit; MARKET uses it
        # as a notional reference (caller must provide for stop/target
        # math even on MARKET, else we have nothing to anchor on).
        if entry_type == "MARKET":
            if limit_price is None:
                raise ValueError(
                    "MARKET entry still needs limit_price as the reference "
                    "for stop/target math (last known price or expected fill)."
                )
            entry_ref = float(limit_price)
        else:
            entry_ref = float(limit_price)  # type: ignore[arg-type]

        stop_price = entry_ref - sign * float(stop_points)
        target_price = entry_ref + sign * float(target_points)

        parent_id = self._ib.client.getReqId()
        target_id = self._ib.client.getReqId()
        stop_id = self._ib.client.getReqId()

        if entry_type == "LIMIT":
            parent = LimitOrder(
                side,
                quantity,
                entry_ref,
                orderId=parent_id,
                transmit=False,
            )
        else:
            parent = MarketOrder(
                side,
                quantity,
                orderId=parent_id,
                transmit=False,
            )

        target = LimitOrder(
            reverse,
            quantity,
            target_price,
            orderId=target_id,
            parentId=parent_id,
            transmit=False,
        )
        stop = StopOrder(
            reverse,
            quantity,
            stop_price,
            orderId=stop_id,
            parentId=parent_id,
            transmit=True,  # last child transmits the entire bracket
        )

        plan = OrderPlan(
            parent_id=parent_id,
            stop_id=stop_id,
            target_id=target_id,
            side=side,
            quantity=quantity,
            entry_price=entry_ref,
            stop_price=stop_price,
            target_price=target_price,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            _LOG.info("dry_run=True: skipping placeOrder for plan=%s", plan)
            return plan

        parent_trade = self._ib.placeOrder(contract, parent)
        target_trade = self._ib.placeOrder(contract, target)
        stop_trade = self._ib.placeOrder(contract, stop)

        M.orders_submitted_total.labels(type="parent").inc()
        M.orders_submitted_total.labels(type="target").inc()
        M.orders_submitted_total.labels(type="stop").inc()

        # Wire execution and status events on each trade so fills and
        # broker-side terminal failures both reach the runner through
        # dispatcher-owned callbacks.
        for trade in (parent_trade, target_trade, stop_trade):
            self._retain_trade(trade)
            self._wire_trade_events(trade)

        return plan

    def cancel_order(self, order_id: int) -> None:
        """Request cancellation of a given order id.

        ib_insync's ``cancelOrder`` takes the original Order object so
        local Trade bookkeeping can observe status/cancel events.
        """
        self._execute(self._cancel_order_on_loop, order_id)

    def _cancel_order_on_loop(self, order_id: int) -> None:
        trade = self._find_trade(order_id)
        if trade is None:
            _LOG.warning("cancel_order(%d): no live trade found; already done?", order_id)
            return
        self._ib.cancelOrder(trade.order)

    def place_market_order(
        self,
        *,
        contract_spec: dict[str, Any],
        side: str,
        quantity: int,
    ) -> int:
        """Submit a plain market order (no children) and return its id.

        Used for flattening: closing a position at the session buffer,
        on a time-stop, or during startup reconciliation. Fills are
        delivered through the same ``on_fill`` callbacks as bracket
        children. When ``dry_run`` is True the id is allocated but
        ``placeOrder`` is skipped.

        Safe to call from any thread when a ``loop_owner`` is
        configured (see ``place_bracket_order``).
        """
        return self._execute(
            self._place_market_order_on_loop,
            contract_spec=contract_spec,
            side=side,
            quantity=quantity,
        )

    def _place_market_order_on_loop(
        self,
        *,
        contract_spec: dict[str, Any],
        side: str,
        quantity: int,
    ) -> int:
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

        # Deferred import: see place_bracket_order.
        from ib_insync import MarketOrder

        from alpha_assay.data.ibkr_adapter import _build_contract

        contract = _build_contract(contract_spec)
        order_id = self._ib.client.getReqId()
        order = MarketOrder(side, quantity, orderId=order_id, transmit=True)

        if self.dry_run:
            _LOG.info("dry_run=True: skipping placeOrder for market %s %d (id=%d)", side, quantity, order_id)
            return order_id

        trade = self._ib.placeOrder(contract, order)
        self._retain_trade(trade)
        M.orders_submitted_total.labels(type="flatten").inc()
        self._wire_trade_events(trade)
        return order_id

    def position_quantity(self, contract_spec: dict[str, Any]) -> float:
        """Net signed position for the spec's symbol + security type.

        Sums across expiries deliberately: a stale position on last
        quarter's contract is still exposure the runner must not trade
        on top of.

        Marshaled when a ``loop_owner`` is configured: ``IB.positions()``
        reads client state the loop thread mutates, so even reads must
        be loop-affine.
        """
        return self._execute(self._position_quantity_on_loop, contract_spec)

    def _position_quantity_on_loop(self, contract_spec: dict[str, Any]) -> float:
        from alpha_assay.data.ibkr_adapter import _build_contract

        contract = _build_contract(contract_spec)
        total = 0.0
        for pos in self._ib.positions():
            held = getattr(pos, "contract", None)
            if held is None:
                continue
            if held.symbol == contract.symbol and held.secType == contract.secType:
                total += float(pos.position)
        return total

    def cancel_open_orders(self, contract_spec: dict[str, Any]) -> int:
        """Cancel every resting order matching the spec's symbol +
        security type. Returns the number of cancels requested.

        Marshaled when a ``loop_owner`` is configured (reads
        ``openTrades`` and submits cancels - both loop-affine).
        """
        return self._execute(self._cancel_open_orders_on_loop, contract_spec)

    def _cancel_open_orders_on_loop(self, contract_spec: dict[str, Any]) -> int:
        from alpha_assay.data.ibkr_adapter import _build_contract

        contract = _build_contract(contract_spec)
        cancelled = 0
        for trade in self._ib.openTrades():
            held = getattr(trade, "contract", None)
            if held is None:
                continue
            if held.symbol == contract.symbol and held.secType == contract.secType:
                self._ib.cancelOrder(trade.order)
                cancelled += 1
        return cancelled

    # --- fills ----------------------------------------------------------

    def on_fill(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a fill callback.

        The callback receives a dict in the canonical AlphaAssay fill
        schema:

            {
                "order_id": int,
                "order_type": str,      # "LMT" / "STP" / "MKT"
                "exec_id": str,
                "timestamp": pd.Timestamp,  # tz-aware UTC
                "quantity": float,
                "price": float,
                "side": str,            # "BOT" / "SLD"
            }
        """
        self._fill_callbacks.append(callback)

    def on_order_status(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback for terminal non-filled order outcomes.

        The callback receives a canonical dict with the order id, broker
        status string, numeric error code when available, and log message.
        Delivery uses the same dispatcher thread as fills when a loop owner
        is configured.
        """
        self._order_status_callbacks.append(callback)

    def _retain_trade(self, trade: Any) -> None:
        """Keep the live Trade object addressable by its order id."""
        order = getattr(trade, "order", None)
        order_id = _coerce_order_id(getattr(order, "orderId", 0))
        if order_id:
            self._trades_by_order_id[order_id] = trade

    def _find_trade(self, order_id: int) -> Any | None:
        """Find a retained or IB-local live Trade for ``order_id``."""
        trade = self._trades_by_order_id.get(order_id)
        if trade is not None:
            return trade
        for candidate in self._ib.trades():
            order = getattr(candidate, "order", None)
            if _coerce_order_id(getattr(order, "orderId", 0)) == order_id:
                self._retain_trade(candidate)
                return candidate
        return None

    def _wire_trade_events(self, trade: Any) -> None:
        """Subscribe a Trade to fill and terminal-status handlers."""
        fill_event = getattr(trade, "fillEvent", None)
        if fill_event is not None:
            fill_event += self._make_fill_handler()
        status_event = getattr(trade, "statusEvent", None)
        if status_event is not None:
            status_event += self._make_order_status_handler()

    def _make_fill_handler(self) -> Callable[[Any, Any], None]:
        """Build the ib_insync ``fillEvent`` handler.

        ib_insync fires the event on the connection-owning loop thread.
        With a fill dispatcher (loop_owner configured + connected), the
        handler only ENQUEUES - delivery happens on the dispatcher
        thread so a callback that takes the strategy runner's lock can
        never block the IB loop (which may be needed to serve a
        marshaled exec call holding that same lock - lock-inversion
        deadlock otherwise). Without a dispatcher the callbacks run
        inline (legacy/test path with fake adapters).
        """

        def _handle(trade: Any, fill: Any) -> None:
            evt = _fill_to_canonical(trade, fill)
            dispatcher = self._fill_dispatcher
            if dispatcher is not None:
                dispatcher.submit(("fill", evt))
            else:
                self._dispatch_fill(evt)

        return _handle

    def _make_order_status_handler(self) -> Callable[[Any], None]:
        """Build the ib_insync ``statusEvent`` rejection handler."""

        def _handle(trade: Any) -> None:
            evt = _order_status_to_canonical(trade)
            if evt is None:
                return
            dispatcher = self._fill_dispatcher
            if dispatcher is not None:
                dispatcher.submit(("order_status", evt))
            else:
                self._dispatch_order_status(evt)

        return _handle

    def _dispatch_dispatcher_event(self, item: Any) -> None:
        """Route dispatcher-thread events to their registered callback list."""
        kind, evt = item
        if kind == "fill":
            self._dispatch_fill(evt)
            return
        if kind == "order_status":
            self._dispatch_order_status(evt)
            return
        _LOG.error("unknown exec dispatcher event kind=%r payload=%r", kind, evt)

    def _dispatch_fill(self, evt: dict[str, Any]) -> None:
        for cb in list(self._fill_callbacks):
            cb(evt)

    def _dispatch_order_status(self, evt: dict[str, Any]) -> None:
        for cb in list(self._order_status_callbacks):
            cb(evt)


def _fill_to_canonical(trade: Any, fill: Any) -> dict[str, Any]:
    order = getattr(trade, "order", None)
    execution = getattr(fill, "execution", None)
    raw_time = getattr(execution, "time", None)
    if raw_time is None:
        ts = pd.Timestamp.utcnow()
    else:
        ts = pd.Timestamp(raw_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    return {
        "order_id": int(getattr(order, "orderId", 0) or 0),
        "order_type": str(getattr(order, "orderType", "") or ""),
        "exec_id": str(getattr(execution, "execId", "") or ""),
        "timestamp": ts,
        "quantity": float(getattr(execution, "shares", 0.0) or 0.0),
        "price": float(getattr(execution, "price", 0.0) or 0.0),
        "side": str(getattr(execution, "side", "") or ""),
    }


def _coerce_order_id(value: Any) -> int:
    """Return a real integer order id from ib_insync values, or zero."""
    if isinstance(value, int | float | str):
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _order_status_to_canonical(trade: Any) -> dict[str, Any] | None:
    """Convert a terminal non-filled Trade status into a canonical event."""
    from ib_insync import OrderStatus

    status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
    if status not in OrderStatus.DoneStates or status == OrderStatus.Filled:
        return None
    order = getattr(trade, "order", None)
    last_entry = trade.log[-1] if getattr(trade, "log", None) else None
    error_code = int(getattr(last_entry, "errorCode", 0) or 0)
    if error_code in _BENIGN_NOTICE_CODES:
        # IBKR sends error 10349 when an order preset rewrites TIF to DAY.
        # ib_insync does not classify that code as a warning, so
        # Wrapper.error forces trade.orderStatus.status = Cancelled and
        # emits statusEvent/cancelledEvent. Production then immediately
        # receives PreSubmitted and execDetails for the same order. That
        # transient Cancelled is broker noise, not a rejection.
        return None
    return {
        "order_id": int(getattr(order, "orderId", 0) or 0),
        "status": status,
        "error_code": error_code,
        "message": str(getattr(last_entry, "message", "") or ""),
    }


def build_exec_adapter(
    *,
    adapter: IBKRAdapter,
    env: dict[str, str] | None = None,
    cli_live: bool = False,
    checklist_path: Path | None = None,
    dry_run: bool = False,
    loop_owner: IBLoopThread | None = None,
) -> IBKRExecAdapter:
    """Construct an ``IBKRExecAdapter`` with the three-lock guard.

    Mode selection:

    - All three locks engaged -> ``ExecMode.LIVE``
    - Any lock missing -> ``ExecMode.PAPER`` and a WARN log names
      every missing lock

    ``loop_owner`` is forwarded to the adapter; pass a started
    ``IBLoopThread`` whenever exec methods will be called from threads
    other than the one that owns the IB connection's event loop.
    """
    locks = check_locks(env=env, cli_flag=cli_live, checklist_path=checklist_path)
    if locks.all_engaged():
        mode = ExecMode.LIVE
    else:
        mode = ExecMode.PAPER
        _LOG.warning(
            "Live mode NOT engaged -- falling back to PAPER. Missing locks: %s",
            ", ".join(locks.missing()),
        )
    return IBKRExecAdapter(adapter=adapter, mode=mode, dry_run=dry_run, loop_owner=loop_owner)


__all__ = [
    "ExecMode",
    "IBKRExecAdapter",
    "LiveModeLocks",
    "OrderPlan",
    "build_exec_adapter",
    "check_locks",
]
