# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""NautilusTrader backtest adapter.

The ONLY module in the library that imports NautilusTrader. Strategies
stay engine-agnostic: they receive DataFrames, emit Series, and return
ExitParams. This adapter bridges.

Public surface:
    - NautilusBacktestRunner: wraps BacktestEngine, accepts a
      BaseStrategy, a DataFrame (ES OHLCV + TICK + ADD columns), an
      instrument symbol (ESM6 / MESM6 / MNQM6), and a starting balance.
      Exposes .run() -> BacktestResult.
    - BacktestResult: engine-agnostic dataclass returned from .run() so
      downstream code (metrics, CLI, tests) never imports Nautilus types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarSpecification, BarType, DataType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import ClientId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy as NautilusStrategy

from alpha_assay.engine.bar_adapter import df_to_bars
from alpha_assay.engine.breadth_adapter import df_to_breadth
from alpha_assay.engine.custom_data import AddIndicator, TickIndicator
from alpha_assay.engine.instrument_factory import (
    make_es_futures,
    make_mes_futures,
    make_mnq_futures,
)
from alpha_assay.engine.logging import LOGGING_CONFIG
from alpha_assay.filters.session_mask import session_mask
from alpha_assay.observability import metrics as M
from alpha_assay.risk.caps import RiskCaps
from alpha_assay.strategy.base import BaseStrategy, Signal

_VENUE = Venue("SIM")
_CLIENT_ID = ClientId("BREADTH")


@dataclass(frozen=True)
class PositionSizer:
    """Risk-based contract sizing.

    Contracts = clamp(floor(account_balance * risk_per_trade_pct /
    stop_dollar), 1, max_contracts) when risk_per_trade_pct is set.
    Falls back to 1 when risk_per_trade_pct is None or 0 (legacy
    behavior - single contract per signal).

    Uses a fixed account_balance reference (not live equity) so risk
    budget stays stable across drawdowns. Anti-martingale scaling can
    be added later if desired.
    """

    account_balance: float
    instrument_multiplier: float
    risk_per_trade_pct: float | None = None
    max_contracts: int = 1

    def compute_contracts(self, stop_points: float) -> int:
        if self.risk_per_trade_pct is None or self.risk_per_trade_pct <= 0:
            return 1
        if stop_points <= 0:
            # Defensive: zero stop would divide by zero. Fall back to 1 -
            # the risk-cap layer should have rejected this already, but
            # the engine must not crash.
            return 1
        risk_dollar = self.account_balance * self.risk_per_trade_pct
        stop_dollar = stop_points * self.instrument_multiplier
        contracts = int(risk_dollar // stop_dollar)
        return max(1, min(self.max_contracts, contracts))


@dataclass
class BacktestResult:
    """Engine-agnostic view of a finished backtest."""

    trades: list[dict[str, Any]] = field(default_factory=list)
    session_metrics: dict[str, Any] = field(default_factory=dict)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


class _NautilusStrategyAdapter(NautilusStrategy):
    """Internal bridge. Holds a rolling pandas buffer and invokes
    BaseStrategy.generate_signals with a causal slice on every bar.
    Emits bracket orders on non-zero signals (wired in J3).
    """

    def __init__(
        self,
        *,
        bar_type,
        instrument_id,
        strategy,
        risk_caps,
        position_sizer: PositionSizer | None = None,
    ) -> None:
        super().__init__()
        self._bar_type = bar_type
        self._instrument_id = instrument_id
        self._strategy = strategy
        self._strategy_name = type(strategy).__name__
        self._risk_caps = risk_caps
        self._position_sizer = position_sizer
        self._bars_records: list[dict[str, Any]] = []
        self._latest_tick: float = 0.0
        self._latest_add: float = 0.0
        self._tick_size: float = 0.25
        self._orders_submitted: int = 0
        self._signals_filtered_risk_cap: int = 0
        self.submitted_signals: list[Signal] = []
        # Map order_list_id -> the strategy-side signal timestamp that
        # produced the bracket. Filled-trade harvest in run() looks up
        # signal_ts here to compute fill latency. Keyed by
        # the OrderListId object directly; Nautilus reuses the same
        # instance across the bracket's three child orders so identity
        # matches without stringification.
        self._signal_ts_by_order_list: dict[Any, pd.Timestamp] = {}

    def on_start(self) -> None:
        self.subscribe_bars(self._bar_type)
        self.subscribe_data(DataType(TickIndicator), instrument_id=self._instrument_id)
        self.subscribe_data(DataType(AddIndicator), instrument_id=self._instrument_id)

    def on_data(self, data) -> None:
        if isinstance(data, TickIndicator):
            self._latest_tick = float(data.value)
        elif isinstance(data, AddIndicator):
            self._latest_add = float(data.value)

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.ts_event, tz="UTC")
        self._bars_records.append(
            {
                "timestamp": ts,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "TICK": self._latest_tick,
                "ADD": self._latest_add,
            }
        )
        df = pd.DataFrame(self._bars_records).set_index("timestamp")
        # Session gate: only evaluate on in-window bars. Spec Section 5
        # lookahead guarantee + Section 7 pre-close policy make this a
        # merge-blocker invariant.
        in_session = session_mask(df.index)
        if not bool(in_session.iloc[-1]):
            M.in_session.set(0.0)
            return
        M.bars_processed_total.labels(feed="ES").inc()
        M.in_session.set(1.0)
        t0 = time.perf_counter()
        signals = self._strategy.generate_signals(df)
        M.signal_eval_seconds.labels(strategy=self._strategy_name).observe(time.perf_counter() - t0)
        latest = int(signals.iloc[-1])
        if latest == 0:
            return
        M.signals_generated_total.labels(
            strategy=self._strategy_name,
            direction=str(latest),
        ).inc()
        signal = Signal(timestamp=df.index[-1], direction=latest)
        exit_params = self._strategy.get_exit_params(signal, df)
        # Risk gate: drop on violation, increment counter. Hard caps
        # stay engine-enforced (spec decision 9).
        try:
            self._risk_caps.validate_exit_params(exit_params)
        except Exception as exc:
            self._signals_filtered_risk_cap += 1
            M.signals_filtered_total.labels(
                strategy=self._strategy_name,
                filter_name="risk_caps",
                reason="risk_cap",
            ).inc()
            self.log.warning(f"risk cap violation; signal dropped: {exc}")
            return
        self.submitted_signals.append(signal)

        last_price = float(bar.close)
        tick = float(self._tick_size)
        if signal.direction > 0:
            entry_side = OrderSide.BUY
            sl_px = last_price - exit_params.stop_points
            tp_px = last_price + exit_params.target_points
        else:
            entry_side = OrderSide.SELL
            sl_px = last_price + exit_params.stop_points
            tp_px = last_price - exit_params.target_points

        def _snap(p: float) -> str:
            snapped = round(p / tick) * tick
            return f"{snapped:.2f}"

        contracts = (
            self._position_sizer.compute_contracts(exit_params.stop_points) if self._position_sizer is not None else 1
        )
        bracket = self.order_factory.bracket(
            instrument_id=self._instrument_id,
            order_side=entry_side,
            quantity=Quantity.from_int(contracts),
            sl_trigger_price=Price.from_str(_snap(sl_px)),
            tp_price=Price.from_str(_snap(tp_px)),
            time_in_force=TimeInForce.GTC,
        )
        # Record the signal timestamp keyed by this bracket's order list
        # id. Every child order in the bracket (entry market + stop +
        # target) carries this same id, so the fill-time harvester can
        # recover signal_ts uniformly across entries and exits.
        self._signal_ts_by_order_list[bracket.id] = signal.timestamp
        self.submit_order_list(bracket)
        self._orders_submitted += 1
        M.signals_fired_total.labels(
            strategy=self._strategy_name,
            direction=str(signal.direction),
        ).inc()
        M.orders_submitted_total.labels(type="entry").inc()
        M.orders_submitted_total.labels(type="stop").inc()
        M.orders_submitted_total.labels(type="target").inc()
        # Backtest: bar-to-order latency is effectively zero; live will
        # measure wall-clock from bar close to broker ack.
        M.bar_to_order_seconds.observe(0.0)


class NautilusBacktestRunner:
    """Wraps a BacktestEngine so BaseStrategy subclasses can run against
    it without importing NautilusTrader.
    """

    def __init__(
        self,
        *,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        instrument_symbol: str,
        starting_balance_usd: float,
        risk_caps: RiskCaps | None = None,
        risk_per_trade_pct: float | None = None,
        max_contracts: int = 1,
    ) -> None:
        self.strategy = strategy
        self.data = data
        self.instrument_symbol = instrument_symbol
        self.starting_balance_usd = starting_balance_usd
        self.risk_caps = risk_caps or RiskCaps(max_stop_pts=5.0, min_target_pts=2.5, min_target_to_stop_ratio=2.0)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_contracts = max_contracts

    def _build_engine(self) -> BacktestEngine:
        config = BacktestEngineConfig(
            trader_id="ALPHA-ASSAY-BACKTEST",
            logging=LOGGING_CONFIG,
        )
        engine = BacktestEngine(config=config)
        engine.add_venue(
            venue=_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(self.starting_balance_usd, USD)],
            default_leverage=Decimal(1),
        )
        return engine

    def _build_instrument(self):
        if self.instrument_symbol.startswith("MNQ"):
            return make_mnq_futures(_VENUE, symbol=self.instrument_symbol)
        if self.instrument_symbol.startswith("MES"):
            return make_mes_futures(_VENUE, symbol=self.instrument_symbol)
        return make_es_futures(_VENUE, symbol=self.instrument_symbol)

    def run(self) -> BacktestResult:
        engine = self._build_engine()
        try:
            instrument = self._build_instrument()
            engine.add_instrument(instrument)
            bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
            bar_type = BarType(instrument.id, bar_spec, AggregationSource.EXTERNAL)
            bars = df_to_bars(self.data, bar_type)
            if "TICK" in self.data.columns and "ADD" in self.data.columns:
                ticks, adds = df_to_breadth(self.data, instrument.id)
                engine.add_data(bars)
                engine.add_data(ticks, client_id=_CLIENT_ID)
                engine.add_data(adds, client_id=_CLIENT_ID)
            else:
                engine.add_data(bars)
            multiplier = float(instrument.multiplier)
            position_sizer = PositionSizer(
                account_balance=self.starting_balance_usd,
                instrument_multiplier=multiplier,
                risk_per_trade_pct=self.risk_per_trade_pct,
                max_contracts=self.max_contracts,
            )
            adapter = _NautilusStrategyAdapter(
                bar_type=bar_type,
                instrument_id=instrument.id,
                strategy=self.strategy,
                risk_caps=self.risk_caps,
                position_sizer=position_sizer,
            )
            engine.add_strategy(adapter)
            engine.run()

            # Harvest results from Nautilus' cache into engine-agnostic
            # dicts + pd.Series. Callers (tests, metrics, CLI) never see
            # Nautilus types.
            cache = engine.cache
            signal_ts_by_order_list = adapter._signal_ts_by_order_list
            trades: list[dict[str, Any]] = []
            for order in cache.orders():
                if order.status.name != "FILLED":
                    continue
                fill_ts = pd.Timestamp(order.ts_last, tz="UTC")
                signal_ts = signal_ts_by_order_list.get(order.order_list_id)
                trades.append(
                    {
                        # `timestamp` preserved for backwards compatibility:
                        # downstream tooling (trades.csv readers, the report
                        # CLI's pair_trades) keys off it. signal_ts and
                        # fill_ts are additive enrichment for live-trading
                        # parity checks (Section 9: +/- 30s fill alignment).
                        "timestamp": fill_ts,
                        "signal_ts": signal_ts,
                        "fill_ts": fill_ts,
                        "side": order.side.name.lower(),
                        "price": float(order.avg_px) if order.avg_px is not None else None,
                        "quantity": float(order.quantity),
                        "order_type": order.order_type.name.lower(),
                    }
                )

            # Equity curve: sample account balance per filled trade. Fall
            # back to a two-point start/end curve on zero-trade runs so
            # plotting code never hits an empty series.
            account = engine.portfolio.account(_VENUE)
            final_balance = float(account.balance_total(USD).as_double()) if account else self.starting_balance_usd
            if trades:
                eq_idx = pd.DatetimeIndex([t["timestamp"] for t in trades])
                eq_vals = [self.starting_balance_usd] * len(trades)
                eq_vals[-1] = final_balance
                equity = pd.Series(eq_vals, index=eq_idx, name="equity_usd")
            else:
                eq_idx = pd.DatetimeIndex([self.data["timestamp"].iloc[0], self.data["timestamp"].iloc[-1]])
                equity = pd.Series(
                    [self.starting_balance_usd, final_balance],
                    index=eq_idx,
                    name="equity_usd",
                )

            return BacktestResult(
                trades=trades,
                session_metrics={
                    "run_status": "completed",
                    "submitted_signals": len(adapter.submitted_signals),
                    "orders_submitted": adapter._orders_submitted,
                    "signals_filtered_risk_cap": adapter._signals_filtered_risk_cap,
                    "final_balance_usd": final_balance,
                },
                equity_curve=equity,
            )
        finally:
            engine.dispose()
