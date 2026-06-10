# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Unit tests for the pluggable paper-strategy runner.

A scripted strategy and a fake exec adapter make every path
deterministic: signal -> bracket submission, sizing math + clamps,
session-mask edges, the max-one-position rule, realized P&L on
target / stop / flatten exits, flat-into-close, time-stop, startup
reconciliation, and the env-selection contract (no PAPER_STRATEGY ->
always-flat default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd
import pytest

from alpha_assay.config.loader import AlphaAssayConfig
from alpha_assay.exec.ibkr import OrderPlan
from alpha_assay.exec.trade_log import TradeLog
from alpha_assay.paper.runner import PaperStrategyRunner, load_paper_strategy, point_value_for
from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

CT = "America/Chicago"
# Tuesday 2026-06-02. Session window with 30/30 trim: 09:00 <= t < 14:30 CT.
SESSION_DATE = "2026-06-02"

CONTRACT_SPEC = {
    "symbol": "ES",
    "sec_type": "FUT",
    "exchange": "CME",
    "currency": "USD",
    "expiry": "20260618",
}


def _ct(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{SESSION_DATE} {hhmm}:00", tz=CT)


def _make_config(
    *,
    risk_per_trade_pct: float | None = 0.005,
    max_contracts: int = 20,
    stop_points: float = 2.0,
    target_points: float = 4.0,
) -> AlphaAssayConfig:
    return AlphaAssayConfig.model_validate(
        {
            "strategy": {
                "class": "tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy",
                "params": {
                    "signal": {"tick_window": 3, "tick_z_threshold": 1.0},
                    "risk": {"stop_points": stop_points, "target_points": target_points},
                },
            },
            "risk_caps": {
                "max_stop_pts": 5.0,
                "min_target_pts": 2.5,
                "min_target_to_stop_ratio": 2.0,
            },
            "session": {"minutes_after_open": 30, "minutes_before_close": 30},
            "execution": {
                "mode": "paper",
                "instrument": "ESM6",
                "risk_per_trade_pct": risk_per_trade_pct,
                "max_contracts": max_contracts,
            },
        }
    )


class ScriptedStrategy(BaseStrategy):
    """Fires +1/-1 at exact timestamps given in config["fire_at"].

    Exit params come from config["risk"]; config["time_stop_min"] adds a
    time-stop. Lets each test place signals at precise session minutes.
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fire_at: dict[pd.Timestamp, int] = self.config["fire_at"]
        result = pd.Series(0, index=data.index, dtype=int)
        for ts, direction in fire_at.items():
            if ts in result.index:
                result[ts] = direction
        return result

    def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
        risk = self.config["risk"]
        time_stop_min = self.config.get("time_stop_min")
        return ExitParams(
            stop_points=float(risk["stop_points"]),
            target_points=float(risk["target_points"]),
            time_stop=timedelta(minutes=time_stop_min) if time_stop_min else None,
        )


@dataclass
class FakeExecAdapter:
    """Records order activity; fires canned fills on demand."""

    next_id: int = 100
    brackets: list[dict[str, Any]] = field(default_factory=list)
    market_orders: list[dict[str, Any]] = field(default_factory=list)
    cancelled: list[int] = field(default_factory=list)
    open_position_qty: float = 0.0
    open_order_count: int = 0
    _callbacks: list = field(default_factory=list)

    def _allocate(self) -> int:
        self.next_id += 1
        return self.next_id

    def place_bracket_order(self, **kwargs: Any) -> OrderPlan:
        self.brackets.append(kwargs)
        side = kwargs["side"]
        sign = 1.0 if side == "BUY" else -1.0
        entry = float(kwargs["limit_price"])
        return OrderPlan(
            parent_id=self._allocate(),
            stop_id=self._allocate(),
            target_id=self._allocate(),
            side=side,
            quantity=kwargs["quantity"],
            entry_price=entry,
            stop_price=entry - sign * kwargs["stop_points"],
            target_price=entry + sign * kwargs["target_points"],
        )

    def place_market_order(self, **kwargs: Any) -> int:
        order_id = self._allocate()
        self.market_orders.append({**kwargs, "order_id": order_id})
        return order_id

    def cancel_order(self, order_id: int) -> None:
        self.cancelled.append(order_id)

    def position_quantity(self, contract_spec: dict[str, Any]) -> float:
        return self.open_position_qty

    def cancel_open_orders(self, contract_spec: dict[str, Any]) -> int:
        return self.open_order_count

    def on_fill(self, callback) -> None:
        self._callbacks.append(callback)

    def fire_fill(self, order_id: int, price: float, quantity: float, ts: str | pd.Timestamp) -> None:
        evt = {
            "order_id": order_id,
            "order_type": "MKT",
            "exec_id": f"exec-{order_id}",
            "timestamp": pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts),
            "quantity": quantity,
            "price": price,
            "side": "BOT",
        }
        for cb in self._callbacks:
            cb(evt)


def _make_runner(
    *,
    fire_at: dict[pd.Timestamp, int] | None = None,
    config: AlphaAssayConfig | None = None,
    trade_log: TradeLog | None = None,
    time_stop_min: int | None = None,
) -> tuple[PaperStrategyRunner, FakeExecAdapter]:
    config = config or _make_config()
    strategy = ScriptedStrategy(
        {
            "fire_at": fire_at or {},
            "risk": {"stop_points": 2.0, "target_points": 4.0},
            "time_stop_min": time_stop_min,
        }
    )
    exec_adapter = FakeExecAdapter()
    runner = PaperStrategyRunner(
        strategy=strategy,
        config=config,
        exec_adapter=exec_adapter,
        contract_spec=CONTRACT_SPEC,
        trade_log=trade_log,
        starting_balance=100_000.0,
    )
    exec_adapter.on_fill(runner.handle_fill)
    return runner, exec_adapter


def _feed_minutes(
    runner: PaperStrategyRunner,
    start_ct: pd.Timestamp,
    n_minutes: int,
    *,
    close: float = 5000.0,
    tick: float = 100.0,
    add: float = 500.0,
) -> None:
    """Feed `n_minutes` of bus-shaped events plus one trailing breadth
    minute so the final row's breadth closes emit (causal rollover).
    """
    minutes = [start_ct + pd.Timedelta(minutes=i) for i in range(n_minutes + 1)]
    for i, minute in enumerate(minutes):
        utc = minute.tz_convert("UTC")
        runner.on_breadth_tick({"timestamp": utc, "value": tick, "symbol": "TICK-NYSE"}, feed_label="tick_nyse")
        runner.on_breadth_tick({"timestamp": utc, "value": add, "symbol": "AD-NYSE"}, feed_label="ad_nyse")
        if i < n_minutes:
            runner.on_bar(
                {
                    "timestamp": utc,
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 2.0,
                    "close": close,
                    "volume": 100,
                },
                feed_label="es",
            )


# --- signal -> bracket submission ----------------------------------------


def test_signal_fires_bracket_in_session():
    fire = _ct("10:05")
    runner, exec_adapter = _make_runner(fire_at={fire: 1})

    _feed_minutes(runner, _ct("10:00"), 8)

    assert len(exec_adapter.brackets) == 1
    bracket = exec_adapter.brackets[0]
    assert bracket["side"] == "BUY"
    assert bracket["entry_type"] == "MARKET"
    assert bracket["stop_points"] == 2.0
    assert bracket["target_points"] == 4.0
    assert bracket["limit_price"] == 5000.0
    assert [s.timestamp for s in runner.fired_signals] == [fire]


def test_short_signal_submits_sell():
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): -1})

    _feed_minutes(runner, _ct("10:00"), 8)

    assert len(exec_adapter.brackets) == 1
    assert exec_adapter.brackets[0]["side"] == "SELL"


def test_no_signal_no_orders():
    runner, exec_adapter = _make_runner(fire_at={})

    _feed_minutes(runner, _ct("10:00"), 10)

    assert exec_adapter.brackets == []
    assert exec_adapter.market_orders == []


# --- sizing ----------------------------------------------------------------


def test_sizing_math_risk_budget_over_stop_dollars():
    # 100k * 0.5% = $500 risk; stop 2.0 pts * $50 = $100 -> 5 contracts.
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1})

    _feed_minutes(runner, _ct("10:00"), 8)

    assert exec_adapter.brackets[0]["quantity"] == 5


def test_sizing_clamps_to_max_contracts():
    config = _make_config(risk_per_trade_pct=0.05, max_contracts=20)
    # $5000 risk / $100 stop = 50 -> clamped to 20.
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1}, config=config)

    _feed_minutes(runner, _ct("10:00"), 8)

    assert exec_adapter.brackets[0]["quantity"] == 20


def test_sizing_floors_at_one_contract():
    config = _make_config(risk_per_trade_pct=0.00001)
    # $1 risk / $100 stop -> floor(0) -> clamped up to 1.
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1}, config=config)

    _feed_minutes(runner, _ct("10:00"), 8)

    assert exec_adapter.brackets[0]["quantity"] == 1


def test_point_value_prefix_match():
    assert point_value_for("ESM6") == 50.0
    assert point_value_for("MESM6") == 5.0
    assert point_value_for("MNQM6") == 2.0
    with pytest.raises(ValueError):
        point_value_for("ZB")


# --- session-mask edges ------------------------------------------------------


def test_signal_in_first_30_minutes_is_filtered():
    # 08:45 CT is inside RTH but before the 30-min post-open trim.
    runner, exec_adapter = _make_runner(fire_at={_ct("08:45"): 1})

    _feed_minutes(runner, _ct("08:40"), 8)

    assert exec_adapter.brackets == []


def test_signal_in_last_30_minutes_is_filtered():
    # 14:45 CT is inside RTH but past the 30-min pre-close cutoff.
    runner, exec_adapter = _make_runner(fire_at={_ct("14:45"): 1})

    _feed_minutes(runner, _ct("14:40"), 8)

    assert exec_adapter.brackets == []


def test_signal_at_window_edges_acts_only_inside():
    # 09:00 is the first actionable minute; 14:29 the last.
    runner, exec_adapter = _make_runner(fire_at={_ct("09:00"): 1})
    _feed_minutes(runner, _ct("08:58"), 4)
    assert len(exec_adapter.brackets) == 1


# --- max one concurrent position ---------------------------------------------


def test_second_signal_while_position_open_is_skipped():
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1, _ct("10:10"): 1})

    _feed_minutes(runner, _ct("10:00"), 15)

    assert len(exec_adapter.brackets) == 1


def test_second_signal_after_round_trip_closes_is_accepted():
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1, _ct("10:10"): 1})

    _feed_minutes(runner, _ct("10:00"), 7)  # rows through 10:06; first fired
    plan_qty = exec_adapter.brackets[0]["quantity"]
    parent_id = runner.open_position.plan.parent_id
    target_id = runner.open_position.plan.target_id
    exec_adapter.fire_fill(parent_id, 5000.25, plan_qty, "2026-06-02 15:06:05+00:00")
    exec_adapter.fire_fill(target_id, 5004.25, plan_qty, "2026-06-02 15:08:00+00:00")
    assert runner.open_position is None

    _feed_minutes(runner, _ct("10:07"), 6)  # second signal at 10:10 now actionable

    assert len(exec_adapter.brackets) == 2


# --- realized P&L (90e.3) ------------------------------------------------------


def test_realized_pnl_on_target_hit_long(tmp_path):
    trade_log = TradeLog(out_dir=tmp_path)
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1}, trade_log=trade_log)

    _feed_minutes(runner, _ct("10:00"), 8)
    pos = runner.open_position
    qty = pos.contracts
    exec_adapter.fire_fill(pos.plan.parent_id, 5000.25, qty, "2026-06-02 15:06:05+00:00")
    exec_adapter.fire_fill(pos.plan.target_id, 5004.25, qty, "2026-06-02 15:09:00+00:00")

    expected = (5004.25 - 5000.25) * qty * 50.0
    assert runner.open_position is None
    assert runner.account_balance == pytest.approx(100_000.0 + expected)

    df = pd.read_parquet(tmp_path / "trades.parquet")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["signal_type"] == "long_entry"
    assert row["mock_fill_price"] == pytest.approx(5000.25)
    assert row["mock_pnl_dollars"] == pytest.approx(expected)
    assert row["account_balance_after"] == pytest.approx(100_000.0 + expected)


def test_realized_pnl_on_stop_hit_long(tmp_path):
    trade_log = TradeLog(out_dir=tmp_path)
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1}, trade_log=trade_log)

    _feed_minutes(runner, _ct("10:00"), 8)
    pos = runner.open_position
    qty = pos.contracts
    exec_adapter.fire_fill(pos.plan.parent_id, 5000.25, qty, "2026-06-02 15:06:05+00:00")
    exec_adapter.fire_fill(pos.plan.stop_id, 4998.25, qty, "2026-06-02 15:12:00+00:00")

    expected = (4998.25 - 5000.25) * qty * 50.0  # negative
    assert expected < 0
    assert runner.account_balance == pytest.approx(100_000.0 + expected)
    df = pd.read_parquet(tmp_path / "trades.parquet")
    assert df.iloc[0]["mock_pnl_dollars"] == pytest.approx(expected)


def test_realized_pnl_short_round_trip():
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): -1})

    _feed_minutes(runner, _ct("10:00"), 8)
    pos = runner.open_position
    qty = pos.contracts
    exec_adapter.fire_fill(pos.plan.parent_id, 4999.75, qty, "2026-06-02 15:06:05+00:00")
    exec_adapter.fire_fill(pos.plan.target_id, 4995.75, qty, "2026-06-02 15:20:00+00:00")

    expected = (4995.75 - 4999.75) * -1 * qty * 50.0  # short profits on the drop
    assert expected > 0
    assert runner.account_balance == pytest.approx(100_000.0 + expected)


def test_trades_parquet_remains_dashboard_loadable(tmp_path):
    """The dashboard loader contract must keep working on runner output."""
    from dashboard.loaders import load_trades

    trade_log = TradeLog(out_dir=tmp_path)
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1}, trade_log=trade_log)
    _feed_minutes(runner, _ct("10:00"), 8)
    pos = runner.open_position
    exec_adapter.fire_fill(pos.plan.parent_id, 5000.25, pos.contracts, "2026-06-02 15:06:05+00:00")
    exec_adapter.fire_fill(pos.plan.target_id, 5004.25, pos.contracts, "2026-06-02 15:09:00+00:00")

    df = load_trades(tmp_path)
    assert len(df) == 1
    assert df["mock_pnl_dollars"].iloc[0] > 0
    assert str(df["timestamp"].dtype).startswith("datetime64[ns, UTC]")


# --- flat-into-close -----------------------------------------------------------


def test_flat_into_close_cancels_children_and_flattens():
    # Fire at 14:25; cutoff row 14:30 must cancel children + market-flatten.
    runner, exec_adapter = _make_runner(fire_at={_ct("14:25"): 1})

    _feed_minutes(runner, _ct("14:20"), 12)

    assert len(exec_adapter.brackets) == 1
    assert len(exec_adapter.market_orders) == 1
    flatten = exec_adapter.market_orders[0]
    assert flatten["side"] == "SELL"
    assert flatten["quantity"] == exec_adapter.brackets[0]["quantity"]
    assert len(exec_adapter.cancelled) == 2  # stop + target children

    # The flatten fill realizes P&L with the flat_close outcome.
    pos = runner.open_position
    exec_adapter.fire_fill(pos.plan.parent_id, 5000.25, pos.contracts, "2026-06-02 19:26:05+00:00")
    exec_adapter.fire_fill(pos.flatten_order_id, 5001.25, pos.contracts, "2026-06-02 19:31:00+00:00")
    assert runner.open_position is None
    assert runner.account_balance > 100_000.0


def test_flatten_submitted_once_not_every_bar():
    runner, exec_adapter = _make_runner(fire_at={_ct("14:25"): 1})

    _feed_minutes(runner, _ct("14:20"), 20)

    assert len(exec_adapter.market_orders) == 1


# --- time-stop -------------------------------------------------------------------


def test_time_stop_flattens_after_configured_minutes():
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1}, time_stop_min=3)

    _feed_minutes(runner, _ct("10:00"), 12)

    assert len(exec_adapter.market_orders) == 1
    assert len(exec_adapter.cancelled) == 2
    pos = runner.open_position
    assert pos.flatten_reason == "time_stop"


# --- risk caps ----------------------------------------------------------------


def test_risk_cap_violation_drops_signal():
    runner, exec_adapter = _make_runner(fire_at={_ct("10:05"): 1})
    # Stop wider than max_stop_pts must be rejected before submission.
    runner._strategy.config["risk"] = {"stop_points": 10.0, "target_points": 20.0}

    _feed_minutes(runner, _ct("10:00"), 8)

    assert exec_adapter.brackets == []


# --- startup reconciliation -------------------------------------------------------


def test_startup_reconcile_flattens_stale_position_and_cancels_orders():
    runner, exec_adapter = _make_runner()
    exec_adapter.open_position_qty = 2.0
    exec_adapter.open_order_count = 3

    qty, cancelled = runner.startup_reconcile()

    assert qty == 2.0
    assert cancelled == 3
    assert len(exec_adapter.market_orders) == 1
    assert exec_adapter.market_orders[0]["side"] == "SELL"
    assert exec_adapter.market_orders[0]["quantity"] == 2


def test_startup_reconcile_noop_when_flat():
    runner, exec_adapter = _make_runner()

    qty, cancelled = runner.startup_reconcile()

    assert (qty, cancelled) == (0.0, 0)
    assert exec_adapter.market_orders == []


# --- env selection contract ---------------------------------------------------------


def test_load_paper_strategy_returns_none_when_unset():
    assert load_paper_strategy({}) is None
    assert load_paper_strategy({"PAPER_STRATEGY": ""}) is None


def test_load_paper_strategy_requires_config_path():
    with pytest.raises(ValueError, match="PAPER_STRATEGY_CONFIG"):
        load_paper_strategy({"PAPER_STRATEGY": "tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy"})


def test_load_paper_strategy_instantiates_class_from_yaml(tmp_path):
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("""
strategy:
  class: tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy
  params:
    signal:
      tick_window: 5
      tick_z_threshold: 1.5
risk_caps:
  max_stop_pts: 5.0
  min_target_pts: 2.5
  min_target_to_stop_ratio: 2.0
session:
  minutes_after_open: 30
  minutes_before_close: 30
execution:
  mode: paper
  instrument: ESM6
  risk_per_trade_pct: 0.005
  max_contracts: 20
""")
    loaded = load_paper_strategy(
        {
            "PAPER_STRATEGY": "tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy",
            "PAPER_STRATEGY_CONFIG": str(config_path),
        }
    )
    assert loaded is not None
    assert type(loaded.strategy).__name__ == "BreadthAwareTestStrategy"
    assert loaded.config.execution.max_contracts == 20


# --- equivalence: runner fires exactly what the strategy emits ----------------------


def test_fired_signals_match_generate_signals_over_end_state_frame():
    """Replay a deterministic frame; the runner's fired set must equal the
    in-session non-zero signals of generate_signals over the frame it
    built (with fills closing each trade so max-1-position never gates)."""
    fire_at = {_ct("10:05"): 1, _ct("11:00"): -1, _ct("08:45"): 1}  # 08:45 out of window
    runner, exec_adapter = _make_runner(fire_at=fire_at)

    start = _ct("08:40")
    n = 150  # through ~11:10
    cursor = start
    for _ in range(n):
        _feed_minutes(runner, cursor, 1)
        cursor += pd.Timedelta(minutes=1)
        pos = runner.open_position
        if pos is not None and pos.entry_qty == 0:
            # Fill entry + target immediately so the next signal is not
            # blocked by the position gate.
            exec_adapter.fire_fill(pos.plan.parent_id, 5000.25, pos.contracts, cursor.tz_convert("UTC"))
            exec_adapter.fire_fill(pos.plan.target_id, 5004.25, pos.contracts, cursor.tz_convert("UTC"))

    frame = runner.frame
    signals = runner._strategy.generate_signals(frame)
    from alpha_assay.filters.session_mask import session_mask

    in_window = session_mask(frame.index, minutes_after_open=30, minutes_before_close=30)
    expected = [(ts, int(v)) for ts, v in signals[in_window & (signals != 0)].items()]
    fired = [(s.timestamp, s.direction) for s in runner.fired_signals]
    assert fired == expected
    assert len(fired) == 2  # the 08:45 signal was masked out
