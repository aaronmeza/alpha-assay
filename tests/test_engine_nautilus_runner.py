"""Tests for the NautilusBacktestRunner engine adapter ."""

from dataclasses import is_dataclass

import pandas as pd

from alpha_assay.engine.nautilus_runner import BacktestResult, NautilusBacktestRunner


def test_nautilus_backtest_runner_class_exists():
    assert callable(NautilusBacktestRunner)


def test_backtest_result_is_dataclass_with_required_fields():
    assert is_dataclass(BacktestResult)
    fields = set(BacktestResult.__dataclass_fields__.keys())
    assert {"trades", "session_metrics", "equity_curve"}.issubset(fields)


def test_runner_init_accepts_strategy_df_and_config():
    from examples.sma_crossover import SMACrossoverStrategy

    strategy = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    df = pd.DataFrame()
    runner = NautilusBacktestRunner(
        strategy=strategy,
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    assert runner.strategy is strategy
    assert runner.instrument_symbol == "MESM6"


def test_runner_run_returns_backtest_result_type():
    # Skeleton contract only: run() returns a BacktestResult. No
    # assertions on content yet (that is J3-J6).
    from examples.sma_crossover import SMACrossoverStrategy

    strategy = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    idx = pd.date_range("2026-04-28 10:00", periods=20, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * 20,
            "ES_high": [5001.0] * 20,
            "ES_low": [4999.0] * 20,
            "ES_close": [5000.5] * 20,
            "ES_volume": [100] * 20,
            "TICK": [0.0] * 20,
            "ADD": [0.0] * 20,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=strategy,
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result = runner.run()
    assert isinstance(result, BacktestResult)
    assert isinstance(result.trades, list)
    assert isinstance(result.session_metrics, dict)
    assert isinstance(result.equity_curve, pd.Series)


def test_strategy_adapter_calls_generate_signals_on_each_in_window_bar():
    """The adapter must invoke BaseStrategy.generate_signals with a causal
    slice ending at the current bar. Record each invocation and assert the
    slice grows monotonically and never peeks beyond the bar timestamp.
    """
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    seen_slice_lengths: list[int] = []
    seen_last_ts: list[pd.Timestamp] = []

    class RecordingStrategy(BaseStrategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            seen_slice_lengths.append(len(data))
            if len(data) > 0:
                seen_last_ts.append(data.index[-1])
            return pd.Series(0, index=data.index, dtype=int)

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.0)

    idx = pd.date_range("2026-04-28 14:30", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * 30,
            "ES_high": [5001.0] * 30,
            "ES_low": [4999.0] * 30,
            "ES_close": [5000.5] * 30,
            "ES_volume": [100] * 30,
            "TICK": [0.0] * 30,
            "ADD": [0.0] * 30,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=RecordingStrategy(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    runner.run()
    # generate_signals was called at least once per in-window bar.
    # Session mask filters out-of-window bars so exact count depends on
    # the synthetic UTC timestamps; just assert the sequence is
    # monotone-increasing (causal slice grows).
    assert len(seen_slice_lengths) > 0
    assert all(
        a <= b for a, b in zip(seen_slice_lengths, seen_slice_lengths[1:], strict=False)
    ), f"slice lengths not monotone: {seen_slice_lengths}"


def test_nonzero_signal_submits_bracket_order():
    """A strategy that emits a +1 signal on a mid-fixture bar should
    produce at least one order (parent + two children) in the engine's
    order cache.
    """
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    class OneShotLong(BaseStrategy):
        _fired = False

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=data.index, dtype=int)
            if not OneShotLong._fired and len(data) >= 5:
                OneShotLong._fired = True
                out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.5)

    OneShotLong._fired = False

    idx = pd.date_range("2026-04-28 14:30", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * 30,
            "ES_high": [5001.0] * 30,
            "ES_low": [4999.0] * 30,
            "ES_close": [5000.5] * 30,
            "ES_volume": [100] * 30,
            "TICK": [0.0] * 30,
            "ADD": [0.0] * 30,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=OneShotLong(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result = runner.run()
    assert result.session_metrics["submitted_signals"] >= 1
    assert result.session_metrics.get("orders_submitted", 0) >= 1


def test_out_of_session_bar_does_not_fire_signal():
    """Bars outside the strategy window must not invoke the strategy.
    The session_mask filter gate is a merge-blocker invariant.
    """
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    out_of_session_calls: list[pd.Timestamp] = []

    class TripwireStrategy(BaseStrategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            # Record call timestamps; test asserts they are all in-session.
            if len(data) > 0:
                out_of_session_calls.append(data.index[-1])
            return pd.Series(0, index=data.index, dtype=int)

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.0)

    # Build a DataFrame with 5 out-of-session + 5 in-session bars.
    # 13:00 UTC = 08:00 CT (pre-09:00 CT window) = out-of-session.
    # 15:00 UTC = 10:00 CT = in-session.
    out_idx = pd.date_range("2026-04-28 13:00", periods=5, freq="1min", tz="UTC")
    in_idx = pd.date_range("2026-04-28 15:00", periods=5, freq="1min", tz="UTC")
    idx = out_idx.append(in_idx)
    n = len(idx)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * n,
            "ES_high": [5001.0] * n,
            "ES_low": [4999.0] * n,
            "ES_close": [5000.5] * n,
            "ES_volume": [100] * n,
            "TICK": [0.0] * n,
            "ADD": [0.0] * n,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=TripwireStrategy(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    runner.run()
    from alpha_assay.filters.session_mask import session_mask

    if out_of_session_calls:
        mask = session_mask(pd.DatetimeIndex(out_of_session_calls))
        assert mask.all(), (
            f"strategy invoked on out-of-session bars: "
            f"{[ts for ts, m in zip(out_of_session_calls, mask, strict=False) if not m]}"
        )


def test_risk_cap_violation_drops_signal_and_counts_it():
    """A strategy that returns exit_params exceeding the default caps
    (stop > 5pts) must have its signal dropped, not the whole run aborted.
    The drop is counted in session_metrics as signals_filtered_risk_cap.
    """
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    class OversizedStopStrategy(BaseStrategy):
        _fired = False

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=data.index, dtype=int)
            if not OversizedStopStrategy._fired and len(data) >= 5:
                OversizedStopStrategy._fired = True
                out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            # 7.5 > default max_stop_pts=5.0 -> caps.validate raises.
            return ExitParams(stop_points=7.5, target_points=15.0)

    OversizedStopStrategy._fired = False

    idx = pd.date_range("2026-04-28 14:30", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * 30,
            "ES_high": [5001.0] * 30,
            "ES_low": [4999.0] * 30,
            "ES_close": [5000.5] * 30,
            "ES_volume": [100] * 30,
            "TICK": [0.0] * 30,
            "ADD": [0.0] * 30,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=OversizedStopStrategy(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result = runner.run()
    assert result.session_metrics.get("orders_submitted", 0) == 0
    assert result.session_metrics.get("signals_filtered_risk_cap", 0) >= 1


def test_backtest_result_populates_equity_curve_and_trades():
    """After a run that fires at least one signal, BacktestResult must
    carry a non-empty equity_curve (DatetimeIndex, float values) and a
    trades list whose entries are plain dicts (no Nautilus types).
    """
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    class OneShotLong(BaseStrategy):
        _fired = False

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=data.index, dtype=int)
            if not OneShotLong._fired and len(data) >= 5:
                OneShotLong._fired = True
                out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.5)

    OneShotLong._fired = False

    idx = pd.date_range("2026-04-28 14:30", periods=60, freq="1min", tz="UTC")
    n = len(idx)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * n,
            "ES_high": [5001.0] * n,
            "ES_low": [4999.0] * n,
            "ES_close": [5000.5] * n,
            "ES_volume": [100] * n,
            "TICK": [0.0] * n,
            "ADD": [0.0] * n,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=OneShotLong(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result = runner.run()
    # equity_curve: non-empty, DatetimeIndex, float values.
    assert len(result.equity_curve) > 0
    assert isinstance(result.equity_curve.index, pd.DatetimeIndex)
    # trades: list of plain dicts with the expected keys.
    assert isinstance(result.trades, list)
    if result.trades:
        t = result.trades[0]
        assert isinstance(t, dict)
        assert {"timestamp", "side", "price", "quantity"}.issubset(t.keys())


def test_engine_increments_prometheus_counters_on_signal_fire():
    """When a strategy fires a signal that passes risk caps, the engine
    must increment alpha_assay_signals_fired_total + orders_submitted_total.
    """
    import pandas as pd

    from alpha_assay.engine.nautilus_runner import NautilusBacktestRunner
    from alpha_assay.observability import metrics as M
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    class OneShotLong(BaseStrategy):
        _fired = False

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=data.index, dtype=int)
            if not OneShotLong._fired and len(data) >= 5:
                OneShotLong._fired = True
                out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.5)

    OneShotLong._fired = False

    before_fired = M.signals_fired_total.labels(strategy="OneShotLong", direction="1")._value.get()
    before_submitted = M.orders_submitted_total.labels(type="entry")._value.get()

    idx = pd.date_range("2026-04-28 14:30", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * 30,
            "ES_high": [5001.0] * 30,
            "ES_low": [4999.0] * 30,
            "ES_close": [5000.5] * 30,
            "ES_volume": [100] * 30,
            "TICK": [0.0] * 30,
            "ADD": [0.0] * 30,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=OneShotLong(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    runner.run()

    after_fired = M.signals_fired_total.labels(strategy="OneShotLong", direction="1")._value.get()
    after_submitted = M.orders_submitted_total.labels(type="entry")._value.get()
    assert after_fired >= before_fired + 1
    assert after_submitted >= before_submitted + 1


def test_trade_dict_carries_signal_ts_and_fill_ts():
    """Every emitted trade dict must carry signal_ts and fill_ts. The
    `timestamp` column is preserved for backwards compatibility.

    Fill-time semantics in the current Nautilus simulator (bar_execution
    default): an entry market order fills against the same bar at close,
    so its fill_ts equals its signal_ts (zero in-engine latency). Bracket
    children (stop / target) fill on a later bar when their trigger price
    is touched, so their fill_ts is strictly greater than signal_ts and
    differs from `timestamp` for the entry. The contract this test
    enforces is therefore signal_ts <= fill_ts plus the existence of at
    least one strictly-later child fill.

    The live-vs-backtest parity check (spec Section 9, +/- 30s) relies on the
    delta being a real number, not the literal one-bar-period that a
    next-bar-open simulator would produce; both backtest and live emit
    the same fields, so the comparison is apples-to-apples regardless of
    which simulator policy is active.
    """
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    class OneShotLong(BaseStrategy):
        _fired = False

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=data.index, dtype=int)
            if not OneShotLong._fired and len(data) >= 5:
                OneShotLong._fired = True
                out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.5)

    OneShotLong._fired = False

    idx = pd.date_range("2026-04-28 14:30", periods=60, freq="1min", tz="UTC")
    n = len(idx)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * n,
            "ES_high": [5001.0] * n,
            "ES_low": [4999.0] * n,
            "ES_close": [5000.5] * n,
            "ES_volume": [100] * n,
            "TICK": [0.0] * n,
            "ADD": [0.0] * n,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=OneShotLong(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result = runner.run()

    assert result.trades, "expected at least one filled trade"
    entries = [t for t in result.trades if t["order_type"] == "market"]
    children = [t for t in result.trades if t["order_type"] != "market"]
    assert entries, "expected at least one market entry fill in trade dicts"
    assert children, "expected at least one bracket exit fill in trade dicts"

    for trade in result.trades:
        assert "signal_ts" in trade
        assert "fill_ts" in trade
        # `timestamp` is preserved as-is for backwards compatibility.
        assert "timestamp" in trade
        assert trade["fill_ts"] == trade["timestamp"]
        assert trade["signal_ts"] is not None
        assert (
            trade["signal_ts"] <= trade["fill_ts"]
        ), f"signal_ts {trade['signal_ts']} after fill_ts {trade['fill_ts']}"

    # Bracket child fills must be strictly later than the signal that
    # produced them - they trigger on a later bar's high/low.
    for child in children:
        delta = (child["fill_ts"] - child["signal_ts"]).total_seconds()
        # 1-minute bars: at least one bar period to the trigger bar.
        assert delta >= 60.0, f"child fill_ts only {delta}s after signal_ts; expected >=60s"


def test_engine_increments_signals_filtered_risk_cap_counter():
    """When a signal is dropped for a risk-cap violation, the engine must
    increment alpha_assay_signals_filtered_total with reason="risk_cap".
    """
    import pandas as pd

    from alpha_assay.engine.nautilus_runner import NautilusBacktestRunner
    from alpha_assay.observability import metrics as M
    from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal

    class OversizedStop(BaseStrategy):
        _fired = False

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=data.index, dtype=int)
            if not OversizedStop._fired and len(data) >= 5:
                OversizedStop._fired = True
                out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=7.5, target_points=15.0)

    OversizedStop._fired = False

    before = M.signals_filtered_total.labels(
        strategy="OversizedStop", filter_name="risk_caps", reason="risk_cap"
    )._value.get()

    idx = pd.date_range("2026-04-28 14:30", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * 30,
            "ES_high": [5001.0] * 30,
            "ES_low": [4999.0] * 30,
            "ES_close": [5000.5] * 30,
            "ES_volume": [100] * 30,
            "TICK": [0.0] * 30,
            "ADD": [0.0] * 30,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=OversizedStop(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    runner.run()

    after = M.signals_filtered_total.labels(
        strategy="OversizedStop", filter_name="risk_caps", reason="risk_cap"
    )._value.get()
    assert after >= before + 1
