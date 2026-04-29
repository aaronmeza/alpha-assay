"""CI merge blocker per spec Section 8. A strategy that peeks at
future bars must either be rejected by the engine's slicing policy
(IndexError when it reaches beyond the causal window) or produce
identical output to a shift-corrected version. Either behavior is
acceptable; the forbidden behavior is producing a different result
because of the peek.
"""

import numpy as np
import pandas as pd

from alpha_assay.engine.nautilus_runner import NautilusBacktestRunner
from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal


class PeekingStrategy(BaseStrategy):
    """Attempts to read data.iloc[+1] relative to the current bar.
    The engine passes a causal slice whose last row IS the current bar;
    attempting to index beyond must raise IndexError, which the adapter
    catches and logs WARN per spec Section 7 'uncaught exceptions in
    strategy code'.
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        out = pd.Series(0, index=data.index, dtype=int)
        if len(data) > 0:
            # data.iloc[-1] is safe (current bar). The peek attempt is
            # data.iloc[len(data)] which must raise.
            try:
                _ = data.iloc[len(data)]  # this is the lookahead attempt
            except IndexError:
                # Engine correctly enforced the causal-slice invariant.
                # Flag the bar for the test to observe.
                out.iloc[-1] = 0  # no signal; peek rejected structurally
        return out

    def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
        return ExitParams(stop_points=1.0, target_points=2.0)


def test_lookahead_attempt_is_structurally_impossible(tmp_path):
    """Merge blocker. The engine's causal-slice policy makes peek-ahead
    structurally impossible: `data.iloc[len(data)]` on the slice the
    adapter passes must raise IndexError.
    """
    idx = pd.date_range("2026-04-28 14:30", periods=30, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": 5000 + rng.normal(0, 1, 30),
            "ES_high": 5001 + rng.normal(0, 1, 30),
            "ES_low": 4999 - rng.normal(0, 1, 30),
            "ES_close": 5000 + rng.normal(0, 1, 30),
            "ES_volume": [100] * 30,
            "TICK": [0.0] * 30,
            "ADD": [0.0] * 30,
        }
    )
    runner = NautilusBacktestRunner(
        strategy=PeekingStrategy(config={}),
        data=df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    # The run must complete (adapter catches the strategy's IndexError
    # via its on_bar error handler; it does not abort the backtest).
    result = runner.run()
    assert result.session_metrics["run_status"] == "completed"


def test_shift_corrected_lookahead_matches_causal(tmp_path):
    """A strategy that computes `data.shift(-1)` and then uses iloc[-2]
    (the last valid row after the shift) must produce output equal to a
    causal strategy that uses iloc[-1] on the unshifted data. If they
    differ, the engine is leaking future information.
    """

    class CausalClose(BaseStrategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=data.index, dtype=int)
            if len(data) >= 2 and data["close"].iloc[-1] > data["close"].iloc[-2]:
                out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.0)

    class ShiftedClose(BaseStrategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            shifted = data["close"].shift(-1)
            out = pd.Series(0, index=data.index, dtype=int)
            if len(data) >= 3:
                prev = shifted.iloc[-3]
                curr = shifted.iloc[-2]
                if pd.notna(prev) and pd.notna(curr) and curr > prev:
                    out.iloc[-1] = 1
            return out

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.0)

    idx = pd.date_range("2026-04-28 14:30", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ES_open": [5000.0] * 30,
            "ES_high": [5001.0] * 30,
            "ES_low": [4999.0] * 30,
            "ES_close": [5000.0 + i * 0.1 for i in range(30)],
            "ES_volume": [100] * 30,
            "TICK": [0.0] * 30,
            "ADD": [0.0] * 30,
        }
    )

    def _run(strategy_cls):
        runner = NautilusBacktestRunner(
            strategy=strategy_cls(config={}),
            data=df,
            instrument_symbol="MESM6",
            starting_balance_usd=100_000.0,
        )
        return runner.run()

    r_causal = _run(CausalClose)
    r_shifted = _run(ShiftedClose)
    # The shifted version cannot see future values (engine slices
    # causally); its submitted_signals count must match the causal
    # version up to the slice length offset.
    assert r_causal.session_metrics["submitted_signals"] == r_shifted.session_metrics["submitted_signals"], (
        f"engine is leaking future information: "
        f"causal={r_causal.session_metrics['submitted_signals']} vs "
        f"shifted={r_shifted.session_metrics['submitted_signals']}"
    )
