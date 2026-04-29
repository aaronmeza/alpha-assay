"""Byte-equal replay check: same config, same data, two runs produce
identical trade timestamps, fill prices, and PnL. Required by spec
Section 8 'test_deterministic_replay'.
"""

from pathlib import Path

import pandas as pd
import pytest

from alpha_assay.engine.nautilus_runner import NautilusBacktestRunner
from examples.sma_crossover import SMACrossoverStrategy

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_2d.csv"


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_two_in_process_runs_byte_equal(fixture_df):
    def _run():
        strategy = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
        runner = NautilusBacktestRunner(
            strategy=strategy,
            data=fixture_df,
            instrument_symbol="MESM6",
            starting_balance_usd=100_000.0,
        )
        return runner.run()

    r1 = _run()
    r2 = _run()

    assert len(r1.trades) == len(r2.trades)
    for t1, t2 in zip(r1.trades, r2.trades, strict=False):
        assert (
            t1["timestamp"] == t2["timestamp"]
        ), f"trade timestamp drift: {t1['timestamp']} vs {t2['timestamp']}"
        assert t1["price"] == t2["price"], f"fill price drift: {t1['price']} vs {t2['price']}"
        assert t1["quantity"] == t2["quantity"]
        assert t1["side"] == t2["side"]

    assert r1.session_metrics["final_balance_usd"] == r2.session_metrics["final_balance_usd"]


def test_equity_curve_byte_equal(fixture_df):
    def _run():
        strategy = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
        runner = NautilusBacktestRunner(
            strategy=strategy,
            data=fixture_df,
            instrument_symbol="MESM6",
            starting_balance_usd=100_000.0,
        )
        return runner.run()

    r1 = _run()
    r2 = _run()
    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)
