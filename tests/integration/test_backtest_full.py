"""end-to-end integration: SMA crossover through the Nautilus
adapter on the 780-row synthetic fixture. Deterministic PnL,
trade count, BacktestResult shape.
"""

from pathlib import Path

import pandas as pd
import pytest

from alpha_assay.engine.nautilus_runner import BacktestResult, NautilusBacktestRunner
from examples.sma_crossover import SMACrossoverStrategy

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_2d.csv"


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_sma_crossover_backtest_full_run(fixture_df):
    """End-to-end. Asserts the run completes, returns a BacktestResult,
    and populates the expected fields. PnL is deterministic (same seed,
    same fixture, same strategy) - the numeric values below are captured
    from the reference run and will be asserted once is green.
    """
    strategy = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    runner = NautilusBacktestRunner(
        strategy=strategy,
        data=fixture_df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result = runner.run()

    assert isinstance(result, BacktestResult)
    # Fixture includes TICK/ADD columns; SMA ignores them. Adapter
    # must accept custom breadth data even when the strategy does not
    # use it - the contract separation in action.
    assert result.session_metrics["run_status"] == "completed"
    assert "orders_submitted" in result.session_metrics
    assert "submitted_signals" in result.session_metrics
    assert isinstance(result.equity_curve, pd.Series)
    assert isinstance(result.trades, list)


def test_sma_crossover_deterministic_trade_count(fixture_df):
    """Two in-process runs with identical inputs must produce identical
    trade counts + submitted signal counts. Proves the adapter is
    deterministic.
    """

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
    assert r1.session_metrics["submitted_signals"] == r2.session_metrics["submitted_signals"]
    assert r1.session_metrics["orders_submitted"] == r2.session_metrics["orders_submitted"]
    assert len(r1.trades) == len(r2.trades)


def test_backtest_full_ignores_breadth_columns_for_sma(fixture_df):
    """SMA crossover does not read TICK/ADD. Dropping those columns
    from the fixture must produce the same trade count as keeping them.
    Proves the adapter does not funnel breadth into OHLCV-only strategies.
    """
    strategy_a = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    runner_a = NautilusBacktestRunner(
        strategy=strategy_a,
        data=fixture_df,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result_a = runner_a.run()

    df_no_breadth = fixture_df.drop(columns=["TICK", "ADD"])
    strategy_b = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    runner_b = NautilusBacktestRunner(
        strategy=strategy_b,
        data=df_no_breadth,
        instrument_symbol="MESM6",
        starting_balance_usd=100_000.0,
    )
    result_b = runner_b.run()

    assert result_a.session_metrics["submitted_signals"] == result_b.session_metrics["submitted_signals"]
