# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""End-to-end CLI test for the breadth-aware backtest path.

Synthesizes a small ES OHLCV parquet plus matching TICK and AD breadth
parquets in the recorder shard shape, then exercises:

1. --tick-data + --ad-data unset: SMA crossover path runs (backwards
   compat) with no breadth columns reaching the engine.
2. --tick-data + --ad-data set: a breadth-aware test strategy reads
   `data["TICK"]` and `data["ADD"]` through the CLI -> joined-loader ->
   runner -> generate_signals path, and produces the same trades.csv +
   session_metrics.json artifacts the report subcommand consumes.
3. Only one of the two breadth flags set: CLI fails with a usage error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from alpha_assay.cli import main as cli_main


def _write_es_parquet(out_path: Path) -> pd.DatetimeIndex:
    """Synthesize 2 RTH days of 1-min ES bars; return the index for join."""
    # NYSE core hours in CT: 08:30-14:59 inclusive = 390 bars/day.
    start = pd.Timestamp("2026-04-13 08:30", tz="America/Chicago")
    days = []
    for d in range(2):
        day_start = start + pd.Timedelta(days=d)
        days.append(pd.date_range(day_start, periods=390, freq="1min"))
    idx = days[0].append(days[1])
    n = len(idx)
    rng = np.random.default_rng(7)
    close = 5000 + np.cumsum(rng.normal(0, 1.0, n))
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.integers(50, 500, n),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), out_path)
    return idx


def _write_breadth_parquet(
    out_path: Path,
    idx: pd.DatetimeIndex,
    symbol: str,
    values: np.ndarray,
) -> None:
    """Write a recorder-shaped breadth shard.

    Schema matches `infra/recorders/`: timestamp (UTC), open, high, low,
    close, n_ticks, symbol. The `close` column is the per-minute breadth
    value the joined loader extracts.
    """
    n = len(idx)
    df = pd.DataFrame(
        {
            "timestamp": idx.tz_convert("UTC"),
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "n_ticks": np.full(n, 30, dtype=np.int64),
            "symbol": [symbol] * n,
        }
    )
    pq.write_table(pa.Table.from_pandas(df), out_path)


def _write_tick_breadth(out_path: Path, idx: pd.DatetimeIndex) -> None:
    """Synthesize a TICK series with a deliberate negative spike on day 1
    so the breadth-aware strategy fires at least once.
    """
    n = len(idx)
    rng = np.random.default_rng(11)
    base = rng.normal(0, 80, n)
    # Inject a deep negative tick spike around bar 60 (still inside
    # session-mask trading window: minutes_after_open=30 trims the first
    # 30 bars per day, so bar 60 is well past the gate).
    base[60] = -800.0
    base[61] = -750.0
    base[62] = -720.0
    _write_breadth_parquet(out_path, idx, "TICK-NYSE", base)


def _write_ad_breadth(out_path: Path, idx: pd.DatetimeIndex) -> None:
    """Synthesize an AD series that stays bullish (>0) so the bias gate
    in the breadth-aware test strategy is always engaged.
    """
    n = len(idx)
    values = np.full(n, 1500.0)
    _write_breadth_parquet(out_path, idx, "AD-NYSE", values)


def _write_sma_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "sma_config.yaml"
    cfg.write_text(
        "strategy:\n"
        "  class: examples.sma_crossover:SMACrossoverStrategy\n"
        "  params:\n"
        "    signal:\n"
        "      fast: 3\n"
        "      slow: 10\n"
        "risk_caps:\n"
        "  max_stop_pts: 5.0\n"
        "  min_target_pts: 2.5\n"
        "  min_target_to_stop_ratio: 2.0\n"
        "session:\n"
        "  minutes_after_open: 30\n"
        "  minutes_before_close: 30\n"
        "execution:\n"
        "  mode: backtest\n"
        "  instrument: MESM6\n"
    )
    return cfg


def _write_breadth_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "breadth_config.yaml"
    cfg.write_text(
        "strategy:\n"
        "  class: tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy\n"
        "  params:\n"
        "    signal:\n"
        "      tick_window: 10\n"
        "      tick_z_threshold: 1.5\n"
        "risk_caps:\n"
        "  max_stop_pts: 5.0\n"
        "  min_target_pts: 2.5\n"
        "  min_target_to_stop_ratio: 2.0\n"
        "session:\n"
        "  minutes_after_open: 30\n"
        "  minutes_before_close: 30\n"
        "execution:\n"
        "  mode: backtest\n"
        "  instrument: MESM6\n"
    )
    return cfg


def test_cli_backtest_omitting_breadth_flags_is_backwards_compat(tmp_path):
    es = tmp_path / "es.parquet"
    _write_es_parquet(es)
    cfg = _write_sma_config(tmp_path)
    out = tmp_path / "run_no_breadth"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backtest",
            "--strategy",
            "examples.sma_crossover:SMACrossoverStrategy",
            "--config",
            str(cfg),
            "--data",
            str(es),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "trades.csv").exists()
    metrics = json.loads((out / "session_metrics.json").read_text())
    assert metrics["run_status"] == "completed"


def test_cli_backtest_with_tick_and_ad_runs_breadth_aware_strategy(tmp_path):
    es = tmp_path / "es.parquet"
    idx = _write_es_parquet(es)

    tick = tmp_path / "tick.parquet"
    ad = tmp_path / "ad.parquet"
    _write_tick_breadth(tick, idx)
    _write_ad_breadth(ad, idx)

    cfg = _write_breadth_config(tmp_path)
    out = tmp_path / "run_breadth"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backtest",
            "--strategy",
            "tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy",
            "--config",
            str(cfg),
            "--data",
            str(es),
            "--tick-data",
            str(tick),
            "--ad-data",
            str(ad),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "trades.csv").exists()
    metrics = json.loads((out / "session_metrics.json").read_text())
    assert metrics["run_status"] == "completed"
    # The breadth path is the load-bearing assertion: a non-zero
    # submitted_signals proves the TICK + AD columns reached the
    # strategy through the canonical CLI -> loader -> runner pipeline.
    assert metrics["submitted_signals"] >= 1, metrics

    trades_df = pd.read_csv(out / "trades.csv")
    # Schema parity with the OHLCV-only path so report.py keeps working.
    for col in ("timestamp", "signal_ts", "fill_ts", "side", "price", "order_type"):
        assert col in trades_df.columns, trades_df.columns.tolist()


def test_cli_backtest_rejects_only_tick_data(tmp_path):
    es = tmp_path / "es.parquet"
    idx = _write_es_parquet(es)
    tick = tmp_path / "tick.parquet"
    _write_tick_breadth(tick, idx)
    cfg = _write_breadth_config(tmp_path)
    out = tmp_path / "run_partial"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backtest",
            "--strategy",
            "tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy",
            "--config",
            str(cfg),
            "--data",
            str(es),
            "--tick-data",
            str(tick),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "tick-data" in result.output and "ad-data" in result.output


def test_cli_backtest_rejects_only_ad_data(tmp_path):
    es = tmp_path / "es.parquet"
    idx = _write_es_parquet(es)
    ad = tmp_path / "ad.parquet"
    _write_ad_breadth(ad, idx)
    cfg = _write_breadth_config(tmp_path)
    out = tmp_path / "run_partial"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backtest",
            "--strategy",
            "tests.fixtures.breadth_test_strategy:BreadthAwareTestStrategy",
            "--config",
            str(cfg),
            "--data",
            str(es),
            "--ad-data",
            str(ad),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "tick-data" in result.output and "ad-data" in result.output
