"""CliRunner-driven tests against the backtest CLI. Uses the synthetic CSV + SMA crossover to stay deterministic."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from alpha_assay.cli import main as cli_main

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "sample_2d.csv"


def _write_fixture_parquet(tmp_path: Path) -> Path:
    df = pd.read_csv(FIXTURE_CSV)
    df = df.rename(
        columns={
            "ES_open": "open",
            "ES_high": "high",
            "ES_low": "low",
            "ES_close": "close",
            "ES_volume": "volume",
        }
    )
    keep = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    # The synthetic fixture occasionally violates OHLC invariants
    # after float generation. load_parquet validates strictly, so clamp
    # defensively here to match bar_adapter's live-data policy.
    oc_max = keep[["open", "close"]].max(axis=1)
    oc_min = keep[["open", "close"]].min(axis=1)
    keep["high"] = keep[["high", "open", "close"]].max(axis=1)
    keep["low"] = keep[["low", "open", "close"]].min(axis=1)
    assert (keep["high"] >= oc_max).all()
    assert (keep["low"] <= oc_min).all()
    out = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pandas(keep), out)
    return out


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
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


def test_cli_backtest_writes_trades_and_metrics(tmp_path):
    data = _write_fixture_parquet(tmp_path)
    config = _write_config(tmp_path)
    out = tmp_path / "run_out"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backtest",
            "--strategy",
            "examples.sma_crossover:SMACrossoverStrategy",
            "--config",
            str(config),
            "--data",
            str(data),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "trades.csv").exists()
    assert (out / "session_metrics.json").exists()

    metrics = json.loads((out / "session_metrics.json").read_text())
    assert metrics["run_status"] == "completed"
    assert "submitted_signals" in metrics


def test_cli_backtest_rejects_missing_args(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli_main, ["backtest"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()
