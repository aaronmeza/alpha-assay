# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""End-to-end CliRunner test for `alpha_assay report`.

Drives the CLI exactly as a user would: write a tiny synthetic
parquet, invoke `alpha_assay backtest` to produce trades.csv +
session_metrics.json, then invoke `alpha_assay report` against that
output dir. Asserts every artifact exists and has the expected schema
+ key fields.

Gated on quantstats being importable: skipped via `pytest.importorskip`
if the `[report]` extra is not installed in the test venv.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from alpha_assay.cli import main as cli_main

# The report command imports quantstats at runtime. If the dev venv
# does not have the [report] extra installed, skip the entire module
# rather than fail.
pytest.importorskip("quantstats")


def _write_fixture_parquet(tmp_path: Path) -> Path:
    """Generate an oscillating intraday OHLCV parquet that triggers
    the SMA crossover at least a handful of times.

    A flat-line sample produces zero crossovers and therefore zero
    trades; the existing backtest test hides that by not asserting
    trade count. The report pipeline cannot validate against zero
    trades, so we synthesize a sine-driven path here that the SMA
    crossover (fast=3, slow=10) reliably trips.
    """
    # 09:30 ET start = 14:30 UTC. Use a tz-aware UTC index, then
    # convert at write time so load_parquet's tz handling matches
    # the rest of the pipeline.
    n = 360
    idx = pd.date_range("2026-04-27 14:30:00", periods=n, freq="1min", tz="UTC")
    base = 5000.0
    # Period ~25 minutes => alternating regimes that cross the 10-bar SMA.
    closes = base + 5.0 * np.sin(np.arange(n) * 2 * math.pi / 25.0)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    # Tiny intrabar range, but always >= max(open, close) for high
    # and <= min(open, close) for low so the OHLC invariants hold.
    spread = 0.5
    high = np.maximum(opens, closes) + spread
    low = np.minimum(opens, closes) - spread
    volume = np.full(n, 100, dtype=int)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": volume,
        }
    )
    out = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pandas(df), out)
    return out


def _write_config(tmp_path: Path) -> Path:
    # SMA's default exit params are stop=1.0, target=2.0. The default
    # v0.1 risk caps require target_pts >= 2.5, so the example fails
    # validation under those caps. Loosen min_target_pts to 2.0 here
    # so the SMA fixture flows trades through the engine without
    # changing the example strategy. Ratio still needs to be >= 2:1.
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
        "  min_target_pts: 2.0\n"
        "  min_target_to_stop_ratio: 2.0\n"
        "session:\n"
        "  minutes_after_open: 30\n"
        "  minutes_before_close: 30\n"
        "execution:\n"
        "  mode: backtest\n"
        "  instrument: MESM6\n"
    )
    return cfg


def _run_backtest(tmp_path: Path) -> Path:
    """Run a backtest via CliRunner and return the output dir."""
    data = _write_fixture_parquet(tmp_path)
    config = _write_config(tmp_path)
    out = tmp_path / "bt_out"

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
    return out


def test_cli_report_writes_all_artifacts(tmp_path):
    bt_out = _run_backtest(tmp_path)
    report_out = tmp_path / "report_out"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "report",
            "--in",
            str(bt_out),
            "--out",
            str(report_out),
            "--format",
            "md",
            "--instrument-multiplier",
            "5.0",
        ],
    )
    assert result.exit_code == 0, result.output
    # Markdown report and the per-trade CSV are always emitted.
    assert (report_out / "per_trade_metrics.csv").exists()
    assert (report_out / "session_metrics.json").exists()
    assert (report_out / "report.md").exists()
    # html-only artifact: skipped because we asked for md only.
    assert not (report_out / "report.html").exists()


def test_cli_report_per_trade_schema(tmp_path):
    bt_out = _run_backtest(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["report", "--in", str(bt_out), "--format", "md", "--instrument-multiplier", "5.0"],
    )
    assert result.exit_code == 0, result.output

    pt = pd.read_csv(bt_out / "per_trade_metrics.csv")
    expected_cols = {
        "entry_ts",
        "exit_ts",
        "side",
        "entry_price",
        "exit_price",
        "quantity",
        "pnl_points",
        "pnl_usd",
        "hold_seconds",
        "mae_points",
        "mfe_points",
        "exit_reason",
    }
    assert expected_cols.issubset(set(pt.columns))
    # Every trade must have a non-null side + valid exit_reason.
    assert pt["side"].isin({"long", "short"}).all()
    assert pt["exit_reason"].isin({"target", "stop", "flip", "unknown"}).all()


def test_cli_report_session_metrics_enriched_additively(tmp_path):
    bt_out = _run_backtest(tmp_path)
    original = json.loads((bt_out / "session_metrics.json").read_text())

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["report", "--in", str(bt_out), "--format", "md", "--instrument-multiplier", "5.0"],
    )
    assert result.exit_code == 0, result.output

    enriched = json.loads((bt_out / "session_metrics.json").read_text())
    # Every original key MUST survive (additive contract).
    for k in original:
        assert k in enriched, f"original key {k!r} dropped from enriched session_metrics"

    # New required keys.
    for k in (
        "instrument_multiplier",
        "n_trades_total",
        "win_rate",
        "profit_factor",
        "per_session_pnl_usd",
        "per_session_trade_count",
        "per_session_avg_fill_time_seconds",
    ):
        assert k in enriched, f"enriched session_metrics missing {k!r}"

    assert enriched["instrument_multiplier"] == pytest.approx(5.0)
    assert isinstance(enriched["per_session_pnl_usd"], dict)
    assert isinstance(enriched["per_session_trade_count"], dict)
    # Trade count must be a non-negative int.
    assert isinstance(enriched["n_trades_total"], int)
    assert enriched["n_trades_total"] >= 0


def test_cli_report_md_contains_section_headers(tmp_path):
    bt_out = _run_backtest(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["report", "--in", str(bt_out), "--format", "md", "--instrument-multiplier", "5.0"],
    )
    assert result.exit_code == 0, result.output

    md = (bt_out / "report.md").read_text()
    for header in (
        "# alpha_assay backtest report",
        "## Run summary",
        "## Standard quant metrics",
        "## Trade-level metrics",
        "## Per-session breakdown (live-vs-backtest parity check)",
    ):
        assert header in md, f"report.md missing section: {header!r}"


def test_cli_report_default_out_is_input_dir(tmp_path):
    # When --out is omitted, artifacts land in --in.
    bt_out = _run_backtest(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["report", "--in", str(bt_out), "--format", "md", "--instrument-multiplier", "5.0"],
    )
    assert result.exit_code == 0, result.output
    assert (bt_out / "per_trade_metrics.csv").exists()
    assert (bt_out / "report.md").exists()


def test_cli_report_missing_input_dir_errors():
    runner = CliRunner()
    result = runner.invoke(cli_main, ["report", "--in", "/no/such/path"])
    assert result.exit_code != 0


def test_cli_report_per_session_avg_fill_time_is_float_when_signal_ts_present(tmp_path):
    """The full backtest -> report pipeline must surface
    per_session_avg_fill_time_seconds as a float (not null) once the
    runner emits signal_ts on every trade. Targets spec Section 9
    live-vs-backtest parity check exit criterion.

    Fill latency is the entry-fill latency: (entry_ts - entry_signal_ts).
    In the current Nautilus simulator entries fill same-bar at close,
    so the in-engine entry latency is 0s. The metric is still meaningful
    on live paper (broker ack time); the parity check compares the two.
    """
    bt_out = _run_backtest(tmp_path)

    # Sanity: trades.csv must include the new signal_ts / fill_ts columns.
    trades = pd.read_csv(bt_out / "trades.csv")
    if not trades.empty:
        assert "signal_ts" in trades.columns
        assert "fill_ts" in trades.columns

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["report", "--in", str(bt_out), "--format", "md", "--instrument-multiplier", "5.0"],
    )
    assert result.exit_code == 0, result.output

    # per_trade_metrics.csv must carry fill_latency_seconds.
    pt = pd.read_csv(bt_out / "per_trade_metrics.csv")
    if not pt.empty:
        assert "fill_latency_seconds" in pt.columns
        latencies = pt["fill_latency_seconds"].dropna()
        # Every trade should now have a numeric latency (>= 0).
        assert not latencies.empty, "expected at least one populated fill_latency_seconds"
        assert (latencies >= 0).all()

    enriched = json.loads((bt_out / "session_metrics.json").read_text())
    fills = enriched["per_session_avg_fill_time_seconds"]
    assert isinstance(fills, dict)
    if pt.empty:
        # Edge case: zero trades -> empty dict, accept and bail.
        assert fills == {}
        return
    # At least one session must have a non-null float now that the
    # runner records signal_ts. (Float, not None - the previous code
    # path emitted null per session.)
    populated = [v for v in fills.values() if isinstance(v, (int, float))]
    assert populated, f"expected at least one float fill-time, got {fills}"
    for v in populated:
        assert v >= 0


def test_cli_report_handles_empty_trades_gracefully(tmp_path):
    # Build a directory with an empty trades.csv but valid session_metrics.
    bt_out = tmp_path / "empty_run"
    bt_out.mkdir()
    pd.DataFrame(columns=["timestamp", "side", "price", "quantity", "order_type"]).to_csv(
        bt_out / "trades.csv", index=False
    )
    (bt_out / "session_metrics.json").write_text(json.dumps({"run_status": "completed", "submitted_signals": 0}))

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["report", "--in", str(bt_out), "--format", "md"],
    )
    assert result.exit_code == 0, result.output
    enriched = json.loads((bt_out / "session_metrics.json").read_text())
    assert enriched["n_trades_total"] == 0
    assert enriched["per_session_pnl_usd"] == {}
