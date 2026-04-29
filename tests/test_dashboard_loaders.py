# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for dashboard.loaders - schema and happy-path coverage."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from dashboard.loaders import (
    describe_run,
    discover_runs,
    filter_by_time_range,
    load_all_runs,
    load_per_trade_metrics,
    load_session_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------


def _write_per_trade_csv(run_dir: Path, rows: list[dict]) -> Path:
    """Write a minimal per_trade_metrics.csv under run_dir/report/."""
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "per_trade_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _write_session_json(run_dir: Path, data: dict) -> Path:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "session_metrics.json"
    path.write_text(json.dumps(data))
    return path


_SAMPLE_ROW = {
    "entry_ts": "2026-04-02 14:46:00+00:00",
    "exit_ts": "2026-04-02 14:47:00+00:00",
    "side": "long",
    "entry_price": 6615.0,
    "exit_price": 6616.0,
    "quantity": 1.0,
    "pnl_points": 1.0,
    "pnl_usd": 50.0,
    "hold_seconds": 60.0,
    "mae_points": 3.75,
    "mfe_points": 9.25,
    "exit_reason": "target",
    "entry_signal_ts": "2026-04-02 14:46:00+00:00",
    "fill_latency_seconds": 0.0,
}


# ---------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------


def test_discover_runs_returns_empty_for_missing_dir(tmp_path):
    assert discover_runs(tmp_path / "nonexistent") == []


def test_discover_runs_skips_dirs_without_report_files(tmp_path):
    (tmp_path / "empty_run").mkdir()
    assert discover_runs(tmp_path) == []


def test_discover_runs_finds_run_with_per_trade_csv(tmp_path):
    run = tmp_path / "my-run"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    result = discover_runs(tmp_path)
    assert result == ["my-run"]


def test_discover_runs_finds_run_with_session_json_only(tmp_path):
    run = tmp_path / "run-json-only"
    _write_session_json(run, {"starting_balance_usd": 100000.0})
    result = discover_runs(tmp_path)
    assert result == ["run-json-only"]


def test_discover_runs_returns_multiple_sorted_newest_first(tmp_path):
    import time

    run_a = tmp_path / "run-a"
    _write_per_trade_csv(run_a, [_SAMPLE_ROW])
    time.sleep(0.01)
    run_b = tmp_path / "run-b"
    _write_per_trade_csv(run_b, [_SAMPLE_ROW])

    result = discover_runs(tmp_path)
    assert result[0] == "run-b"
    assert result[1] == "run-a"


# ---------------------------------------------------------------------------
# load_per_trade_metrics
# ---------------------------------------------------------------------------


def test_load_per_trade_metrics_returns_none_when_absent(tmp_path):
    run = tmp_path / "empty-run"
    run.mkdir()
    assert load_per_trade_metrics(run) is None


def test_load_per_trade_metrics_parses_timestamps_as_utc(tmp_path):
    run = tmp_path / "run1"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    df = load_per_trade_metrics(run)
    assert df is not None
    assert pd.api.types.is_datetime64_any_dtype(df["entry_ts"])
    assert str(df["entry_ts"].dt.tz) == "UTC"


def test_load_per_trade_metrics_required_columns_present(tmp_path):
    run = tmp_path / "run2"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    df = load_per_trade_metrics(run)
    for col in ("entry_ts", "exit_ts", "side", "pnl_usd", "pnl_points", "hold_seconds"):
        assert col in df.columns, f"Missing column: {col}"


def test_load_per_trade_metrics_numeric_columns_are_float(tmp_path):
    run = tmp_path / "run3"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    df = load_per_trade_metrics(run)
    assert pd.api.types.is_float_dtype(df["pnl_usd"])
    assert pd.api.types.is_float_dtype(df["hold_seconds"])


def test_load_per_trade_metrics_multi_row(tmp_path):
    run = tmp_path / "run4"
    rows = [_SAMPLE_ROW, {**_SAMPLE_ROW, "pnl_usd": -25.0, "side": "short"}]
    _write_per_trade_csv(run, rows)
    df = load_per_trade_metrics(run)
    assert len(df) == 2


def test_load_per_trade_metrics_mae_mfe_optional_but_surfaced(tmp_path):
    """mae_points / mfe_points are optional; when present they must be numeric."""
    run = tmp_path / "run5"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    df = load_per_trade_metrics(run)
    assert pd.api.types.is_float_dtype(df["mae_points"])
    assert pd.api.types.is_float_dtype(df["mfe_points"])


def test_load_per_trade_metrics_no_mae_mfe_columns(tmp_path):
    """CSV without mae_points/mfe_points must still load cleanly."""
    run = tmp_path / "run6"
    row_no_mae = {k: v for k, v in _SAMPLE_ROW.items() if k not in ("mae_points", "mfe_points")}
    _write_per_trade_csv(run, [row_no_mae])
    df = load_per_trade_metrics(run)
    assert df is not None
    assert len(df) == 1


# ---------------------------------------------------------------------------
# load_session_metrics
# ---------------------------------------------------------------------------


def test_load_session_metrics_returns_empty_dict_when_absent(tmp_path):
    run = tmp_path / "no-meta"
    run.mkdir()
    assert load_session_metrics(run) == {}


def test_load_session_metrics_parses_json(tmp_path):
    run = tmp_path / "run-meta"
    payload = {"starting_balance_usd": 100_000.0, "n_trades_total": 9, "win_rate": 0.333}
    _write_session_json(run, payload)
    result = load_session_metrics(run)
    assert result["starting_balance_usd"] == 100_000.0
    assert result["n_trades_total"] == 9


# ---------------------------------------------------------------------------
# filter_by_time_range
# ---------------------------------------------------------------------------


def _make_df() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            "2026-04-01 14:00:00+00:00",
            "2026-04-02 14:00:00+00:00",
            "2026-04-03 14:00:00+00:00",
        ]
    )
    return pd.DataFrame({"entry_ts": ts, "pnl_usd": [50.0, -25.0, 50.0]})


def test_filter_no_bounds_returns_all():
    df = _make_df()
    result = filter_by_time_range(df, None, None)
    assert len(result) == 3


def test_filter_start_only():
    df = _make_df()
    start = pd.Timestamp("2026-04-02 00:00:00", tz="UTC")
    result = filter_by_time_range(df, start, None)
    assert len(result) == 2
    assert result["entry_ts"].min() >= start


def test_filter_end_only():
    df = _make_df()
    end = pd.Timestamp("2026-04-02 23:59:59", tz="UTC")
    result = filter_by_time_range(df, None, end)
    assert len(result) == 2


def test_filter_start_and_end():
    df = _make_df()
    start = pd.Timestamp("2026-04-02 00:00:00", tz="UTC")
    end = pd.Timestamp("2026-04-02 23:59:59", tz="UTC")
    result = filter_by_time_range(df, start, end)
    assert len(result) == 1


def test_filter_empty_df_returns_empty():
    result = filter_by_time_range(pd.DataFrame(), None, None)
    assert result.empty


def test_filter_none_df_returns_none():
    result = filter_by_time_range(None, None, None)
    assert result is None


# ---------------------------------------------------------------------------
# describe_run
# ---------------------------------------------------------------------------


def test_describe_run_returns_name_when_no_csv(tmp_path):
    run = tmp_path / "my-run"
    run.mkdir()
    result = describe_run(run)
    assert result == "my-run"


def test_describe_run_includes_trade_count_and_dates(tmp_path):
    run = tmp_path / "my-run"
    rows = [
        {**_SAMPLE_ROW, "entry_ts": "2026-03-26 14:00:00+00:00"},
        {**_SAMPLE_ROW, "entry_ts": "2026-04-24 14:00:00+00:00"},
    ]
    _write_per_trade_csv(run, rows)
    result = describe_run(run)
    assert "my-run" in result
    assert "2 trades" in result
    assert "2026-03-26" in result
    assert "2026-04-24" in result


def test_describe_run_single_trade_shows_same_start_end(tmp_path):
    run = tmp_path / "solo"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    result = describe_run(run)
    assert "1 trades" in result
    # Both start and end should be the same date.
    assert result.count("2026-04-02") == 2


# ---------------------------------------------------------------------------
# load_all_runs
# ---------------------------------------------------------------------------


def test_load_all_runs_returns_none_for_missing_dir(tmp_path):
    result = load_all_runs(tmp_path / "nonexistent")
    assert result is None


def test_load_all_runs_returns_none_when_no_runs(tmp_path):
    # Directory exists but has no qualifying run subdirectories.
    result = load_all_runs(tmp_path)
    assert result is None


def test_load_all_runs_single_run_matches_direct_load(tmp_path):
    run = tmp_path / "run-a"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    combined = load_all_runs(tmp_path)
    direct = load_per_trade_metrics(run)
    assert combined is not None
    assert len(combined) == len(direct)


def test_load_all_runs_concatenates_multiple_runs(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_per_trade_csv(run_a, [_SAMPLE_ROW])
    _write_per_trade_csv(run_b, [_SAMPLE_ROW, {**_SAMPLE_ROW, "pnl_usd": -25.0}])
    combined = load_all_runs(tmp_path)
    assert combined is not None
    assert len(combined) == 3


def test_load_all_runs_adds_run_name_column(tmp_path):
    run = tmp_path / "run-x"
    _write_per_trade_csv(run, [_SAMPLE_ROW])
    combined = load_all_runs(tmp_path)
    assert combined is not None
    assert "run_name" in combined.columns
    assert combined["run_name"].iloc[0] == "run-x"


def test_load_all_runs_sorted_by_entry_ts(tmp_path):
    run = tmp_path / "run-mixed"
    rows = [
        {**_SAMPLE_ROW, "entry_ts": "2026-04-03 14:00:00+00:00"},
        {**_SAMPLE_ROW, "entry_ts": "2026-04-01 14:00:00+00:00"},
    ]
    _write_per_trade_csv(run, rows)
    combined = load_all_runs(tmp_path)
    assert combined is not None
    ts_values = combined["entry_ts"].tolist()
    assert ts_values == sorted(ts_values)


def test_load_all_runs_skips_runs_without_trade_csv(tmp_path):
    # A run with only session_metrics.json but no per_trade_metrics.csv.
    run_meta_only = tmp_path / "meta-only"
    _write_session_json(run_meta_only, {"starting_balance_usd": 100_000.0})
    # A run with actual trades.
    run_trades = tmp_path / "with-trades"
    _write_per_trade_csv(run_trades, [_SAMPLE_ROW])
    combined = load_all_runs(tmp_path)
    assert combined is not None
    assert len(combined) == 1
    assert combined["run_name"].iloc[0] == "with-trades"
