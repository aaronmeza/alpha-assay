# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for dashboard.loaders - paper-trial trade log shape."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_assay.exec.trade_log import TradeLog, TradeRecord
from dashboard.loaders import (
    empty_trades_df,
    filter_by_date_range,
    load_trades,
    required_columns,
    trade_log_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    *,
    ts: str,
    signal: str = "long_entry",
    pnl: float = 50.0,
    balance: float = 100_050.0,
) -> TradeRecord:
    return TradeRecord(
        timestamp=pd.Timestamp(ts, tz="UTC"),
        signal_type=signal,
        entry_price=6615.0,
        stop=6613.0,
        target=6619.0,
        mock_fill_price=6615.25,
        mock_pnl_dollars=pnl,
        account_balance_after=balance,
    )


def _write_trades(out_dir: Path, records: list[TradeRecord]) -> Path:
    log = TradeLog(out_dir=out_dir)
    for r in records:
        log.write(r)
    log.flush()
    return out_dir / "trades.parquet"


# ---------------------------------------------------------------------------
# load_trades
# ---------------------------------------------------------------------------


def test_load_trades_returns_empty_df_when_file_missing(tmp_path):
    df = load_trades(tmp_path / "nonexistent")
    assert df.empty
    for col in required_columns():
        assert col in df.columns


def test_load_trades_returns_empty_df_when_dir_empty(tmp_path):
    df = load_trades(tmp_path)
    assert df.empty


def test_load_trades_reads_single_record(tmp_path):
    _write_trades(tmp_path, [_record(ts="2026-04-02 14:46:00+00:00")])
    df = load_trades(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["signal_type"] == "long_entry"


def test_load_trades_normalises_timestamp_to_utc(tmp_path):
    _write_trades(tmp_path, [_record(ts="2026-04-02 14:46:00+00:00")])
    df = load_trades(tmp_path)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_load_trades_numeric_columns_are_float(tmp_path):
    _write_trades(tmp_path, [_record(ts="2026-04-02 14:46:00+00:00")])
    df = load_trades(tmp_path)
    for col in (
        "entry_price",
        "stop",
        "target",
        "mock_fill_price",
        "mock_pnl_dollars",
        "account_balance_after",
    ):
        assert pd.api.types.is_float_dtype(df[col]), col


def test_load_trades_sorted_by_timestamp_ascending(tmp_path):
    _write_trades(
        tmp_path,
        [
            _record(ts="2026-04-03 14:00:00+00:00"),
            _record(ts="2026-04-01 14:00:00+00:00"),
            _record(ts="2026-04-02 14:00:00+00:00"),
        ],
    )
    df = load_trades(tmp_path)
    ts_values = df["timestamp"].tolist()
    assert ts_values == sorted(ts_values)


def test_load_trades_multi_record(tmp_path):
    _write_trades(
        tmp_path,
        [
            _record(ts="2026-04-02 14:46:00+00:00", pnl=50.0),
            _record(ts="2026-04-03 14:46:00+00:00", pnl=-25.0, signal="short_entry"),
        ],
    )
    df = load_trades(tmp_path)
    assert len(df) == 2
    assert set(df["signal_type"].unique()) == {"long_entry", "short_entry"}


# ---------------------------------------------------------------------------
# empty_trades_df
# ---------------------------------------------------------------------------


def test_empty_trades_df_has_full_schema():
    df = empty_trades_df()
    for col in required_columns():
        assert col in df.columns
    assert len(df) == 0


# ---------------------------------------------------------------------------
# filter_by_date_range
# ---------------------------------------------------------------------------


def _df_three_days() -> pd.DataFrame:
    rows = [
        {
            "timestamp": pd.Timestamp("2026-04-01 14:00:00", tz="UTC"),
            "signal_type": "long_entry",
            "entry_price": 6615.0,
            "stop": 6613.0,
            "target": 6619.0,
            "mock_fill_price": 6615.25,
            "mock_pnl_dollars": 50.0,
            "account_balance_after": 100_050.0,
        },
        {
            "timestamp": pd.Timestamp("2026-04-02 14:00:00", tz="UTC"),
            "signal_type": "short_entry",
            "entry_price": 6615.0,
            "stop": 6617.0,
            "target": 6611.0,
            "mock_fill_price": 6614.75,
            "mock_pnl_dollars": -25.0,
            "account_balance_after": 100_025.0,
        },
        {
            "timestamp": pd.Timestamp("2026-04-03 14:00:00", tz="UTC"),
            "signal_type": "long_entry",
            "entry_price": 6615.0,
            "stop": 6613.0,
            "target": 6619.0,
            "mock_fill_price": 6615.25,
            "mock_pnl_dollars": 75.0,
            "account_balance_after": 100_100.0,
        },
    ]
    return pd.DataFrame(rows)


def test_filter_no_bounds_returns_all():
    df = _df_three_days()
    result = filter_by_date_range(df, None, None)
    assert len(result) == 3


def test_filter_start_only():
    df = _df_three_days()
    start = pd.Timestamp("2026-04-02 00:00:00", tz="UTC")
    result = filter_by_date_range(df, start, None)
    assert len(result) == 2


def test_filter_end_only():
    df = _df_three_days()
    end = pd.Timestamp("2026-04-02 23:59:59", tz="UTC")
    result = filter_by_date_range(df, None, end)
    assert len(result) == 2


def test_filter_start_and_end_inclusive():
    df = _df_three_days()
    start = pd.Timestamp("2026-04-02 00:00:00", tz="UTC")
    end = pd.Timestamp("2026-04-02 23:59:59", tz="UTC")
    result = filter_by_date_range(df, start, end)
    assert len(result) == 1


def test_filter_naive_bounds_treated_as_utc():
    df = _df_three_days()
    start = pd.Timestamp("2026-04-02 00:00:00")
    end = pd.Timestamp("2026-04-02 23:59:59")
    result = filter_by_date_range(df, start, end)
    assert len(result) == 1


def test_filter_empty_df_returns_empty():
    result = filter_by_date_range(pd.DataFrame(), None, None)
    assert result is None or result.empty


# ---------------------------------------------------------------------------
# trade_log_summary
# ---------------------------------------------------------------------------


def test_trade_log_summary_when_missing(tmp_path):
    s = trade_log_summary(tmp_path)
    assert s["exists"] is False
    assert s["n_rows"] == 0
    assert s["modified_at"] is None


def test_trade_log_summary_when_present(tmp_path):
    _write_trades(tmp_path, [_record(ts="2026-04-02 14:46:00+00:00")])
    s = trade_log_summary(tmp_path)
    assert s["exists"] is True
    assert s["n_rows"] == 1
    assert s["size_bytes"] > 0
    assert s["modified_at"] is not None


# ---------------------------------------------------------------------------
# Round-trip: TradeLog -> load_trades preserves shape exactly
# ---------------------------------------------------------------------------


def test_round_trip_preserves_field_values(tmp_path):
    rec = _record(ts="2026-04-02 14:46:00+00:00", pnl=42.5, balance=100_042.5)
    _write_trades(tmp_path, [rec])
    df = load_trades(tmp_path)
    row = df.iloc[0]
    assert row["signal_type"] == rec.signal_type
    assert row["entry_price"] == pytest.approx(rec.entry_price)
    assert row["mock_pnl_dollars"] == pytest.approx(rec.mock_pnl_dollars)
    assert row["account_balance_after"] == pytest.approx(rec.account_balance_after)
