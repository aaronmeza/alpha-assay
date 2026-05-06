# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for dashboard.metrics - paper-trial aggregate computations."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.metrics import (
    compute_aggregate_metrics,
    compute_equity_curve,
    compute_per_day_summary,
    compute_signal_histogram,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    ts: str,
    signal: str = "long_entry",
    pnl: float = 50.0,
    balance: float = 100_050.0,
) -> dict:
    return {
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "signal_type": signal,
        "entry_price": 6615.0,
        "stop": 6613.0,
        "target": 6619.0,
        "mock_fill_price": 6615.25,
        "mock_pnl_dollars": pnl,
        "account_balance_after": balance,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_empty():
    m = compute_aggregate_metrics(pd.DataFrame())
    assert m["n_trades"] == 0
    assert m["win_rate"] is None
    assert m["profit_factor"] is None
    assert m["sharpe"] is None
    assert m["current_balance"] is None


def test_aggregate_none():
    m = compute_aggregate_metrics(None)
    assert m["n_trades"] == 0


def test_aggregate_single_win():
    df = _df([_row(ts="2026-04-02 14:46:00+00:00", pnl=50.0, balance=100_050.0)])
    m = compute_aggregate_metrics(df)
    assert m["n_trades"] == 1
    assert m["total_pnl_usd"] == pytest.approx(50.0)
    assert m["win_rate"] == pytest.approx(1.0)
    assert m["profit_factor"] is None  # no losses
    assert m["current_balance"] == pytest.approx(100_050.0)


def test_aggregate_single_loss():
    df = _df([_row(ts="2026-04-02 14:46:00+00:00", pnl=-25.0, balance=99_975.0)])
    m = compute_aggregate_metrics(df)
    assert m["total_pnl_usd"] == pytest.approx(-25.0)
    assert m["win_rate"] == pytest.approx(0.0)
    assert m["profit_factor"] == pytest.approx(0.0)
    assert m["current_balance"] == pytest.approx(99_975.0)


def test_aggregate_mixed():
    df = _df(
        [
            _row(ts="2026-04-01 14:00:00+00:00", pnl=50.0, balance=100_050.0),
            _row(ts="2026-04-02 14:00:00+00:00", pnl=-25.0, balance=100_025.0),
        ]
    )
    m = compute_aggregate_metrics(df)
    assert m["n_trades"] == 2
    assert m["total_pnl_usd"] == pytest.approx(25.0)
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx(2.0)
    assert m["avg_pnl_per_trade"] == pytest.approx(12.5)


def test_aggregate_max_drawdown_zero_for_winners():
    df = _df(
        [
            _row(ts=f"2026-04-0{i} 14:00:00+00:00", pnl=50.0, balance=100_000.0 + 50.0 * i)
            for i in range(1, 5)
        ]
    )
    m = compute_aggregate_metrics(df)
    assert m["max_drawdown_usd"] == pytest.approx(0.0)


def test_aggregate_max_drawdown_negative_after_loss():
    df = _df(
        [
            _row(ts="2026-04-01 14:00:00+00:00", pnl=100.0),
            _row(ts="2026-04-02 14:00:00+00:00", pnl=-60.0),
            _row(ts="2026-04-03 14:00:00+00:00", pnl=10.0),
        ]
    )
    m = compute_aggregate_metrics(df)
    assert m["max_drawdown_usd"] == pytest.approx(-60.0)


def test_aggregate_sharpe_none_for_single_trade():
    df = _df([_row(ts="2026-04-01 14:00:00+00:00")])
    m = compute_aggregate_metrics(df)
    assert m["sharpe"] is None


def test_aggregate_sharpe_float_for_multiple_trades():
    df = _df(
        [
            _row(ts="2026-04-01 14:00:00+00:00", pnl=50.0),
            _row(ts="2026-04-02 14:00:00+00:00", pnl=-25.0),
            _row(ts="2026-04-03 14:00:00+00:00", pnl=75.0),
        ]
    )
    m = compute_aggregate_metrics(df)
    assert isinstance(m["sharpe"], float)


def test_aggregate_current_balance_uses_latest_timestamp():
    df = _df(
        [
            _row(ts="2026-04-03 14:00:00+00:00", balance=100_500.0),
            _row(ts="2026-04-01 14:00:00+00:00", balance=100_050.0),
            _row(ts="2026-04-02 14:00:00+00:00", balance=100_200.0),
        ]
    )
    m = compute_aggregate_metrics(df)
    assert m["current_balance"] == pytest.approx(100_500.0)


# ---------------------------------------------------------------------------
# compute_equity_curve
# ---------------------------------------------------------------------------


def test_equity_curve_empty_returns_empty():
    ec = compute_equity_curve(pd.DataFrame())
    assert ec.empty
    assert list(ec.columns) == ["timestamp", "account_balance_after"]


def test_equity_curve_uses_recorded_balance():
    df = _df(
        [
            _row(ts="2026-04-01 14:00:00+00:00", pnl=50.0, balance=100_050.0),
            _row(ts="2026-04-02 14:00:00+00:00", pnl=-25.0, balance=100_025.0),
            _row(ts="2026-04-03 14:00:00+00:00", pnl=75.0, balance=100_100.0),
        ]
    )
    ec = compute_equity_curve(df)
    assert len(ec) == 3
    assert ec["account_balance_after"].iloc[0] == pytest.approx(100_050.0)
    assert ec["account_balance_after"].iloc[2] == pytest.approx(100_100.0)


def test_equity_curve_sorted_ascending():
    df = _df(
        [
            _row(ts="2026-04-03 14:00:00+00:00", balance=100_100.0),
            _row(ts="2026-04-01 14:00:00+00:00", balance=100_050.0),
        ]
    )
    ec = compute_equity_curve(df)
    assert ec["timestamp"].iloc[0] < ec["timestamp"].iloc[1]


# ---------------------------------------------------------------------------
# compute_per_day_summary
# ---------------------------------------------------------------------------


def test_per_day_summary_empty():
    assert compute_per_day_summary(pd.DataFrame()) == []


def test_per_day_summary_groups_by_chicago_date():
    df = _df(
        [
            # UTC 04:59 on Apr 2 = Apr 1 23:59 Chicago (CDT = UTC-5)
            _row(ts="2026-04-02 04:59:00+00:00", pnl=50.0),
            # UTC 18:00 on Apr 2 = Apr 2 13:00 Chicago
            _row(ts="2026-04-02 18:00:00+00:00", pnl=-25.0),
        ]
    )
    summaries = compute_per_day_summary(df)
    dates = {s["date"] for s in summaries}
    assert len(dates) == 2


def test_per_day_summary_pnl_total_and_winrate():
    df = _df(
        [
            _row(ts="2026-04-02 14:00:00+00:00", pnl=50.0),
            _row(ts="2026-04-02 15:00:00+00:00", pnl=-25.0),
        ]
    )
    summaries = compute_per_day_summary(df)
    assert len(summaries) == 1
    s = summaries[0]
    assert s["n_trades"] == 2
    assert s["total_pnl_usd"] == pytest.approx(25.0)
    assert s["win_rate"] == pytest.approx(0.5)


def test_per_day_summary_largest_win_loss():
    df = _df(
        [
            _row(ts="2026-04-02 14:00:00+00:00", pnl=100.0),
            _row(ts="2026-04-02 14:30:00+00:00", pnl=50.0),
            _row(ts="2026-04-02 15:00:00+00:00", pnl=-25.0),
            _row(ts="2026-04-02 15:30:00+00:00", pnl=-10.0),
        ]
    )
    summaries = compute_per_day_summary(df)
    s = summaries[0]
    assert s["largest_win_usd"] == pytest.approx(100.0)
    assert s["largest_loss_usd"] == pytest.approx(-25.0)


def test_per_day_summary_ending_balance():
    df = _df(
        [
            _row(ts="2026-04-02 14:00:00+00:00", balance=100_100.0),
            _row(ts="2026-04-02 15:00:00+00:00", balance=100_050.0),
            _row(ts="2026-04-02 16:00:00+00:00", balance=100_075.0),
        ]
    )
    summaries = compute_per_day_summary(df)
    assert summaries[0]["ending_balance"] == pytest.approx(100_075.0)


def test_per_day_summary_sorted_ascending():
    df = _df(
        [
            _row(ts="2026-04-05 14:00:00+00:00", pnl=50.0),
            _row(ts="2026-04-03 14:00:00+00:00", pnl=-25.0),
        ]
    )
    summaries = compute_per_day_summary(df)
    assert summaries[0]["date"] < summaries[1]["date"]


# ---------------------------------------------------------------------------
# compute_signal_histogram
# ---------------------------------------------------------------------------


def test_signal_histogram_empty():
    h = compute_signal_histogram(pd.DataFrame())
    assert h.empty
    assert list(h.columns) == ["signal_type", "n_trades", "total_pnl_usd"]


def test_signal_histogram_groups_by_signal_type():
    df = _df(
        [
            _row(ts="2026-04-01 14:00:00+00:00", signal="long_entry", pnl=50.0),
            _row(ts="2026-04-02 14:00:00+00:00", signal="long_entry", pnl=25.0),
            _row(ts="2026-04-03 14:00:00+00:00", signal="short_entry", pnl=-30.0),
        ]
    )
    h = compute_signal_histogram(df)
    assert len(h) == 2
    long_row = h[h["signal_type"] == "long_entry"].iloc[0]
    short_row = h[h["signal_type"] == "short_entry"].iloc[0]
    assert long_row["n_trades"] == 2
    assert long_row["total_pnl_usd"] == pytest.approx(75.0)
    assert short_row["n_trades"] == 1
    assert short_row["total_pnl_usd"] == pytest.approx(-30.0)


def test_signal_histogram_sorted_descending_by_count():
    df = _df(
        [
            _row(ts="2026-04-01 14:00:00+00:00", signal="rare"),
            _row(ts="2026-04-02 14:00:00+00:00", signal="common"),
            _row(ts="2026-04-03 14:00:00+00:00", signal="common"),
            _row(ts="2026-04-04 14:00:00+00:00", signal="common"),
        ]
    )
    h = compute_signal_histogram(df)
    assert h["signal_type"].iloc[0] == "common"
    assert h["n_trades"].iloc[0] == 3
