# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for dashboard.metrics - aggregate computation coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.metrics import (
    compute_aggregate_metrics,
    compute_equity_curve,
    compute_per_day_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal per-trade DataFrame from row dicts."""
    df = pd.DataFrame(rows)
    for col in ("entry_ts", "exit_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


_LONG_WIN = {
    "entry_ts": "2026-04-02 14:46:00+00:00",
    "exit_ts": "2026-04-02 14:47:00+00:00",
    "side": "long",
    "pnl_usd": 50.0,
    "pnl_points": 1.0,
    "hold_seconds": 60.0,
    "exit_reason": "target",
}
_SHORT_LOSS = {
    "entry_ts": "2026-04-03 14:00:00+00:00",
    "exit_ts": "2026-04-03 14:01:00+00:00",
    "side": "short",
    "pnl_usd": -25.0,
    "pnl_points": -0.5,
    "hold_seconds": 60.0,
    "exit_reason": "stop",
}


# ---------------------------------------------------------------------------
# compute_aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_empty_df():
    m = compute_aggregate_metrics(pd.DataFrame())
    assert m["n_trades"] == 0
    assert m["win_rate"] is None
    assert m["profit_factor"] is None
    assert m["sharpe"] is None


def test_aggregate_none_df():
    m = compute_aggregate_metrics(None)
    assert m["n_trades"] == 0


def test_aggregate_single_win():
    df = _make_trades([_LONG_WIN])
    m = compute_aggregate_metrics(df, starting_balance_usd=100_000.0)
    assert m["n_trades"] == 1
    assert m["total_pnl_usd"] == pytest.approx(50.0)
    assert m["win_rate"] == pytest.approx(1.0)
    assert m["profit_factor"] is None  # no losses


def test_aggregate_single_loss():
    df = _make_trades([_SHORT_LOSS])
    m = compute_aggregate_metrics(df, starting_balance_usd=100_000.0)
    assert m["n_trades"] == 1
    assert m["total_pnl_usd"] == pytest.approx(-25.0)
    assert m["win_rate"] == pytest.approx(0.0)
    assert m["profit_factor"] == pytest.approx(0.0)


def test_aggregate_mixed():
    df = _make_trades([_LONG_WIN, _SHORT_LOSS])
    m = compute_aggregate_metrics(df, starting_balance_usd=100_000.0)
    assert m["n_trades"] == 2
    assert m["total_pnl_usd"] == pytest.approx(25.0)
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx(2.0)


def test_aggregate_pnl_pct():
    df = _make_trades([_LONG_WIN])
    m = compute_aggregate_metrics(df, starting_balance_usd=100_000.0)
    # $50 gain on $100k = 0.05%
    assert m["total_pnl_pct"] == pytest.approx(0.05, rel=1e-3)


def test_aggregate_max_drawdown_zero_for_monotone_winners():
    rows = [
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": f"2026-04-0{i} 14:00:00+00:00"}
        for i in range(2, 6)
    ]
    df = _make_trades(rows)
    m = compute_aggregate_metrics(df)
    assert m["max_drawdown_usd"] == pytest.approx(0.0)


def test_aggregate_max_drawdown_negative_after_loss():
    rows = [
        {**_LONG_WIN, "pnl_usd": 100.0, "entry_ts": "2026-04-01 14:00:00+00:00"},
        {**_SHORT_LOSS, "pnl_usd": -60.0, "entry_ts": "2026-04-02 14:00:00+00:00"},
        {**_LONG_WIN, "pnl_usd": 10.0, "entry_ts": "2026-04-03 14:00:00+00:00"},
    ]
    df = _make_trades(rows)
    m = compute_aggregate_metrics(df)
    assert m["max_drawdown_usd"] == pytest.approx(-60.0)


def test_aggregate_sharpe_is_none_for_single_trade():
    df = _make_trades([_LONG_WIN])
    m = compute_aggregate_metrics(df)
    assert m["sharpe"] is None


def test_aggregate_sharpe_is_float_for_multiple_trades():
    rows = [
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": "2026-04-01 14:00:00+00:00"},
        {**_SHORT_LOSS, "pnl_usd": -25.0, "entry_ts": "2026-04-02 14:00:00+00:00"},
        {**_LONG_WIN, "pnl_usd": 75.0, "entry_ts": "2026-04-03 14:00:00+00:00"},
    ]
    df = _make_trades(rows)
    m = compute_aggregate_metrics(df)
    assert isinstance(m["sharpe"], float)


# ---------------------------------------------------------------------------
# compute_equity_curve
# ---------------------------------------------------------------------------


def test_equity_curve_empty_returns_empty():
    ec = compute_equity_curve(pd.DataFrame())
    assert ec.empty
    assert list(ec.columns) == ["entry_ts", "cumulative_pnl_usd"]


def test_equity_curve_cumulative_sum():
    rows = [
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": "2026-04-01 14:00:00+00:00"},
        {**_LONG_WIN, "pnl_usd": -25.0, "entry_ts": "2026-04-02 14:00:00+00:00"},
        {**_LONG_WIN, "pnl_usd": 75.0, "entry_ts": "2026-04-03 14:00:00+00:00"},
    ]
    df = _make_trades(rows)
    ec = compute_equity_curve(df)
    assert len(ec) == 3
    assert ec["cumulative_pnl_usd"].iloc[0] == pytest.approx(50.0)
    assert ec["cumulative_pnl_usd"].iloc[1] == pytest.approx(25.0)
    assert ec["cumulative_pnl_usd"].iloc[2] == pytest.approx(100.0)


def test_equity_curve_sorted_ascending():
    rows = [
        {**_LONG_WIN, "pnl_usd": 10.0, "entry_ts": "2026-04-03 14:00:00+00:00"},
        {**_LONG_WIN, "pnl_usd": 20.0, "entry_ts": "2026-04-01 14:00:00+00:00"},
    ]
    df = _make_trades(rows)
    ec = compute_equity_curve(df)
    assert ec["cumulative_pnl_usd"].iloc[0] == pytest.approx(20.0)
    assert ec["cumulative_pnl_usd"].iloc[1] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# compute_per_day_summary
# ---------------------------------------------------------------------------


def test_per_day_summary_empty():
    result = compute_per_day_summary(pd.DataFrame())
    assert result == []


def test_per_day_summary_groups_by_local_date():
    """Two UTC timestamps that fall on different Chicago calendar dates."""
    rows = [
        # UTC 04:59 on Apr 2 = Apr 1 23:59 Chicago (CDT = UTC-5)
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": "2026-04-02 04:59:00+00:00"},
        # UTC 18:00 on Apr 2 = Apr 2 13:00 Chicago
        {**_SHORT_LOSS, "pnl_usd": -25.0, "entry_ts": "2026-04-02 18:00:00+00:00"},
    ]
    df = _make_trades(rows)
    summaries = compute_per_day_summary(df, starting_balance_usd=100_000.0)
    dates = {s["date"] for s in summaries}
    # The two rows land on different Chicago calendar dates
    assert len(dates) == 2


def test_per_day_summary_pnl_total():
    rows = [
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": "2026-04-02 14:00:00+00:00"},
        {**_SHORT_LOSS, "pnl_usd": -25.0, "entry_ts": "2026-04-02 15:00:00+00:00"},
    ]
    df = _make_trades(rows)
    summaries = compute_per_day_summary(df, starting_balance_usd=100_000.0)
    assert len(summaries) == 1
    s = summaries[0]
    assert s["n_trades"] == 2
    assert s["total_pnl_usd"] == pytest.approx(25.0)
    assert s["win_rate"] == pytest.approx(0.5)


def test_per_day_summary_side_counts():
    rows = [
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": "2026-04-02 14:00:00+00:00"},
        {**_SHORT_LOSS, "pnl_usd": -25.0, "entry_ts": "2026-04-02 15:00:00+00:00"},
    ]
    df = _make_trades(rows)
    summaries = compute_per_day_summary(df, starting_balance_usd=100_000.0)
    s = summaries[0]
    assert s["n_long"] == 1
    assert s["n_short"] == 1


def test_per_day_summary_sorted_ascending():
    rows = [
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": "2026-04-05 14:00:00+00:00"},
        {**_SHORT_LOSS, "pnl_usd": -25.0, "entry_ts": "2026-04-03 14:00:00+00:00"},
    ]
    df = _make_trades(rows)
    summaries = compute_per_day_summary(df, starting_balance_usd=100_000.0)
    assert summaries[0]["date"] < summaries[1]["date"]


def test_per_day_summary_largest_win_and_loss():
    rows = [
        {**_LONG_WIN, "pnl_usd": 100.0, "entry_ts": "2026-04-02 14:00:00+00:00"},
        {**_LONG_WIN, "pnl_usd": 50.0, "entry_ts": "2026-04-02 14:30:00+00:00"},
        {**_SHORT_LOSS, "pnl_usd": -25.0, "entry_ts": "2026-04-02 15:00:00+00:00"},
        {**_SHORT_LOSS, "pnl_usd": -10.0, "entry_ts": "2026-04-02 15:30:00+00:00"},
    ]
    df = _make_trades(rows)
    summaries = compute_per_day_summary(df, starting_balance_usd=100_000.0)
    s = summaries[0]
    assert s["largest_win_usd"] == pytest.approx(100.0)
    assert s["largest_loss_usd"] == pytest.approx(-25.0)
