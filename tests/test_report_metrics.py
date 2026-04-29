# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Unit tests for the metric calculations in `alpha_assay.cli.report`.

Two layers of math here, both worth pinning down with hand-crafted
fixtures:

- `build_per_trade_frame` and `compute_mae_mfe`: per-trade derivations
  that downstream metrics rely on. PnL sign + USD conversion are easy
  to flip; this is the place to lock them down.
- `_trade_metrics` / `_per_session_metrics`: aggregate stats that show
  up in the headline JSON and the parity-check section.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_assay.cli.report import (
    PairedTrade,
    _per_session_metrics,
    _trade_metrics,
    build_per_trade_frame,
    compute_mae_mfe,
    pair_trades,
)


def _trade(side: str, entry: float, exit_p: float, hold_min: int, reason: str = "target"):
    entry_ts = pd.Timestamp("2026-04-27 14:00:00", tz="UTC")
    exit_ts = entry_ts + pd.Timedelta(minutes=hold_min)
    return PairedTrade(
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        side=side,
        entry_price=entry,
        exit_price=exit_p,
        quantity=1.0,
        exit_reason=reason,
    )


def test_pnl_long_winner_positive():
    t = _trade("long", 100.0, 102.0, 1)
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    assert df.iloc[0]["pnl_points"] == pytest.approx(2.0)
    assert df.iloc[0]["pnl_usd"] == pytest.approx(100.0)


def test_pnl_long_loser_negative():
    t = _trade("long", 100.0, 99.0, 1)
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    assert df.iloc[0]["pnl_points"] == pytest.approx(-1.0)
    assert df.iloc[0]["pnl_usd"] == pytest.approx(-50.0)


def test_pnl_short_winner_positive():
    t = _trade("short", 100.0, 98.0, 1)
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    assert df.iloc[0]["pnl_points"] == pytest.approx(2.0)
    assert df.iloc[0]["pnl_usd"] == pytest.approx(100.0)


def test_pnl_short_loser_negative():
    t = _trade("short", 100.0, 101.5, 1)
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    assert df.iloc[0]["pnl_points"] == pytest.approx(-1.5)
    assert df.iloc[0]["pnl_usd"] == pytest.approx(-75.0)


def test_pnl_scales_with_quantity():
    t = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 14:00:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 14:01:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=102.0,
        quantity=3.0,
        exit_reason="target",
    )
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    # 2 points * $50/point * 3 contracts = $300
    assert df.iloc[0]["pnl_usd"] == pytest.approx(300.0)


def test_hold_seconds_matches_timedelta():
    t = _trade("long", 100.0, 102.0, 5)
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    assert df.iloc[0]["hold_seconds"] == pytest.approx(300.0)


def test_mae_mfe_long_window():
    # Long entry @100 exits @102. Within window: low touches 98, high
    # touches 103. MAE should be 2 (worst-against), MFE should be 3.
    bars = pd.DataFrame(
        {
            "open": [100, 99, 102],
            "high": [101, 99, 103],
            "low": [100, 98, 101],
            "close": [99, 99, 102],
            "volume": [10, 10, 10],
        },
        index=pd.DatetimeIndex(
            [
                "2026-04-27 14:00:00",
                "2026-04-27 14:01:00",
                "2026-04-27 14:02:00",
            ],
            tz="UTC",
        ),
    )
    t = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 14:00:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 14:02:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=102.0,
        quantity=1.0,
        exit_reason="target",
    )
    mae, mfe = compute_mae_mfe(t, bars)
    assert mae == pytest.approx(2.0)
    assert mfe == pytest.approx(3.0)


def test_mae_mfe_short_window():
    # Short entry @100 exits @98. Within window: high touches 102, low
    # touches 97. MAE should be 2 (worst-against on a short = up),
    # MFE should be 3 (best-for on a short = down).
    bars = pd.DataFrame(
        {
            "open": [100, 101, 98],
            "high": [102, 101, 99],
            "low": [99, 97, 97],
            "close": [101, 98, 98],
            "volume": [10, 10, 10],
        },
        index=pd.DatetimeIndex(
            [
                "2026-04-27 14:00:00",
                "2026-04-27 14:01:00",
                "2026-04-27 14:02:00",
            ],
            tz="UTC",
        ),
    )
    t = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 14:00:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 14:02:00", tz="UTC"),
        side="short",
        entry_price=100.0,
        exit_price=98.0,
        quantity=1.0,
        exit_reason="target",
    )
    mae, mfe = compute_mae_mfe(t, bars)
    assert mae == pytest.approx(2.0)
    assert mfe == pytest.approx(3.0)


def test_mae_mfe_returns_none_when_bars_missing():
    t = _trade("long", 100.0, 102.0, 1)
    mae, mfe = compute_mae_mfe(t, None)
    assert mae is None
    assert mfe is None


def test_trade_metrics_basic():
    trades = [
        _trade("long", 100.0, 102.0, 1),  # +2 pts win
        _trade("long", 100.0, 99.0, 1),  # -1 pt loss
        _trade("short", 100.0, 98.0, 2),  # +2 pts win
        _trade("short", 100.0, 101.0, 1),  # -1 pt loss
    ]
    df = build_per_trade_frame(trades, instrument_multiplier=50.0, bars=None)
    m = _trade_metrics(df)
    assert m["n_trades_total"] == 4
    assert m["n_trades_long"] == 2
    assert m["n_trades_short"] == 2
    assert m["win_rate"] == pytest.approx(0.5)
    # PF = sum(wins) / abs(sum(losses)) = (100+100) / (50+50) = 2.0
    assert m["profit_factor"] == pytest.approx(2.0)
    assert m["avg_win_usd"] == pytest.approx(100.0)
    assert m["avg_loss_usd"] == pytest.approx(-50.0)
    assert m["avg_trade_pnl_usd"] == pytest.approx(25.0)
    assert m["largest_win_usd"] == pytest.approx(100.0)
    assert m["largest_loss_usd"] == pytest.approx(-50.0)


def test_trade_metrics_all_winners_profit_factor_handles_zero_loss():
    trades = [_trade("long", 100.0, 102.0, 1) for _ in range(3)]
    df = build_per_trade_frame(trades, instrument_multiplier=50.0, bars=None)
    m = _trade_metrics(df)
    assert m["win_rate"] == pytest.approx(1.0)
    # No losses => profit_factor undefined (we return None rather than inf).
    assert m["profit_factor"] is None


def test_trade_metrics_empty_input():
    df = pd.DataFrame()
    m = _trade_metrics(df)
    assert m["n_trades_total"] == 0
    assert m["win_rate"] is None
    assert m["profit_factor"] is None


def test_per_session_metrics_buckets_by_exit_date():
    t1 = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 14:00:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 14:30:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=102.0,
        quantity=1.0,
        exit_reason="target",
    )
    t2 = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 19:00:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 19:30:00", tz="UTC"),
        side="short",
        entry_price=100.0,
        exit_price=99.0,
        quantity=1.0,
        exit_reason="target",
    )
    t3 = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-28 14:00:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-28 14:30:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=99.0,
        quantity=1.0,
        exit_reason="stop",
    )
    df = build_per_trade_frame([t1, t2, t3], instrument_multiplier=50.0, bars=None)
    sessions = _per_session_metrics(df)
    pnl = sessions["per_session_pnl_usd"]
    counts = sessions["per_session_trade_count"]
    fills = sessions["per_session_avg_fill_time_seconds"]
    assert set(pnl) == {"2026-04-27", "2026-04-28"}
    # Session 1: +100 (long winner) + 50 (short winner) = +150
    assert pnl["2026-04-27"] == pytest.approx(150.0)
    # Session 2: -50 (long loser)
    assert pnl["2026-04-28"] == pytest.approx(-50.0)
    assert counts["2026-04-27"] == 2
    assert counts["2026-04-28"] == 1
    # Trades constructed without entry_signal_ts fall back to null
    # fill latency (graceful handling of older backtest output).
    assert fills["2026-04-27"] is None
    assert fills["2026-04-28"] is None


def test_per_session_metrics_empty():
    sessions = _per_session_metrics(pd.DataFrame())
    assert sessions["per_session_pnl_usd"] == {}
    assert sessions["per_session_trade_count"] == {}
    assert sessions["per_session_avg_fill_time_seconds"] == {}


def test_fill_latency_seconds_in_per_trade_frame():
    """When entry_signal_ts is set, build_per_trade_frame must surface
    fill_latency_seconds = (entry_ts - entry_signal_ts).total_seconds().
    """
    signal_ts = pd.Timestamp("2026-04-27 14:00:00", tz="UTC")
    fill_ts = pd.Timestamp("2026-04-27 14:01:00", tz="UTC")
    t = PairedTrade(
        entry_ts=fill_ts,
        exit_ts=fill_ts + pd.Timedelta(minutes=5),
        side="long",
        entry_price=100.0,
        exit_price=102.0,
        quantity=1.0,
        exit_reason="target",
        entry_signal_ts=signal_ts,
    )
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    assert "fill_latency_seconds" in df.columns
    assert df.iloc[0]["fill_latency_seconds"] == pytest.approx(60.0)
    assert df.iloc[0]["entry_signal_ts"] == signal_ts


def test_fill_latency_seconds_falls_back_to_none_without_signal_ts():
    t = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 14:01:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 14:06:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=102.0,
        quantity=1.0,
        exit_reason="target",
    )
    df = build_per_trade_frame([t], instrument_multiplier=50.0, bars=None)
    assert "fill_latency_seconds" in df.columns
    assert df.iloc[0]["fill_latency_seconds"] is None or pd.isna(df.iloc[0]["fill_latency_seconds"])


def test_per_session_avg_fill_time_seconds_aggregates_when_signal_ts_present():
    """When fill_latency_seconds is populated, the per-session aggregate
    must be the mean across the session's trades.
    """
    # Session 1: two trades, latencies of 60s and 90s -> mean 75s.
    t1 = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 14:01:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 14:30:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=102.0,
        quantity=1.0,
        exit_reason="target",
        entry_signal_ts=pd.Timestamp("2026-04-27 14:00:00", tz="UTC"),
    )
    t2 = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 19:01:30", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 19:30:00", tz="UTC"),
        side="short",
        entry_price=100.0,
        exit_price=99.0,
        quantity=1.0,
        exit_reason="target",
        entry_signal_ts=pd.Timestamp("2026-04-27 19:00:00", tz="UTC"),
    )
    # Session 2: one trade, latency of 60s.
    t3 = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-28 14:01:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-28 14:30:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=99.0,
        quantity=1.0,
        exit_reason="stop",
        entry_signal_ts=pd.Timestamp("2026-04-28 14:00:00", tz="UTC"),
    )
    df = build_per_trade_frame([t1, t2, t3], instrument_multiplier=50.0, bars=None)
    sessions = _per_session_metrics(df)
    fills = sessions["per_session_avg_fill_time_seconds"]
    # Session 1: (60 + 90) / 2 = 75 seconds.
    assert fills["2026-04-27"] == pytest.approx(75.0)
    # Session 2: 60 seconds.
    assert fills["2026-04-28"] == pytest.approx(60.0)


def test_per_session_avg_fill_time_seconds_partial_signal_ts_coverage():
    """A session with one trade that has signal_ts and one without
    should fall back to averaging only the populated rows.
    """
    t_with = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 14:01:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 14:30:00", tz="UTC"),
        side="long",
        entry_price=100.0,
        exit_price=102.0,
        quantity=1.0,
        exit_reason="target",
        entry_signal_ts=pd.Timestamp("2026-04-27 14:00:00", tz="UTC"),
    )
    t_without = PairedTrade(
        entry_ts=pd.Timestamp("2026-04-27 19:01:00", tz="UTC"),
        exit_ts=pd.Timestamp("2026-04-27 19:30:00", tz="UTC"),
        side="short",
        entry_price=100.0,
        exit_price=99.0,
        quantity=1.0,
        exit_reason="target",
    )
    df = build_per_trade_frame([t_with, t_without], instrument_multiplier=50.0, bars=None)
    sessions = _per_session_metrics(df)
    fills = sessions["per_session_avg_fill_time_seconds"]
    assert fills["2026-04-27"] == pytest.approx(60.0)


def test_pair_trades_threads_signal_ts_through():
    """Orders with a signal_ts column must propagate to the
    PairedTrade.entry_signal_ts field.
    """
    from alpha_assay.cli.report import pair_trades

    rows = [
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "signal_ts": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:35:00+00:00",
            "signal_ts": "2026-04-27 14:29:00+00:00",
            "side": "sell",
            "price": 102.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
    ]
    paired = pair_trades(pd.DataFrame(rows))
    assert len(paired) == 1
    assert paired[0].entry_signal_ts == pd.Timestamp("2026-04-27 14:29:00", tz="UTC")


def test_pair_trades_then_metrics_end_to_end():
    # End-to-end smoke: take orders, pair them, run metrics, sanity check.
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "side": "sell",
            "price": 102.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
        {
            "timestamp": "2026-04-27 14:34:00+00:00",
            "side": "sell",
            "price": 99.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:35:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "stop_market",
        },
    ]
    paired = pair_trades(pd.DataFrame(rows))
    df = build_per_trade_frame(paired, instrument_multiplier=50.0, bars=None)
    m = _trade_metrics(df)
    assert m["n_trades_total"] == 2
    assert m["n_trades_long"] == 1
    assert m["n_trades_short"] == 1
    # Long made +$100, short lost -$50 => net +$50, win rate 50%
    assert m["avg_trade_pnl_usd"] == pytest.approx(25.0)
    assert m["win_rate"] == pytest.approx(0.5)
