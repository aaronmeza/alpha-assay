# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-trade output writer."""

from __future__ import annotations

import pandas as pd

from alpha_assay.exec.trade_log import TradeLog, TradeRecord


def test_trade_log_appends_to_parquet(tmp_path):
    log = TradeLog(out_dir=tmp_path / "paper-live")
    log.write(
        TradeRecord(
            timestamp=pd.Timestamp("2026-05-06T13:30:00", tz="UTC"),
            signal_type="long_entry",
            entry_price=7250.5,
            stop=7245.0,
            target=7260.0,
            mock_fill_price=7250.75,
            mock_pnl_dollars=0.0,
            account_balance_after=100_000.0,
        )
    )
    log.flush()

    files = list((tmp_path / "paper-live").glob("trades.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert len(df) == 1
    assert df.iloc[0]["signal_type"] == "long_entry"


def test_trade_log_multiple_appends_in_one_session(tmp_path):
    log = TradeLog(out_dir=tmp_path / "paper-live")
    for i in range(5):
        log.write(
            TradeRecord(
                timestamp=pd.Timestamp(f"2026-05-06T13:30:0{i}", tz="UTC"),
                signal_type="long_entry",
                entry_price=100.0 + i,
                stop=99.0,
                target=101.0,
                mock_fill_price=100.0 + i,
                mock_pnl_dollars=0.0,
                account_balance_after=100_000.0,
            )
        )
    log.flush()

    df = pd.read_parquet(tmp_path / "paper-live" / "trades.parquet")
    assert len(df) == 5


def test_trade_log_preserves_history_across_sessions(tmp_path):
    """Re-opening the log appends to existing rows, doesn't overwrite."""
    log1 = TradeLog(out_dir=tmp_path / "paper-live")
    log1.write(TradeRecord(
        timestamp=pd.Timestamp("2026-05-06T13:30:00", tz="UTC"),
        signal_type="long_entry", entry_price=100.0, stop=99.0, target=101.0,
        mock_fill_price=100.0, mock_pnl_dollars=0.0, account_balance_after=100_000.0,
    ))
    log1.flush()

    # New session, same path.
    log2 = TradeLog(out_dir=tmp_path / "paper-live")
    log2.write(TradeRecord(
        timestamp=pd.Timestamp("2026-05-07T13:30:00", tz="UTC"),
        signal_type="short_entry", entry_price=200.0, stop=201.0, target=199.0,
        mock_fill_price=200.0, mock_pnl_dollars=5.0, account_balance_after=100_005.0,
    ))
    log2.flush()

    df = pd.read_parquet(tmp_path / "paper-live" / "trades.parquet")
    assert len(df) == 2
    # Sorted oldest first (insertion order preserved).
    assert df.iloc[0]["signal_type"] == "long_entry"
    assert df.iloc[1]["signal_type"] == "short_entry"
