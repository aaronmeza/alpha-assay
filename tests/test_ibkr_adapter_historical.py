# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for IBKRAdapter.historical_bars_async (backfill)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from alpha_assay.data.ibkr_adapter import IBKRAdapter


def _epoch(s: str) -> int:
    """Parse a UTC ISO timestamp into epoch seconds (formatDate=2 shape)."""
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def _fake_bar(*, ts_iso: str, o: float, h: float, low: float, c: float, vol: int) -> SimpleNamespace:
    """Build a SimpleNamespace that quacks like ib_insync.BarData with formatDate=2."""
    return SimpleNamespace(
        date=_epoch(ts_iso),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=vol,
        average=(o + c) / 2.0,
        barCount=42,
    )


def test_historical_bars_async_returns_canonical_schema():
    fake_bars = [
        _fake_bar(
            ts_iso="2026-04-21T13:30:00",
            o=5200.0,
            h=5202.0,
            low=5199.5,
            c=5201.25,
            vol=100,
        ),
        _fake_bar(
            ts_iso="2026-04-21T13:31:00",
            o=5201.25,
            h=5203.0,
            low=5200.0,
            c=5202.5,
            vol=120,
        ),
    ]

    ib = MagicMock(name="IB")
    ib.reqHistoricalDataAsync = AsyncMock(return_value=fake_bars)

    adapter = IBKRAdapter(ib=ib)
    spec = {
        "symbol": "ES",
        "sec_type": "FUT",
        "exchange": "CME",
        "currency": "USD",
        "expiry": "20260618",
    }

    out = asyncio.run(
        adapter.historical_bars_async(
            spec,
            end_datetime="20260421 14:00:00 UTC",
            duration_str="1 W",
        )
    )

    assert len(out) == 2
    for bar in out:
        assert set(bar.keys()) == {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "feed",
        }
        assert isinstance(bar["timestamp"], pd.Timestamp)
        assert bar["timestamp"].tzinfo is not None
        assert str(bar["timestamp"].tz) == "UTC"
        assert bar["feed"] == "ES-FUT-20260618"

    # Timestamps round-trip the epoch seconds correctly.
    assert out[0]["timestamp"] == pd.Timestamp("2026-04-21T13:30:00", tz="UTC")
    assert out[1]["timestamp"] == pd.Timestamp("2026-04-21T13:31:00", tz="UTC")

    # Verify we requested the right IBKR parameters (no keepUpToDate; formatDate=2).
    ib.reqHistoricalDataAsync.assert_awaited_once()
    kwargs = ib.reqHistoricalDataAsync.call_args.kwargs
    assert kwargs["endDateTime"] == "20260421 14:00:00 UTC"
    assert kwargs["durationStr"] == "1 W"
    assert kwargs["barSizeSetting"] == "1 min"
    assert kwargs["whatToShow"] == "TRADES"
    assert kwargs["useRTH"] is False
    assert kwargs["formatDate"] == 2
    assert "keepUpToDate" not in kwargs


def test_historical_bars_async_empty_result_returns_empty_list():
    ib = MagicMock(name="IB")
    ib.reqHistoricalDataAsync = AsyncMock(return_value=[])

    adapter = IBKRAdapter(ib=ib)
    spec = {
        "symbol": "ES",
        "sec_type": "FUT",
        "exchange": "CME",
        "currency": "USD",
        "expiry": "20260618",
    }

    out = asyncio.run(adapter.historical_bars_async(spec))
    assert out == []


def test_historical_bars_async_clamps_ohlc_invariants():
    """A bar with high < close must come back clamped (per ADR Appendix A)."""
    bad = _fake_bar(
        ts_iso="2026-04-21T13:30:00",
        o=5200.0,
        h=5200.5,  # below close => violation
        low=5199.0,
        c=5202.25,
        vol=100,
    )
    ib = MagicMock(name="IB")
    ib.reqHistoricalDataAsync = AsyncMock(return_value=[bad])

    adapter = IBKRAdapter(ib=ib)
    spec = {
        "symbol": "ES",
        "sec_type": "FUT",
        "exchange": "CME",
        "expiry": "20260618",
    }
    out = asyncio.run(adapter.historical_bars_async(spec))

    assert len(out) == 1
    bar = out[0]
    assert bar["high"] >= max(bar["open"], bar["close"])
    assert bar["low"] <= min(bar["open"], bar["close"])
    assert bar["high"] == 5202.25  # clamped up to close


def test_historical_bars_async_uses_rth_flag():
    ib = MagicMock(name="IB")
    ib.reqHistoricalDataAsync = AsyncMock(return_value=[])

    adapter = IBKRAdapter(ib=ib)
    spec = {
        "symbol": "ES",
        "sec_type": "FUT",
        "exchange": "CME",
        "expiry": "20260618",
    }
    asyncio.run(adapter.historical_bars_async(spec, use_rth=True))
    assert ib.reqHistoricalDataAsync.call_args.kwargs["useRTH"] is True
