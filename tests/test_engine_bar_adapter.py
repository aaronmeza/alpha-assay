import pandas as pd
import pytest
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

from alpha_assay.engine.bar_adapter import df_to_bars


@pytest.fixture
def bar_type() -> BarType:
    instrument_id = InstrumentId(Symbol("ESM6"), Venue("SIM"))
    spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
    return BarType(instrument_id, spec, AggregationSource.EXTERNAL)


def _df(rows):
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_df_to_bars_happy_path(bar_type):
    df = _df(
        [
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "ES_open": 5000.0,
                "ES_high": 5001.0,
                "ES_low": 4999.0,
                "ES_close": 5000.5,
                "ES_volume": 100,
            },
            {
                "timestamp": "2026-04-28T14:31:00Z",
                "ES_open": 5000.5,
                "ES_high": 5002.0,
                "ES_low": 5000.0,
                "ES_close": 5001.5,
                "ES_volume": 150,
            },
        ]
    )
    bars = df_to_bars(df, bar_type)
    assert len(bars) == 2
    assert all(isinstance(b, Bar) for b in bars)
    assert str(bars[0].open) == "5000.00"
    assert str(bars[1].close) == "5001.50"


def test_df_to_bars_clamps_inverted_high(bar_type):
    # After rounding, high ends up below open. Adapter must clamp
    # high = max(high, open, close) so Nautilus' invariant holds.
    df = _df(
        [
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "ES_open": 5000.00,
                "ES_high": 4999.99,  # inverted; rounds to 5000.00 still < open
                "ES_low": 4998.50,
                "ES_close": 5000.20,
                "ES_volume": 100,
            },
        ]
    )
    bars = df_to_bars(df, bar_type)
    # clamped: high = max(4999.99, 5000.00, 5000.20) = 5000.20
    assert str(bars[0].high) == "5000.20"


def test_df_to_bars_clamps_inverted_low(bar_type):
    df = _df(
        [
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "ES_open": 5000.00,
                "ES_high": 5001.00,
                "ES_low": 5000.10,  # above open; inverted
                "ES_close": 4999.50,
                "ES_volume": 100,
            },
        ]
    )
    bars = df_to_bars(df, bar_type)
    # clamped: low = min(5000.10, 5000.00, 4999.50) = 4999.50
    assert str(bars[0].low) == "4999.50"


def test_df_to_bars_empty_df_returns_empty(bar_type):
    df = pd.DataFrame(
        columns=["timestamp", "ES_open", "ES_high", "ES_low", "ES_close", "ES_volume"]
    )
    bars = df_to_bars(df, bar_type)
    assert bars == []


def test_df_to_bars_accepts_lowercase_ohlc(bar_type):
    # Some data paths use lowercase "open"/"high"/"low"/"close"/"volume"
    # (e.g. csv_replay). Adapter tries uppercase-ES_ prefix first, then
    # falls back to lowercase for the Databento canonical schema.
    df = _df(
        [
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "open": 5000.0,
                "high": 5001.0,
                "low": 4999.0,
                "close": 5000.5,
                "volume": 100,
            },
        ]
    )
    bars = df_to_bars(df, bar_type)
    assert len(bars) == 1
    assert str(bars[0].open) == "5000.00"
