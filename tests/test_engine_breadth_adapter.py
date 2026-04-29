import pandas as pd
import pytest
from nautilus_trader.model.data import CustomData, DataType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

from alpha_assay.engine.breadth_adapter import df_to_breadth
from alpha_assay.engine.custom_data import AddIndicator, TickIndicator


def _df(rows):
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_df_to_breadth_wraps_both_feeds_in_custom_data():
    instrument_id = InstrumentId(Symbol("ESM6"), Venue("SIM"))
    df = _df(
        [
            {"timestamp": "2026-04-28T14:30:00Z", "TICK": -842.0, "ADD": 1250.0},
            {"timestamp": "2026-04-28T14:31:00Z", "TICK": 200.0, "ADD": 1500.0},
        ]
    )
    ticks, adds = df_to_breadth(df, instrument_id)
    assert len(ticks) == 2
    assert len(adds) == 2
    assert all(isinstance(e, CustomData) for e in ticks)
    assert all(isinstance(e, CustomData) for e in adds)


def test_df_to_breadth_ts_event_matches_timestamp_column():
    instrument_id = InstrumentId(Symbol("ESM6"), Venue("SIM"))
    df = _df(
        [
            {"timestamp": "2026-04-28T14:30:00Z", "TICK": -10.0, "ADD": 100.0},
        ]
    )
    ticks, adds = df_to_breadth(df, instrument_id)
    # CustomData.data.ts_event is the inner Data's ts_event, in unix nanos.
    expected_ns = int(pd.Timestamp("2026-04-28T14:30:00Z").value)
    assert ticks[0].data.ts_event == expected_ns
    assert adds[0].data.ts_event == expected_ns


def test_df_to_breadth_data_types_are_correct():
    instrument_id = InstrumentId(Symbol("ESM6"), Venue("SIM"))
    df = _df(
        [
            {"timestamp": "2026-04-28T14:30:00Z", "TICK": -10.0, "ADD": 100.0},
        ]
    )
    ticks, adds = df_to_breadth(df, instrument_id)
    assert ticks[0].data_type == DataType(TickIndicator)
    assert adds[0].data_type == DataType(AddIndicator)
    assert isinstance(ticks[0].data, TickIndicator)
    assert isinstance(adds[0].data, AddIndicator)


def test_df_to_breadth_preserves_signed_values():
    instrument_id = InstrumentId(Symbol("ESM6"), Venue("SIM"))
    df = _df(
        [
            {"timestamp": "2026-04-28T14:30:00Z", "TICK": -842.0, "ADD": -500.0},
        ]
    )
    ticks, adds = df_to_breadth(df, instrument_id)
    assert ticks[0].data.value == -842.0
    assert adds[0].data.value == -500.0


def test_df_to_breadth_780_row_fixture_shape():
    # Replicate the 780-row synthetic fixture shape. Two RTH days
    # at 390 bars each = 780.
    import numpy as np

    instrument_id = InstrumentId(Symbol("ESM6"), Venue("SIM"))
    idx = pd.date_range("2026-04-28 09:00", periods=780, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "TICK": rng.normal(0, 200, 780),
            "ADD": rng.normal(0, 500, 780),
        }
    )
    ticks, adds = df_to_breadth(df, instrument_id)
    assert len(ticks) == 780
    assert len(adds) == 780


def test_df_to_breadth_missing_columns_raises():
    instrument_id = InstrumentId(Symbol("ESM6"), Venue("SIM"))
    df = _df([{"timestamp": "2026-04-28T14:30:00Z", "TICK": 0.0}])  # no ADD
    with pytest.raises(KeyError, match="ADD"):
        df_to_breadth(df, instrument_id)
