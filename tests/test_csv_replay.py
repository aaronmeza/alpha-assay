import pandas as pd
import pytest

from alpha_assay.data.csv_replay import DataQualityError, load_csv_bars


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_loads_csv_and_normalizes_tz(tmp_path):
    p = tmp_path / "bars.csv"
    _write_csv(
        p,
        [
            {
                "timestamp": "2026-04-21T14:00:00Z",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100.5,
                "volume": 50,
            },
            {
                "timestamp": "2026-04-21T14:01:00Z",
                "open": 100.5,
                "high": 101.5,
                "low": 100,
                "close": 101,
                "volume": 60,
            },
        ],
    )
    df = load_csv_bars(p)
    assert str(df.index.tz) == "America/Chicago"
    assert len(df) == 2
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_rejects_non_monotonic(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv(
        p,
        [
            {
                "timestamp": "2026-04-21T14:01:00Z",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100.5,
                "volume": 50,
            },
            {
                "timestamp": "2026-04-21T14:00:00Z",
                "open": 100.5,
                "high": 101.5,
                "low": 100,
                "close": 101,
                "volume": 60,
            },
        ],
    )
    with pytest.raises(DataQualityError, match="monotonic"):
        load_csv_bars(p)


def test_rejects_inverted_high_low(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv(
        p,
        [
            {
                "timestamp": "2026-04-21T14:00:00Z",
                "open": 100,
                "high": 99,
                "low": 101,
                "close": 100.5,
                "volume": 50,
            },
        ],
    )
    with pytest.raises(DataQualityError, match="high.*low"):
        load_csv_bars(p)


def test_rejects_negative_volume(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv(
        p,
        [
            {
                "timestamp": "2026-04-21T14:00:00Z",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100.5,
                "volume": -1,
            },
        ],
    )
    with pytest.raises(DataQualityError, match="volume"):
        load_csv_bars(p)


def test_rejects_nan_ohlc(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv(
        p,
        [
            {
                "timestamp": "2026-04-21T14:00:00Z",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": float("nan"),
                "volume": 50,
            },
        ],
    )
    with pytest.raises(DataQualityError, match="NaN"):
        load_csv_bars(p)
