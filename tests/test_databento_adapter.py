import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alpha_assay.data.databento_adapter import DatabentoSchemaError, load_parquet


def _write_parquet(path, rows):
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), path)


def test_load_parquet_happy_path(tmp_path):
    p = tmp_path / "es.parquet"
    _write_parquet(
        p,
        [
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "open": 5000.0,
                "high": 5001.0,
                "low": 4999.0,
                "close": 5000.5,
                "volume": 100,
            },
            {
                "timestamp": "2026-04-28T14:31:00Z",
                "open": 5000.5,
                "high": 5002.0,
                "low": 5000.0,
                "close": 5001.5,
                "volume": 150,
            },
        ],
    )
    df = load_parquet(p)
    assert len(df) == 2
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    # tz normalized to America/Chicago
    assert str(df.index.tz) == "America/Chicago"


def test_load_parquet_rejects_missing_column(tmp_path):
    p = tmp_path / "bad.parquet"
    _write_parquet(
        p,
        [
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "open": 5000.0,
                "high": 5001.0,
                "low": 4999.0,
                "close": 5000.5,
            },
        ],
    )
    with pytest.raises(DatabentoSchemaError, match="volume"):
        load_parquet(p)


def test_load_parquet_rejects_non_monotonic(tmp_path):
    p = tmp_path / "bad.parquet"
    _write_parquet(
        p,
        [
            {
                "timestamp": "2026-04-28T14:31:00Z",
                "open": 5000.5,
                "high": 5002.0,
                "low": 5000.0,
                "close": 5001.5,
                "volume": 150,
            },
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "open": 5000.0,
                "high": 5001.0,
                "low": 4999.0,
                "close": 5000.5,
                "volume": 100,
            },
        ],
    )
    with pytest.raises(DatabentoSchemaError, match="monotonic"):
        load_parquet(p)


def test_load_parquet_accepts_databento_naming_variants(tmp_path):
    """Databento schemas vary by product; the adapter accepts both
    `timestamp` and `ts_event` as the index column.
    """
    p = tmp_path / "variant.parquet"
    _write_parquet(
        p,
        [
            {
                "ts_event": "2026-04-28T14:30:00Z",
                "open": 5000.0,
                "high": 5001.0,
                "low": 4999.0,
                "close": 5000.5,
                "volume": 100,
            },
        ],
    )
    df = load_parquet(p)
    assert len(df) == 1


def test_load_parquet_rejects_inverted_high_low(tmp_path):
    p = tmp_path / "bad.parquet"
    _write_parquet(
        p,
        [
            {
                "timestamp": "2026-04-28T14:30:00Z",
                "open": 5000.0,
                "high": 4999.0,
                "low": 5001.0,
                "close": 5000.5,
                "volume": 100,
            },
        ],
    )
    with pytest.raises(DatabentoSchemaError, match="high.*low"):
        load_parquet(p)
