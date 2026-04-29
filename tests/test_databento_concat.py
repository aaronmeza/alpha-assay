"""Shard-concat helper for bulk-pulled Databento parquet files."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


def _reload_module():
    sys.modules.pop("scripts.databento_concat", None)
    return importlib.import_module("scripts.databento_concat")


def _write_shard(path: Path, start: str, n_rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(start=start, periods=n_rows, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_event": idx,
            "open": [100.0] * n_rows,
            "high": [101.0] * n_rows,
            "low": [99.5] * n_rows,
            "close": [100.5] * n_rows,
            "volume": [10] * n_rows,
        }
    )
    df.to_parquet(path)


def test_happy_path_three_shards_merged(tmp_path):
    base = tmp_path / "data" / "ES_FUT" / "ohlcv-1m" / "2026"
    _write_shard(base / "2026-04-20.parquet", "2026-04-20T14:30:00Z")
    _write_shard(base / "2026-04-21.parquet", "2026-04-21T14:30:00Z")
    _write_shard(base / "2026-04-22.parquet", "2026-04-22T14:30:00Z")

    mod = _reload_module()
    mod.main(
        argv=[
            "--symbol",
            "ES_FUT",
            "--schema",
            "ohlcv-1m",
            "--out-dir",
            str(tmp_path / "data"),
        ]
    )

    merged_path = tmp_path / "data" / "ES_FUT_ohlcv-1m_merged.parquet"
    assert merged_path.exists()

    merged = pd.read_parquet(merged_path)
    assert len(merged) == 15
    ts = pd.to_datetime(merged["ts_event"], utc=True)
    assert ts.is_monotonic_increasing


def test_empty_directory_warns_and_writes_nothing(tmp_path, capsys):
    mod = _reload_module()
    (tmp_path / "data" / "ES_FUT" / "ohlcv-1m").mkdir(parents=True)

    mod.main(
        argv=[
            "--symbol",
            "ES_FUT",
            "--schema",
            "ohlcv-1m",
            "--out-dir",
            str(tmp_path / "data"),
        ]
    )

    merged_path = tmp_path / "data" / "ES_FUT_ohlcv-1m_merged.parquet"
    assert not merged_path.exists()
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "no shards" in combined.lower() or "warning" in combined.lower()


def test_non_monotonic_shards_sorted(tmp_path):
    """Unsorted shards must be sorted into monotonic merged output."""
    base = tmp_path / "data" / "ES_FUT" / "ohlcv-1m" / "2026"
    # write shards in reverse chronological order on disk — but concat must sort
    _write_shard(base / "2026-04-22.parquet", "2026-04-22T14:30:00Z")
    _write_shard(base / "2026-04-20.parquet", "2026-04-20T14:30:00Z")
    _write_shard(base / "2026-04-21.parquet", "2026-04-21T14:30:00Z")

    mod = _reload_module()
    mod.main(
        argv=[
            "--symbol",
            "ES_FUT",
            "--schema",
            "ohlcv-1m",
            "--out-dir",
            str(tmp_path / "data"),
        ]
    )

    merged = pd.read_parquet(tmp_path / "data" / "ES_FUT_ohlcv-1m_merged.parquet")
    ts = pd.to_datetime(merged["ts_event"], utc=True)
    assert ts.is_monotonic_increasing
    assert len(merged) == 15


def test_merged_path_override(tmp_path):
    base = tmp_path / "data" / "ES_FUT" / "ohlcv-1m" / "2026"
    _write_shard(base / "2026-04-20.parquet", "2026-04-20T14:30:00Z")

    mod = _reload_module()
    custom_out = tmp_path / "custom" / "my_merged.parquet"
    mod.main(
        argv=[
            "--symbol",
            "ES_FUT",
            "--schema",
            "ohlcv-1m",
            "--out-dir",
            str(tmp_path / "data"),
            "--merged-path",
            str(custom_out),
        ]
    )
    assert custom_out.exists()
