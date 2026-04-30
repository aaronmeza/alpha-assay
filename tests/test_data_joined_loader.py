# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Unit tests for `data/joined_loader.py`.

Schema-and-shape assertions; the end-to-end CLI integration lives in
`test_cli_backtest_with_breadth.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alpha_assay.data.joined_loader import BreadthSchemaError, load_es_with_breadth


def _write_es(path: Path, idx: pd.DatetimeIndex) -> None:
    n = len(idx)
    rng = np.random.default_rng(3)
    close = 5000 + np.cumsum(rng.normal(0, 1.0, n))
    high = close + 0.5
    low = close - 0.5
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.integers(50, 500, n),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), path)


def _write_breadth(path: Path, idx: pd.DatetimeIndex, values: np.ndarray, symbol: str) -> None:
    n = len(idx)
    df = pd.DataFrame(
        {
            "timestamp": idx.tz_convert("UTC"),
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "n_ticks": np.full(n, 30, dtype=np.int64),
            "symbol": [symbol] * n,
        }
    )
    pq.write_table(pa.Table.from_pandas(df), path)


def test_load_es_with_breadth_returns_canonical_columns(tmp_path):
    idx = pd.date_range("2026-04-13 09:00", periods=60, freq="1min", tz="America/Chicago")
    es = tmp_path / "es.parquet"
    tick = tmp_path / "tick.parquet"
    ad = tmp_path / "ad.parquet"
    _write_es(es, idx)
    _write_breadth(tick, idx, np.full(60, -100.0), "TICK-NYSE")
    _write_breadth(ad, idx, np.full(60, 1200.0), "AD-NYSE")

    df = load_es_with_breadth(es, tick_path=tick, ad_path=ad)
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "America/Chicago"
    for col in ("open", "high", "low", "close", "volume", "TICK", "ADD"):
        assert col in df.columns
    assert (df["TICK"] == -100.0).all()
    assert (df["ADD"] == 1200.0).all()
    assert len(df) == 60


def test_load_es_with_breadth_inner_join_drops_unmatched(tmp_path):
    es_idx = pd.date_range("2026-04-13 09:00", periods=60, freq="1min", tz="America/Chicago")
    tick_idx = es_idx[10:50]  # narrower window
    ad_idx = es_idx[10:50]
    es = tmp_path / "es.parquet"
    tick = tmp_path / "tick.parquet"
    ad = tmp_path / "ad.parquet"
    _write_es(es, es_idx)
    _write_breadth(tick, tick_idx, np.full(40, -50.0), "TICK-NYSE")
    _write_breadth(ad, ad_idx, np.full(40, 800.0), "AD-NYSE")

    df = load_es_with_breadth(es, tick_path=tick, ad_path=ad)
    assert len(df) == 40


def test_load_es_with_breadth_omits_absent_breadth(tmp_path):
    idx = pd.date_range("2026-04-13 09:00", periods=30, freq="1min", tz="America/Chicago")
    es = tmp_path / "es.parquet"
    _write_es(es, idx)

    df = load_es_with_breadth(es)
    assert "TICK" not in df.columns
    assert "ADD" not in df.columns
    assert len(df) == 30


def test_load_es_with_breadth_empty_breadth_file_raises(tmp_path):
    idx = pd.date_range("2026-04-13 09:00", periods=30, freq="1min", tz="America/Chicago")
    es = tmp_path / "es.parquet"
    tick = tmp_path / "tick.parquet"
    ad = tmp_path / "ad.parquet"
    _write_es(es, idx)
    # zero-row breadth file
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime([], utc=True),
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "n_ticks": pd.Series([], dtype="int64"),
                    "symbol": [],
                }
            )
        ),
        tick,
    )
    _write_breadth(ad, idx, np.full(30, 800.0), "AD-NYSE")

    with pytest.raises(BreadthSchemaError, match="zero rows"):
        load_es_with_breadth(es, tick_path=tick, ad_path=ad)


def test_load_es_with_breadth_no_overlap_raises(tmp_path):
    es_idx = pd.date_range("2024-04-13 09:00", periods=30, freq="1min", tz="America/Chicago")
    breadth_idx = pd.date_range("2024-05-13 09:00", periods=30, freq="1min", tz="America/Chicago")
    es = tmp_path / "es.parquet"
    tick = tmp_path / "tick.parquet"
    ad = tmp_path / "ad.parquet"
    _write_es(es, es_idx)
    _write_breadth(tick, breadth_idx, np.full(30, -50.0), "TICK-NYSE")
    _write_breadth(ad, breadth_idx, np.full(30, 800.0), "AD-NYSE")

    with pytest.raises(BreadthSchemaError, match="empty"):
        load_es_with_breadth(es, tick_path=tick, ad_path=ad)
