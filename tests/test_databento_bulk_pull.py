"""Resume-safe Databento bulk pull CLI.

All tests mock the databento SDK — no network, no credentials required.
"""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _reload_module():
    """Ensure a fresh import each test (fresh CWD, fresh argparse state)."""
    sys.modules.pop("scripts.databento_bulk_pull", None)
    return importlib.import_module("scripts.databento_bulk_pull")


def _mock_df(n_rows: int = 10, start: str = "2026-04-20T14:30:00Z") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n_rows, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "ts_event": idx,
            "open": [100.0] * n_rows,
            "high": [101.0] * n_rows,
            "low": [99.5] * n_rows,
            "close": [100.5] * n_rows,
            "volume": [10] * n_rows,
        }
    )


def _install_client_mock(monkeypatch, per_call_rows: int = 10):
    """Install a databento.Historical mock; each get_range returns a fresh df."""
    mock_client = MagicMock()

    def _get_range(**kwargs):
        resp = MagicMock()
        resp.to_df.return_value = _mock_df(
            n_rows=per_call_rows,
            start=f"{kwargs['start'][:10]}T14:30:00Z",
        )
        return resp

    mock_client.timeseries.get_range.side_effect = _get_range
    mock_client.metadata.get_cost.return_value = 1.23
    return mock_client


def test_happy_path_three_days_two_schemas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    mock_client = _install_client_mock(monkeypatch)
    mod = _reload_module()

    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",  # Monday
                "--end-date",
                "2026-04-22",  # Wednesday
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m,mbp-1",
            ]
        )

    base = tmp_path / "data" / "ES_FUT"
    # 3 days x 2 schemas = 6 parquet shards
    shards = list(base.rglob("*.parquet"))
    assert len(shards) == 6, f"expected 6 shards, got {len(shards)}: {shards}"

    manifest = json.loads((base / "manifest.json").read_text())
    assert len(manifest) == 6
    schemas = {entry["schema"] for entry in manifest}
    assert schemas == {"ohlcv-1m", "mbp-1"}
    dates = {entry["date"] for entry in manifest}
    assert dates == {"2026-04-20", "2026-04-21", "2026-04-22"}

    # expected path structure
    assert (base / "ohlcv-1m" / "2026" / "2026-04-20.parquet").exists()
    assert (base / "mbp-1" / "2026" / "2026-04-22.parquet").exists()


def test_resume_skips_existing_days(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    mock_client = _install_client_mock(monkeypatch)
    mod = _reload_module()

    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-22",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
            ]
        )
    first_call_count = mock_client.timeseries.get_range.call_count
    assert first_call_count == 3

    manifest_path = tmp_path / "data" / "ES_FUT" / "manifest.json"
    manifest_before = json.loads(manifest_path.read_text())

    # re-run same range
    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-22",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
            ]
        )
    assert mock_client.timeseries.get_range.call_count == first_call_count, "resume should skip"

    manifest_after = json.loads(manifest_path.read_text())
    assert manifest_after == manifest_before, "manifest should be unchanged on resume"


def test_force_repulls(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    mock_client = _install_client_mock(monkeypatch)
    mod = _reload_module()

    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-20",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
            ]
        )
    assert mock_client.timeseries.get_range.call_count == 1

    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-20",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
                "--force",
            ]
        )
    assert mock_client.timeseries.get_range.call_count == 2


def test_weekend_skip(tmp_path, monkeypatch):
    """A Sat/Sun in the range yields no API call."""
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    mock_client = _install_client_mock(monkeypatch)
    mod = _reload_module()

    # 2026-04-18 is Saturday, 2026-04-19 is Sunday, 2026-04-20 is Monday
    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-18",
                "--end-date",
                "2026-04-20",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
            ]
        )
    # only Monday should trigger a call
    assert mock_client.timeseries.get_range.call_count == 1
    call = mock_client.timeseries.get_range.call_args
    assert call.kwargs["start"].startswith("2026-04-20")


def test_estimate_only_invokes_get_cost_and_skips_pull(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    mock_client = _install_client_mock(monkeypatch)
    mod = _reload_module()

    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-22",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m,mbp-1",
                "--estimate-only",
            ]
        )

    # one get_cost per schema, no timeseries.get_range calls
    assert mock_client.metadata.get_cost.call_count == 2
    assert mock_client.timeseries.get_range.call_count == 0

    captured = capsys.readouterr()
    # should print projected USD total somewhere
    assert "USD" in captured.out or "usd" in captured.out.lower()


def test_missing_api_key_exits_2(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    mod = _reload_module()

    with pytest.raises(SystemExit) as exc:
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-20",
                "--out-dir",
                str(tmp_path / "data"),
            ]
        )
    assert exc.value.code == 2


def test_symbol_sanitization(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    mock_client = _install_client_mock(monkeypatch)
    mod = _reload_module()

    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-20",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
            ]
        )
    assert (tmp_path / "data" / "ES_FUT").is_dir()
    assert not (tmp_path / "data" / "ES.FUT").exists()

    # continuous contract style
    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.c.0",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-20",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
            ]
        )
    assert (tmp_path / "data" / "ES_c_0").is_dir()


def test_idempotent_manifest_on_force_replaces_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    mock_client = _install_client_mock(monkeypatch)
    mod = _reload_module()

    # initial 3-day pull
    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-22",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
            ]
        )
    manifest_path = tmp_path / "data" / "ES_FUT" / "manifest.json"
    entries = json.loads(manifest_path.read_text())
    assert len(entries) == 3

    # force re-pull just day 1
    with patch("databento.Historical", return_value=mock_client):
        mod.main(
            argv=[
                "--symbol",
                "ES.FUT",
                "--start-date",
                "2026-04-20",
                "--end-date",
                "2026-04-20",
                "--out-dir",
                str(tmp_path / "data"),
                "--schemas",
                "ohlcv-1m",
                "--force",
            ]
        )
    entries_after = json.loads(manifest_path.read_text())
    # still 3 entries — the (2026-04-20, ohlcv-1m) entry is REPLACED, not appended
    assert len(entries_after) == 3
    key = ("2026-04-20", "ohlcv-1m")
    matches = [e for e in entries_after if (e["date"], e["schema"]) == key]
    assert len(matches) == 1, "exactly one entry for the re-pulled day"


def test_sanitize_symbol_helper():
    mod = _reload_module()
    assert mod.sanitize_symbol("ES.FUT") == "ES_FUT"
    assert mod.sanitize_symbol("ES.c.0") == "ES_c_0"
    assert mod.sanitize_symbol("ESM6") == "ESM6"


def test_daterange_helper_filters_weekends():
    mod = _reload_module()
    days = list(mod.iter_trading_days("2026-04-18", "2026-04-22"))
    # Sat/Sun skipped, Mon-Wed included
    assert days == ["2026-04-20", "2026-04-21", "2026-04-22"]
