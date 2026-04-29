"""The smoke-pull CLI calls the Databento SDK with the expected kwargs
and writes to data/databento_smoke/. The SDK is mocked; no network
required.
"""

from unittest.mock import MagicMock, patch

import pytest


def test_smoke_pull_main_calls_databento_with_expected_args(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key-abc")
    monkeypatch.chdir(tmp_path)

    mock_client = MagicMock()
    mock_client.timeseries.get_range.return_value.to_df.return_value = None

    with patch("databento.Historical", return_value=mock_client) as Historical:
        from scripts import databento_smoke_pull

        databento_smoke_pull.main(argv=["--symbol", "ES.FUT", "--date", "2026-04-21"])

    Historical.assert_called_once_with(key="test-key-abc")
    call = mock_client.timeseries.get_range.call_args
    assert call.kwargs["dataset"] == "GLBX.MDP3"
    assert call.kwargs["symbols"] == "ES.FUT"
    assert call.kwargs["schema"] == "ohlcv-1m"
    assert str(call.kwargs["start"]).startswith("2026-04-21")


def test_smoke_pull_refuses_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    from scripts import databento_smoke_pull

    with pytest.raises(SystemExit):
        databento_smoke_pull.main(argv=["--symbol", "ES.FUT", "--date", "2026-04-21"])
