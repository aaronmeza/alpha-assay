# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Unit tests for the always-flat paper dry-run.

All tests mock IBKR end-to-end (same pattern as ``tests/test_ibkr_adapter.py``).
The key invariants under test:

- The dry-run strategy NEVER calls ``exec_adapter.place_bracket_order``
  regardless of bar / breadth content.
- Every received bar increments
  ``alpha_assay_bars_processed_total{feed="es"}``; every breadth tick
  increments the ``feed="tick_nyse"`` child of the same counter.
- Disconnect events are handled gracefully (no crash, heartbeat
  continues, reconnect logged).
- Connection params come from environment variables.

The live end-to-end test lives in ``tests/integration/test_e2e_paper_dryrun.py``
and is opt-in via ``RUN_LIVE_E2E=1``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from alpha_assay.observability import metrics as M

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "paper_dryrun.py"
_MODULE_NAME = "paper_dryrun"


def _load_script_module():
    # Cache in sys.modules so dataclasses' string-annotation resolution
    # (which looks up cls.__module__ in sys.modules) succeeds for the
    # script's @dataclass declarations.
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


def _make_bar(feed: str = "ES-FUT-20260618") -> dict:
    return {
        "timestamp": "2026-04-27T14:30:00+00:00",
        "open": 5200.0,
        "high": 5201.0,
        "low": 5199.5,
        "close": 5200.5,
        "volume": 123,
        "feed": feed,
    }


def _make_tick(symbol: str = "TICK-NYSE") -> dict:
    return {"timestamp": "2026-04-27T14:30:00+00:00", "value": 245.0, "symbol": symbol}


# --- always-flat invariant ---------------------------------------------


def test_dryrun_strategy_never_submits_order():
    """Feed 100 mock bars + 100 mock breadth ticks; assert
    ``place_bracket_order`` is NEVER called."""
    module = _load_script_module()

    exec_adapter = MagicMock(name="IBKRExecAdapter")
    strategy = module.AlwaysFlatStrategy(exec_adapter=exec_adapter)

    for _ in range(100):
        strategy.on_bar(_make_bar(), feed_label="es")
    for _ in range(100):
        strategy.on_breadth_tick(_make_tick(), feed_label="tick_nyse")

    exec_adapter.place_bracket_order.assert_not_called()


def test_dryrun_strategy_never_returns_nonzero_signal():
    """Verify the strategy's ``decide`` method only ever returns the
    flat / no-op signal regardless of inputs."""
    module = _load_script_module()

    exec_adapter = MagicMock(name="IBKRExecAdapter")
    strategy = module.AlwaysFlatStrategy(exec_adapter=exec_adapter)

    # Throw a wide range of bar contents at the strategy.
    for close in (1.0, 5200.0, 99999.0, -1.0):
        bar = _make_bar()
        bar["close"] = close
        sig = strategy.decide(bar)
        assert sig == 0, f"AlwaysFlatStrategy.decide must always return 0, got {sig}"


# --- per-feed metric labels --------------------------------------------


def test_dryrun_increments_bars_processed_per_feed_label():
    module = _load_script_module()

    exec_adapter = MagicMock(name="IBKRExecAdapter")
    strategy = module.AlwaysFlatStrategy(exec_adapter=exec_adapter)

    es_before = _counter_value(M.bars_processed_total, feed="es")
    tick_before = _counter_value(M.bars_processed_total, feed="tick_nyse")

    for _ in range(7):
        strategy.on_bar(_make_bar(), feed_label="es")
    for _ in range(11):
        strategy.on_breadth_tick(_make_tick(), feed_label="tick_nyse")

    es_after = _counter_value(M.bars_processed_total, feed="es")
    tick_after = _counter_value(M.bars_processed_total, feed="tick_nyse")

    assert es_after == es_before + 7
    assert tick_after == tick_before + 11


# --- disconnect handling -----------------------------------------------


def test_dryrun_handles_disconnect_event_gracefully():
    """Mock a disconnect; assert the strategy keeps a record of it and
    does not raise."""
    module = _load_script_module()

    exec_adapter = MagicMock(name="IBKRExecAdapter")
    strategy = module.AlwaysFlatStrategy(exec_adapter=exec_adapter)

    # Should be a no-op for the always-flat case but must not raise.
    strategy.on_disconnect()
    strategy.on_disconnect()

    # The strategy tracks disconnect count for observability / heartbeat.
    assert strategy.disconnect_count == 2

    # Subsequent bars still increment the counter.
    before = _counter_value(M.bars_processed_total, feed="es")
    strategy.on_bar(_make_bar(), feed_label="es")
    after = _counter_value(M.bars_processed_total, feed="es")
    assert after == before + 1


# --- env-driven connection params --------------------------------------


def test_dryrun_reads_env_for_connection_params(monkeypatch):
    module = _load_script_module()

    monkeypatch.setenv("IBKR_HOST", "10.0.0.99")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "77")
    monkeypatch.setenv("IBKR_ACCOUNT", "DU1234567")
    monkeypatch.setenv("METRICS_PORT", "18000")
    monkeypatch.setenv("ES_EXPIRY", "20260618")

    cfg = module.load_config_from_env()

    assert cfg.ibkr_host == "10.0.0.99"
    assert cfg.ibkr_port == 4002
    assert cfg.ibkr_client_id == 77
    assert cfg.ibkr_account == "DU1234567"
    assert cfg.metrics_port == 18000
    assert cfg.es_expiry == "20260618"


def test_dryrun_defaults_when_env_unset(monkeypatch):
    """Confirm the documented defaults match spec."""
    module = _load_script_module()

    for key in (
        "IBKR_HOST",
        "IBKR_PORT",
        "IBKR_CLIENT_ID",
        "IBKR_ACCOUNT",
        "METRICS_PORT",
        "ES_EXPIRY",
        "DRYRUN_DURATION_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = module.load_config_from_env()

    assert cfg.ibkr_host == "127.0.0.1"
    assert cfg.ibkr_port == 4002
    assert cfg.ibkr_client_id == 1
    assert cfg.ibkr_account == ""
    assert cfg.metrics_port == 8000
    # Documented hardcoded fallback for ES front-month (June 2026 on 2026-04-27).
    assert cfg.es_expiry == "20260618"
    assert cfg.duration_seconds == 0


# --- contract spec -----------------------------------------------------


def test_dryrun_es_contract_spec_uses_env_expiry(monkeypatch):
    module = _load_script_module()

    monkeypatch.setenv("ES_EXPIRY", "202509")
    cfg = module.load_config_from_env()
    spec = module.es_contract_spec(cfg)

    assert spec["symbol"] == "ES"
    assert spec["sec_type"] == "FUT"
    assert spec["exchange"] == "CME"
    assert spec["currency"] == "USD"
    assert spec["expiry"] == "202509"


# --- script wiring -----------------------------------------------------


def test_script_constructs_read_only_adapter_in_dry_run(monkeypatch):
    """The script must construct the read-side adapter with
    ``read_only=True`` AND wrap it in ``IBKRExecAdapter`` with
    ``dry_run=True`` even though the always-flat strategy never enters."""
    module = _load_script_module()

    captured: dict = {}

    def _fake_adapter(**kwargs):
        captured["adapter_kwargs"] = kwargs
        return SimpleNamespace(
            connect=lambda: None,
            disconnect=lambda: None,
            is_connected=True,
            _ib=MagicMock(),
        )

    def _fake_exec(**kwargs):
        captured["exec_kwargs"] = kwargs
        return MagicMock(name="IBKRExecAdapter")

    monkeypatch.setattr(module, "IBKRAdapter", _fake_adapter)
    monkeypatch.setattr(module, "IBKRExecAdapter", _fake_exec)

    cfg = module.DryrunConfig(
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_client_id=1,
        ibkr_account="",
        metrics_port=8000,
        es_expiry="20260618",
        duration_seconds=0,
    )
    adapter, exec_adapter = module.build_adapters(cfg)

    assert captured["adapter_kwargs"]["host"] == "127.0.0.1"
    assert captured["adapter_kwargs"]["port"] == 4002
    assert captured["adapter_kwargs"]["client_id"] == 1
    assert captured["adapter_kwargs"]["read_only"] is True

    assert captured["exec_kwargs"]["dry_run"] is True
    # mode must be PAPER (ExecMode.PAPER); no live locks engaged in dry-run.
    from alpha_assay.exec.ibkr import ExecMode

    assert captured["exec_kwargs"]["mode"] is ExecMode.PAPER


def test_script_main_module_docstring_mentions_always_flat():
    """Smoke check the documented contract surface of the script."""
    module = _load_script_module()

    doc = module.__doc__ or ""
    assert "always-flat" in doc.lower()
    assert "phase v" in doc.lower() or "paper_trader_stub" in doc.lower()
