# SPDX-License-Identifier: Apache-2.0
"""paper_dryrun reads bars from the bus."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import fakeredis
import pandas as pd

from alpha_assay.bus.consumer import Consumer
from alpha_assay.bus.producer import Producer
from alpha_assay.bus.streams import stream_name_for_bars
from alpha_assay.exec.trade_log import TradeLog

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "paper_dryrun.py"
_MODULE_NAME = "paper_dryrun"


def _load_script_module():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def test_paper_dryrun_consumer_construction_works(tmp_path: Path):
    """Smoke: paper-trader-style Consumer can be constructed and reads from bus."""
    redis_client = fakeredis.FakeRedis()
    p = Producer(redis_client=redis_client)
    payload = {
        "open": 7250.5,
        "high": 7252.5,
        "low": 7246.5,
        "close": 7247.25,
        "volume": 14338,
        "ts_minute_utc": int(
            pd.Timestamp("2026-05-06T13:30:00+00:00").value // 10**9
        ),
    }
    # Publish BEFORE constructing consumer (consumer uses start_id="0" for test ergonomics
    # since fakeredis blocking semantics are tricky).
    p.publish(
        "bars.es.cme.20260618",
        payload,
        ts_event_ns=pd.Timestamp("2026-05-06T13:30:00+00:00").value,
    )

    spec = {"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"}
    c = Consumer(
        redis_client=redis_client,
        stream=stream_name_for_bars(spec),
        consumer_id="paper-trader-test",
        start_id="0",
    )
    msgs = list(c.iter_messages(max_messages=1))
    assert len(msgs) == 1
    assert msgs[0].payload["close"] == 7247.25


def test_build_bus_consumers_returns_bars_and_breadth_consumers():
    """_build_bus_consumers returns (bars_consumer, breadth_consumer)
    pointing at the correct streams."""
    module = _load_script_module()
    redis_client = fakeredis.FakeRedis()

    cfg = module.DryrunConfig(
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_client_id=1,
        ibkr_account="",
        metrics_port=8000,
        es_expiry="20260618",
        duration_seconds=0,
        bus_redis_url="redis://localhost:6379/0",
    )
    bars_c, breadth_c = module._build_bus_consumers(cfg, redis_client)

    assert bars_c._stream == "bars.es.cme.20260618"
    assert breadth_c._stream == "ticks.tick-nyse"
    # Both start from $ (latest-only for paper-trader).
    assert bars_c._cursor == "$"
    assert breadth_c._cursor == "$"


def test_load_config_bus_fields_from_env(monkeypatch):
    """BUS_REDIS_URL and RUNS_DIR are resolved from env."""
    module = _load_script_module()

    monkeypatch.setenv("BUS_REDIS_URL", "redis://bus-host:6379/1")
    monkeypatch.setenv("RUNS_DIR", "/tmp/my-runs")

    cfg = module.load_config_from_env()
    assert cfg.bus_redis_url == "redis://bus-host:6379/1"
    assert cfg.runs_dir == "/tmp/my-runs"


def test_load_config_bus_defaults(monkeypatch):
    """Without BUS_REDIS_URL the default is empty (direct-IBKR mode)."""
    module = _load_script_module()

    monkeypatch.delenv("BUS_REDIS_URL", raising=False)
    monkeypatch.delenv("RUNS_DIR", raising=False)

    cfg = module.load_config_from_env()
    assert cfg.bus_redis_url == ""
    assert cfg.runs_dir == module.DEFAULT_RUNS_DIR


def test_strategy_trade_log_attached(tmp_path: Path):
    """AlwaysFlatStrategy accepts trade_log kwarg; the instance is stored."""
    module = _load_script_module()

    trade_log = TradeLog(out_dir=tmp_path / "paper-live")
    strategy = module.AlwaysFlatStrategy(exec_adapter=object(), trade_log=trade_log)

    assert strategy._trade_log is trade_log


def test_strategy_maybe_emit_trade_noop_when_signal_zero(tmp_path: Path):
    """_maybe_emit_trade is a no-op when signal is 0 (always-flat guard)."""
    module = _load_script_module()

    trade_log = TradeLog(out_dir=tmp_path / "paper-live")
    strategy = module.AlwaysFlatStrategy(exec_adapter=object(), trade_log=trade_log)

    bar = {
        "timestamp": "2026-05-06T13:30:00+00:00",
        "open": 7250.0,
        "high": 7252.0,
        "low": 7248.0,
        "close": 7250.0,
        "volume": 100,
    }
    strategy._maybe_emit_trade(bar, signal=0)
    trade_log.flush()

    # No file written because no trade.
    parquet_path = tmp_path / "paper-live" / "trades.parquet"
    assert not parquet_path.exists()


def test_strategy_on_bar_increments_counter_in_bus_mode(tmp_path: Path):
    """on_bar works normally regardless of bus vs direct mode."""
    module = _load_script_module()

    trade_log = TradeLog(out_dir=tmp_path / "paper-live")
    strategy = module.AlwaysFlatStrategy(exec_adapter=object(), trade_log=trade_log)

    bar = {
        "timestamp": "2026-05-06T13:30:00+00:00",
        "open": 7250.0,
        "high": 7252.0,
        "low": 7248.0,
        "close": 7250.0,
        "volume": 100,
    }
    strategy.on_bar(bar, feed_label="es")
    assert strategy.bars_seen == 1
