# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""ES bars recorder consuming from bus produces the same parquet shard
as the old direct-IBKR path for identical input bars."""

from __future__ import annotations

import asyncio
from pathlib import Path

import fakeredis
import pandas as pd
import pytest

from alpha_assay.bus.producer import Producer
from infra.recorders.ibkr_es_bars.recorder import ESBarsRecorder


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


def _bar(ts_iso: str, c: float = 100.0, v: int = 1) -> dict:
    return {
        "timestamp": pd.Timestamp(ts_iso).value // 10**9,
        "open": c,
        "high": c,
        "low": c,
        "close": c,
        "volume": v,
        "ts_minute_utc": pd.Timestamp(ts_iso).value // 10**9,
    }


def test_recorder_consumes_from_bus_and_writes_parquet(redis_client, tmp_path: Path):
    # 3 RTH bars on the bus.
    p = Producer(redis_client=redis_client)
    for ts in (
        "2026-05-06T13:30:00+00:00",
        "2026-05-06T13:31:00+00:00",
        "2026-05-06T13:32:00+00:00",
    ):
        p.publish(
            "bars.es.cme.20260618",
            _bar(ts),
            ts_event_ns=pd.Timestamp(ts).value,
        )

    rec = ESBarsRecorder(
        out_dir=tmp_path,
        contract_spec={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"},
        bus_redis=redis_client,
        bus_consumer_id="es-bars-test",
    )

    async def _drive():
        # Consume up to 3 messages then stop.
        await rec.consume_n_messages_for_test(3)
        rec.flush()

    asyncio.run(_drive())

    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    # All 3 bars persisted, monotonic, with the canonical schema.
    assert len(df) == 3
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].is_monotonic_increasing


def test_recorder_bus_mode_drops_out_of_rth_bars(redis_client, tmp_path: Path):
    """Bus-consumer mode must respect the same RTH filter as direct mode."""
    p = Producer(redis_client=redis_client)
    # Pre-RTH bar (07:00 CT = 12:00 UTC) - should be dropped.
    p.publish(
        "bars.es.cme.20260618",
        _bar("2026-05-06T12:00:00+00:00"),
        ts_event_ns=pd.Timestamp("2026-05-06T12:00:00+00:00").value,
    )
    # In-RTH bar (09:31 CT = 14:31 UTC) - should be kept.
    p.publish(
        "bars.es.cme.20260618",
        _bar("2026-05-06T14:31:00+00:00"),
        ts_event_ns=pd.Timestamp("2026-05-06T14:31:00+00:00").value,
    )

    rec = ESBarsRecorder(
        out_dir=tmp_path,
        contract_spec={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"},
        bus_redis=redis_client,
        bus_consumer_id="es-bars-rth-test",
    )

    async def _drive():
        await rec.consume_n_messages_for_test(2)
        rec.flush()

    asyncio.run(_drive())

    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert len(df) == 1  # only the in-RTH bar


def test_recorder_direct_mode_still_works_when_no_bus_redis(tmp_path: Path):
    """Passing bus_redis=None must leave _consumer as None (direct-IBKR path unaffected)."""
    from unittest.mock import MagicMock

    from alpha_assay.data.ibkr_adapter import IBKRAdapter

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = False
    adapter = IBKRAdapter(ib=ib)

    rec = ESBarsRecorder(
        adapter=adapter,
        out_dir=tmp_path,
    )
    assert rec._consumer is None


def test_recorder_bus_mode_schema_parity_with_direct_mode(redis_client, tmp_path: Path):
    """Parquet output from bus-consumer mode must be byte-schema-equivalent
    to the direct-IBKR mode: same column order, same dtypes, monotonic timestamps.
    """
    from alpha_assay.data.databento_adapter import load_parquet

    p = Producer(redis_client=redis_client)
    rth_bars = [
        ("2026-05-06T13:30:00+00:00", 5000.0, 5010.0, 4990.0, 5005.0, 100),
        ("2026-05-06T13:31:00+00:00", 5005.0, 5015.0, 4995.0, 5008.0, 150),
        ("2026-05-06T13:32:00+00:00", 5008.0, 5020.0, 5000.0, 5012.0, 200),
    ]
    for ts, o, h, lo, c, v in rth_bars:
        p.publish(
            "bars.es.cme.20260618",
            {
                "timestamp": pd.Timestamp(ts).value // 10**9,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": v,
                "ts_minute_utc": pd.Timestamp(ts).value // 10**9,
            },
            ts_event_ns=pd.Timestamp(ts).value,
        )

    rec = ESBarsRecorder(
        out_dir=tmp_path,
        contract_spec={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"},
        bus_redis=redis_client,
        bus_consumer_id="es-bars-parity-test",
    )

    async def _drive():
        await rec.consume_n_messages_for_test(3)
        rec.flush()

    asyncio.run(_drive())

    # Must round-trip through the same load_parquet validator used by the strategy.
    df = load_parquet(tmp_path / "2026-05-06.parquet")
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "America/Chicago"
    assert df.index.is_monotonic_increasing
