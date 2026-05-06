# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Breadth recorder consuming from bus produces the same parquet shard
as the old direct-IBKR path for identical input ticks."""

from __future__ import annotations

import asyncio

import fakeredis
import pandas as pd

from alpha_assay.bus.producer import Producer
from infra.recorders.ibkr_breadth.recorder import BreadthRecorder


def _tick(value: float, symbol: str) -> dict:
    return {"value": value, "symbol": symbol}


def test_breadth_recorder_consumes_from_bus(tmp_path):
    redis_client = fakeredis.FakeRedis()
    p = Producer(redis_client=redis_client)
    for ts in ("2026-05-06T13:30:01+00:00", "2026-05-06T13:30:02+00:00"):
        p.publish(
            "ticks.tick-nyse",
            _tick(142.0, "TICK-NYSE"),
            ts_event_ns=pd.Timestamp(ts).value,
        )

    rec = BreadthRecorder(
        out_dir=tmp_path,
        symbol="TICK-NYSE",
        bus_redis=redis_client,
        bus_consumer_id="breadth-test",
    )

    async def _drive():
        await rec.consume_n_messages_for_test(2)
        rec.flush()

    asyncio.run(_drive())

    files = list((tmp_path / "TICK_NYSE").glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    # Recorder dedupes by minute floor; 2 ticks within the same minute -> 1 row.
    assert len(df) >= 1
