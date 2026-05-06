#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""breadth-recorder entrypoint.

Reads connection + output configuration from environment variables:

    OUT_DIR              Parquet output root (default: /data/breadth)
    METRICS_PORT         Prometheus HTTP port (default: 8001)
    LOG_LEVEL            Python logging level name (default: INFO)
    BREADTH_SYMBOLS      Comma-separated list (default: TICK-NYSE,AD-NYSE)

Bus-consumer mode (preferred, reads from Redis Streams):
    BUS_REDIS_URL        Redis URL, e.g. redis://localhost:6379/0
                         When set, one recorder process per BREADTH_SYMBOL is
                         started; IBKR connection params are ignored.
    BUS_CONSUMER_ID      Consumer identity label (default: breadth-recorder)

Direct-IBKR mode (fallback, used when BUS_REDIS_URL is unset):
    IBKR_HOST            TWS / Gateway host (default: 127.0.0.1)
    IBKR_PORT            TWS paper 7497, IB Gateway paper 4002 (default 7497)
    IBKR_CLIENT_ID       Unique per-connection client id (default: 21)

In bus-consumer mode when multiple symbols are configured, one asyncio.Task
per symbol is launched so a single process covers the full breadth set.

Starts the Prometheus exporter, constructs the recorder, and runs until
SIGTERM/SIGINT. No CLI framework; invoked directly by the Docker image.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import redis as redis_pkg
from prometheus_client import start_http_server

from alpha_assay.data.ibkr_adapter import IBKRAdapter
from infra.recorders.ibkr_breadth.recorder import BreadthRecorder


def _parse_symbols(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("TICK-NYSE", "AD-NYSE")
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("alpha_assay.breadth_recorder")

    out_dir = Path(os.environ.get("OUT_DIR", "/data/breadth"))
    metrics_port = int(os.environ.get("METRICS_PORT", "8001"))
    symbols = _parse_symbols(os.environ.get("BREADTH_SYMBOLS"))
    bus_redis_url = os.environ.get("BUS_REDIS_URL")
    bus_consumer_id = os.environ.get("BUS_CONSUMER_ID", "breadth-recorder")

    # Start Prometheus HTTP exporter before the recorder loop so scrapes
    # succeed immediately.
    start_http_server(metrics_port)

    if bus_redis_url:
        log.info(
            "breadth-recorder starting in bus-consumer mode "
            "(out_dir=%s metrics_port=%d symbols=%s bus=%s)",
            out_dir,
            metrics_port,
            symbols,
            bus_redis_url,
        )
        redis_client = redis_pkg.from_url(bus_redis_url)

        async def _run_all() -> None:
            tasks = [
                asyncio.create_task(
                    BreadthRecorder(
                        out_dir=out_dir,
                        symbol=sym,
                        bus_redis=redis_client,
                        bus_consumer_id=f"{bus_consumer_id}-{sym.lower()}",
                    ).run()
                )
                for sym in symbols
            ]
            await asyncio.gather(*tasks)

        asyncio.run(_run_all())
    else:
        ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
        ibkr_port = int(os.environ.get("IBKR_PORT", "7497"))
        client_id = int(os.environ.get("IBKR_CLIENT_ID", "21"))

        log.info(
            "breadth-recorder starting in direct-IBKR mode "
            "(out_dir=%s metrics_port=%d symbols=%s ibkr=%s:%d)",
            out_dir,
            metrics_port,
            symbols,
            ibkr_host,
            ibkr_port,
        )

        adapter = IBKRAdapter(
            host=ibkr_host,
            port=ibkr_port,
            client_id=client_id,
            read_only=True,
        )
        recorder = BreadthRecorder(adapter=adapter, out_dir=out_dir, symbols=symbols)
        asyncio.run(recorder.run())


if __name__ == "__main__":
    main()
