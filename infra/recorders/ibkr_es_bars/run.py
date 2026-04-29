#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""ES-bars recorder entrypoint.

Reads connection + output configuration from environment variables:

    OUT_DIR              Parquet output root (default: /data/es_bars)
    METRICS_PORT         Prometheus HTTP port (default: 8002)
    IBKR_HOST            TWS / Gateway host (default: 127.0.0.1)
    IBKR_PORT            TWS paper 7497, IB Gateway paper 4002 (default 4002)
    IBKR_CLIENT_ID       Unique per-connection client id (default: 22)
    IBKR_ACCOUNT         Optional account code; empty -> default account
    ES_SYMBOL            Futures root (default: ES)
    ES_EXCHANGE          Futures exchange (default: CME)
    ES_CURRENCY          Quote currency (default: USD)
    ES_EXPIRY            ``YYYYMMDD`` last-trade-date (default: 20260618).
                         Matches ``scripts/paper_dryrun.py``'s hardcoded
                         fallback. The short YYYYMM form is rejected by
                         IBKR with "No security definition has been found".
                         Update on each quarterly roll.
    LOG_LEVEL            Python logging level name (default: INFO)

Starts the Prometheus exporter, constructs the IBKR adapter + recorder,
and runs until SIGTERM/SIGINT. No CLI framework; invoked directly by the Docker image.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from prometheus_client import start_http_server

from alpha_assay.data.ibkr_adapter import IBKRAdapter
from infra.recorders.ibkr_es_bars.recorder import ESBarsRecorder

# Default client id is offset from the breadth recorder's 21 and the
# paper-trader's 1 so all three services can share an IB Gateway.
_DEFAULT_CLIENT_ID = 22
_DEFAULT_ES_EXPIRY = "20260618"


def _build_contract_spec() -> dict[str, Any]:
    return {
        "symbol": os.environ.get("ES_SYMBOL", "ES"),
        "sec_type": "FUT",
        "exchange": os.environ.get("ES_EXCHANGE", "CME"),
        "currency": os.environ.get("ES_CURRENCY", "USD"),
        "expiry": os.environ.get("ES_EXPIRY", _DEFAULT_ES_EXPIRY),
    }


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("alpha_assay.es_bars_recorder")

    out_dir = Path(os.environ.get("OUT_DIR", "/data/es_bars"))
    metrics_port = int(os.environ.get("METRICS_PORT", "8002"))
    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.environ.get("IBKR_PORT", "4002"))
    client_id = int(os.environ.get("IBKR_CLIENT_ID", str(_DEFAULT_CLIENT_ID)))
    account = os.environ.get("IBKR_ACCOUNT", "")
    contract_spec = _build_contract_spec()

    log.info(
        "es-bars-recorder starting (out_dir=%s metrics_port=%d ibkr=%s:%d " "client_id=%d contract=%s)",
        out_dir,
        metrics_port,
        ibkr_host,
        ibkr_port,
        client_id,
        contract_spec,
    )

    if contract_spec["expiry"] == _DEFAULT_ES_EXPIRY:
        log.warning(
            "ES_EXPIRY not set; using hardcoded fallback %s. Update on quarterly roll.",
            _DEFAULT_ES_EXPIRY,
        )

    # Start Prometheus HTTP exporter before the recorder loop so scrapes
    # succeed immediately.
    start_http_server(metrics_port)

    adapter = IBKRAdapter(
        host=ibkr_host,
        port=ibkr_port,
        client_id=client_id,
        account=account or None,
        read_only=True,
    )
    recorder = ESBarsRecorder(
        adapter=adapter,
        out_dir=out_dir,
        contract_spec=contract_spec,
    )

    asyncio.run(recorder.run())


if __name__ == "__main__":
    main()
