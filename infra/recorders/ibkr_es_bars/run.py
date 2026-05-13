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
    ES_EXPIRY            Emergency override for the front-month contract code,
                         ``YYYYMMDD``. When set, this value is used directly and
                         Redis is not consulted. When unset (normal operation),
                         the resolved expiry is read from the Redis metadata key
                         written by ibkr-feed at startup. If neither is available
                         at runtime the process fails closed with
                         ``FrontMonthMissingError``.
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

import redis as redis_pkg
from prometheus_client import start_http_server

from alpha_assay.data.front_month import (
    FrontMonthMissingError,
    read_front_month,
    validate_yyyymmdd,
)
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from infra.recorders.ibkr_es_bars.recorder import ESBarsRecorder

# Default client id is offset from the breadth recorder's 21 and the
# paper-trader's 1 so all three services can share an IB Gateway.
_DEFAULT_CLIENT_ID = 22


def _resolve_es_expiry_with_wait(
    redis_client: redis_pkg.Redis | None,
    *,
    max_wait_seconds: float = 60.0,
    poll_interval_seconds: float = 5.0,
) -> tuple[str, str]:
    """Resolve ES expiry, polling Redis for up to max_wait_seconds.

    Env-override path is checked first (instant, validated). If env is
    unset and Redis is provided, poll the key for up to max_wait_seconds
    before giving up. Handles the cold-start race where the consumer boots
    before ibkr-feed has written the metadata key.

    Returns:
        Tuple of (expiry_value, source_label). Log source for operator
        traceability.

    Raises:
        InvalidExpiryError: env override is set but not a valid YYYYMMDD.
        FrontMonthMissingError: neither env nor Redis key available after
            max_wait_seconds.
    """
    import time

    override = os.environ.get("ES_EXPIRY", "").strip()
    if override:
        # Validate before using - bad env value must fail loudly.
        validated = validate_yyyymmdd(override, source="ES_EXPIRY env")
        return validated, "ES_EXPIRY env"
    if redis_client is None:
        raise FrontMonthMissingError(
            "ES_EXPIRY env var is unset and no Redis client - cannot resolve "
            "ES front-month. Set ES_EXPIRY to YYYYMMDD or ensure BUS_REDIS_URL "
            "is set and ibkr-feed has published the front-month metadata key."
        )
    deadline = time.monotonic() + max_wait_seconds
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            # read_front_month validates the stored value internally (Fix A).
            value = read_front_month(redis_client, symbol="ES", exchange="CME")
            return value, "Redis metadata key"
        except FrontMonthMissingError as e:
            last_err = e
            time.sleep(poll_interval_seconds)
    raise FrontMonthMissingError(
        f"ES front-month not in Redis after {max_wait_seconds:.0f}s wait; "
        f"ibkr-feed may not have started or failed to resolve. Last error: {last_err}"
    )


def _resolve_es_expiry(
    redis_client: redis_pkg.Redis | None,
) -> tuple[str, str]:
    """Resolve the ES expiry for this recorder run.

    Delegates to _resolve_es_expiry_with_wait with default timeouts.
    Kept for backward-compat with callers that use this name directly.
    """
    return _resolve_es_expiry_with_wait(redis_client)


def _build_contract_spec(expiry: str) -> dict[str, Any]:
    return {
        "symbol": os.environ.get("ES_SYMBOL", "ES"),
        "sec_type": "FUT",
        "exchange": os.environ.get("ES_EXCHANGE", "CME"),
        "currency": os.environ.get("ES_CURRENCY", "USD"),
        "expiry": expiry,
    }


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("alpha_assay.es_bars_recorder")

    out_dir = Path(os.environ.get("OUT_DIR", "/data/es_bars"))
    metrics_port = int(os.environ.get("METRICS_PORT", "8002"))
    bus_redis_url = os.environ.get("BUS_REDIS_URL", "")

    # Start Prometheus HTTP exporter before the recorder loop so scrapes
    # succeed immediately.
    start_http_server(metrics_port)

    if bus_redis_url:
        # Bus-consumer mode: read bars from Redis Stream produced by the ibkr-feed daemon.
        # Resolve the front-month expiry now that we have a Redis connection available.
        bus_redis = redis_pkg.from_url(bus_redis_url)
        expiry, source = _resolve_es_expiry(bus_redis)
        log.info("resolved ES expiry from %s: %s", source, expiry)
        contract_spec = _build_contract_spec(expiry)
        log.info(
            "es-bars-recorder starting in bus-consumer mode (out_dir=%s metrics_port=%d bus=%s contract=%s)",
            out_dir,
            metrics_port,
            bus_redis_url,
            contract_spec,
        )
        recorder = ESBarsRecorder(
            out_dir=out_dir,
            contract_spec=contract_spec,
            bus_redis=bus_redis,
            bus_consumer_id=os.environ.get("BUS_CONSUMER_ID", "es-bars-recorder"),
        )
    else:
        # Direct-IBKR mode (legacy): recorder subscribes directly to IB Gateway.
        # ES_EXPIRY env var must be set when running without a bus Redis connection;
        # _resolve_es_expiry raises FrontMonthMissingError (fails closed) if neither is available.
        expiry, source = _resolve_es_expiry(None)
        log.info("resolved ES expiry from %s: %s", source, expiry)
        contract_spec = _build_contract_spec(expiry)
        ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
        ibkr_port = int(os.environ.get("IBKR_PORT", "4002"))
        client_id = int(os.environ.get("IBKR_CLIENT_ID", str(_DEFAULT_CLIENT_ID)))
        account = os.environ.get("IBKR_ACCOUNT", "")
        log.info(
            "es-bars-recorder starting in direct-IBKR mode (out_dir=%s metrics_port=%d ibkr=%s:%d "
            "client_id=%d contract=%s)",
            out_dir,
            metrics_port,
            ibkr_host,
            ibkr_port,
            client_id,
            contract_spec,
        )
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
