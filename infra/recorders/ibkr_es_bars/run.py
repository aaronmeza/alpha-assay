#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Futures-bars recorder entrypoint (ES by default; any futures root via env).

Reads connection + output configuration from environment variables:

    OUT_DIR              Parquet output root (default: /data/es_bars)
    METRICS_PORT         Prometheus HTTP port (default: 8002)
    IBKR_HOST            TWS / Gateway host (default: 127.0.0.1)
    IBKR_PORT            TWS paper 7497, IB Gateway paper 4002 (default 4002)
    IBKR_CLIENT_ID       Unique per-connection client id (default: 22)
    IBKR_ACCOUNT         Optional account code; empty -> default account
    BARS_SYMBOL          Futures root (default: ES). Generic alias; takes
                         precedence over the legacy ES_SYMBOL.
    BARS_EXCHANGE        Futures exchange (default: CME); alias of ES_EXCHANGE.
    BARS_CURRENCY        Quote currency (default: USD); alias of ES_CURRENCY.
    BARS_EXPIRY          Emergency override for the front-month contract code,
                         ``YYYYMMDD``. When set, this value is used directly and
                         Redis is not consulted. When unset (normal operation),
                         the resolved expiry is read from the per-root Redis
                         metadata key written by ibkr-feed at startup. If
                         neither is available at runtime the process fails
                         closed with ``FrontMonthMissingError``.
    ES_SYMBOL            Legacy aliases of the BARS_* vars above, honoured for
    ES_EXCHANGE          back-compat with existing deployments. ES_EXPIRY is
    ES_CURRENCY          honoured ONLY when the resolved symbol is ES, so a
    ES_EXPIRY            stack-wide ES emergency pin cannot mis-pin a second
                         recorder instance covering another root (e.g. NQ).
    LOG_LEVEL            Python logging level name (default: INFO)

A second instance of this recorder with ``BARS_SYMBOL=NQ`` and
``OUT_DIR=/data/nq_bars`` records NQ front-month bars; the parquet shard
layout and metric names are identical (series are distinguished by the
``feed`` label and the scrape job).

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

from alpha_assay.data.front_month import read_front_month_with_wait
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from infra.recorders.ibkr_es_bars.recorder import ESBarsRecorder

# Default client id is offset from the breadth recorder's 21 and the
# paper-trader's 1 so all three services can share an IB Gateway.
_DEFAULT_CLIENT_ID = 22


def _env_first(*names: str, default: str) -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _configured_root() -> tuple[str, str, str]:
    """Resolve the (symbol, exchange, currency) futures root from env.

    Generic ``BARS_*`` vars take precedence; legacy ``ES_*`` vars are
    honoured for back-compat with existing deployments.
    """
    symbol = _env_first("BARS_SYMBOL", "ES_SYMBOL", default="ES")
    exchange = _env_first("BARS_EXCHANGE", "ES_EXCHANGE", default="CME")
    currency = _env_first("BARS_CURRENCY", "ES_CURRENCY", default="USD")
    return symbol, exchange, currency


def _expiry_override(symbol: str) -> tuple[str, str]:
    """Return ``(value, env_var_name)`` for the operator expiry pin, or ("", "").

    ``BARS_EXPIRY`` always applies (it is set per-service). The legacy
    ``ES_EXPIRY`` applies ONLY when the configured root is ES - it is
    commonly set stack-wide as an emergency pin and must never leak into
    a recorder instance covering a different root.
    """
    override = os.environ.get("BARS_EXPIRY", "").strip()
    if override:
        return override, "BARS_EXPIRY"
    if symbol == "ES":
        override = os.environ.get("ES_EXPIRY", "").strip()
        if override:
            return override, "ES_EXPIRY"
    return "", ""


def _resolve_es_expiry_with_wait(
    redis_client: redis_pkg.Redis | None,
    *,
    symbol: str = "ES",
    exchange: str = "CME",
    max_wait_seconds: float = 60.0,
    poll_interval_seconds: float = 5.0,
) -> tuple[str, str]:
    """Resolve the front-month expiry, polling Redis for up to max_wait_seconds.

    Thin wrapper over the shared
    :func:`alpha_assay.data.front_month.read_front_month_with_wait`: this
    function only owns the recorder's env-pin *policy* (``BARS_EXPIRY``
    always, ``ES_EXPIRY`` ES-only via :func:`_expiry_override`) and hands
    the resolved override plus the (symbol, exchange) to the shared
    resolver. Keeps the historical name + signature for callers/tests.

    Returns:
        Tuple of (expiry_value, source_label). Log source for operator
        traceability.

    Raises:
        InvalidExpiryError: env override is set but not a valid YYYYMMDD.
        FrontMonthMissingError: neither env nor Redis key available after
            max_wait_seconds.
    """
    override, override_var = _expiry_override(symbol)
    return read_front_month_with_wait(
        redis_client,
        symbol=symbol,
        exchange=exchange,
        env_override=override,
        env_override_source=f"{override_var} env" if override_var else "env override",
        max_wait_seconds=max_wait_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def _resolve_es_expiry(
    redis_client: redis_pkg.Redis | None,
    *,
    symbol: str = "ES",
    exchange: str = "CME",
) -> tuple[str, str]:
    """Resolve the front-month expiry for this recorder run.

    Delegates to _resolve_es_expiry_with_wait with default timeouts.
    Kept for backward-compat with callers that use this name directly.
    """
    return _resolve_es_expiry_with_wait(redis_client, symbol=symbol, exchange=exchange)


def _build_contract_spec(expiry: str) -> dict[str, Any]:
    symbol, exchange, currency = _configured_root()
    return {
        "symbol": symbol,
        "sec_type": "FUT",
        "exchange": exchange,
        "currency": currency,
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
    symbol, exchange, _currency = _configured_root()

    # Start Prometheus HTTP exporter before the recorder loop so scrapes
    # succeed immediately.
    start_http_server(metrics_port)

    if bus_redis_url:
        # Bus-consumer mode: read bars from Redis Stream produced by the ibkr-feed daemon.
        # Resolve the front-month expiry now that we have a Redis connection available.
        bus_redis = redis_pkg.from_url(bus_redis_url)
        expiry, source = _resolve_es_expiry(bus_redis, symbol=symbol, exchange=exchange)
        log.info("resolved %s expiry from %s: %s", symbol, source, expiry)
        contract_spec = _build_contract_spec(expiry)
        log.info(
            "bars-recorder starting in bus-consumer mode (out_dir=%s metrics_port=%d bus=%s contract=%s)",
            out_dir,
            metrics_port,
            bus_redis_url,
            contract_spec,
        )
        # An operator expiry pin (BARS_EXPIRY / ES_EXPIRY) freezes the
        # roll-rebind watcher: the operator has taken manual control of the
        # contract, so the recorder must not auto-rebind out from under them.
        override, _override_var = _expiry_override(symbol)
        bus_consumer_id = os.environ.get("BUS_CONSUMER_ID", f"{symbol.lower()}-bars-recorder")
        recorder = ESBarsRecorder(
            out_dir=out_dir,
            contract_spec=contract_spec,
            bus_redis=bus_redis,
            bus_consumer_id=bus_consumer_id,
            front_month_pinned=bool(override),
            service_label=bus_consumer_id,
        )
    else:
        # Direct-IBKR mode (legacy): recorder subscribes directly to IB Gateway.
        # An expiry env override must be set when running without a bus Redis connection;
        # _resolve_es_expiry raises FrontMonthMissingError (fails closed) if neither is available.
        expiry, source = _resolve_es_expiry(None, symbol=symbol, exchange=exchange)
        log.info("resolved %s expiry from %s: %s", symbol, source, expiry)
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
