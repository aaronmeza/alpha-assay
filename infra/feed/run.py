# SPDX-License-Identifier: Apache-2.0
"""ibkr-feed daemon entrypoint.

Reads MANIFEST_PATH (default /app/configs/feed-manifest.yaml), connects
to Redis at REDIS_URL, IB Gateway at IBKR_HOST:IBKR_PORT, runs all
subscriptions concurrently. Exposes Prometheus metrics on
METRICS_PORT (default 8003).

A connection watchdog runs alongside the subscriptions: IB Gateway
performs a nightly IBC restart (~22:50 CT) and the socket also drops on
the occasional network blip. When that happens the ib_insync event
stream simply goes quiet - the subscription coroutines would otherwise
block forever in ``await queue.get()`` with the process still "Up" and
no data flowing (the failure that silently halted collection for ~6
days after the Phase T deploy). The watchdog notices
``adapter.is_connected`` flip to False, marks the feed disconnected,
and the daemon exits non-zero so the container restart policy
(``restart: unless-stopped``) recycles it with a fresh connection.

Exits non-zero on connection loss, FeedLockHeldError, a subscription
fault, or fatal Redis errors so the container restart policy kicks in.
A clean SIGTERM / SIGINT exits zero.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import redis
from prometheus_client import start_http_server

from alpha_assay.data.front_month import write_front_month
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from alpha_assay.observability import metrics as M
from infra.feed.feed import FeedManifest, IBKRFeedDaemon, Subscription

LOG = logging.getLogger("alpha_assay.ibkr_feed")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Docker `restart: unless-stopped` recycles the container on any non-zero exit.
EXIT_OK = 0
EXIT_RESTART = 2


async def _watch_connection(adapter, poll_seconds: float = 5.0) -> None:
    """Block until the IBKR connection is lost.

    Polls ``adapter.is_connected``; returns as soon as it reads False.
    Cheap (one bool check per ``poll_seconds``) and side-effect free -
    the caller decides what to do with the news.
    """
    while adapter.is_connected:
        await asyncio.sleep(poll_seconds)


async def _connect_with_retry(
    adapter,
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> None:
    """Connect to IB Gateway, retrying with exponential backoff.

    A freshly (re)started container can land mid-IBC-cycle when IB
    Gateway is briefly down. Rather than exit immediately and lean
    entirely on Docker's restart loop, give the gateway a short window
    to come back. If every attempt fails the last exception is
    re-raised - the process then exits and Docker restarts it with its
    own backoff.
    """
    for i in range(attempts):
        try:
            await adapter.connect_async()
            return
        except Exception as exc:  # noqa: BLE001 - retry on any connect failure
            if i == attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2**i))
            LOG.warning(
                "IBKR connect attempt %d/%d failed (%s); retrying in %.0fs",
                i + 1,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


def _mark_disconnected(adapter) -> None:
    """Reflect a lost connection in the metrics + the adapter state.

    ``adapter.disconnect()`` is a no-op once ``isConnected()`` is already
    False (the peer closed the socket for us), so it would not move the
    gauge or the counter. Set them explicitly, then call disconnect()
    anyway for any local teardown it still wants to do.
    """
    M.ibkr_connected.set(0)
    M.ibkr_connection_events_total.labels(event="disconnected").inc()
    try:
        adapter.disconnect()
    except Exception:  # noqa: BLE001 - teardown is best-effort
        LOG.debug("adapter.disconnect() raised during teardown", exc_info=True)


async def _run_subscriptions(
    daemon: IBKRFeedDaemon,
    adapter,
    subscriptions,
    stop_event: asyncio.Event,
    *,
    watchdog_poll: float = 5.0,
) -> int:
    """Run all subscriptions plus the connection watchdog until one ends.

    Returns ``EXIT_OK`` only on a requested stop (SIGTERM/SIGINT).
    Returns ``EXIT_RESTART`` on connection loss, a subscription raising,
    or a subscription stream ending unexpectedly - in every one of those
    cases the right move is "exit and let the supervisor bring us back".
    """
    sub_tasks = [
        asyncio.create_task(daemon.run_subscription(sub), name=f"sub-{sub.kind}-{i}")
        for i, sub in enumerate(subscriptions)
    ]
    wd_task = asyncio.create_task(_watch_connection(adapter, poll_seconds=watchdog_poll), name="watchdog")
    stop_task = asyncio.create_task(stop_event.wait(), name="stop")

    done, pending = await asyncio.wait([*sub_tasks, wd_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if stop_task in done:
        LOG.info("stop requested; shutting down ibkr-feed cleanly")
        return EXIT_OK

    if wd_task in done:
        LOG.error("IBKR connection lost; exiting so the container restart policy reconnects")
        _mark_disconnected(adapter)
        return EXIT_RESTART

    # A subscription task finished. Surface why, then exit for a restart.
    for t in done:
        if t is stop_task or t is wd_task:
            continue
        exc = t.exception()
        if exc is not None:
            LOG.error("subscription task %s failed: %s", t.get_name(), exc, exc_info=exc)
        else:
            LOG.error("subscription task %s ended unexpectedly (stream closed?)", t.get_name())
    _mark_disconnected(adapter)
    return EXIT_RESTART


async def _resolve_or_pin_es(
    manifest: FeedManifest,
    adapter,
    redis_client,
) -> list[Subscription]:
    """Resolve (or pin) the ES front-month expiry and return a patched subscription list.

    Three paths:

    1. **Skip** - no ES bars subscriptions in the manifest. Logs and returns
       ``manifest.subscriptions`` unchanged. No IBKR call, no Redis write, no
       gauge update. Appropriate for breadth-only deployments.

    2. **Override** - ``ES_EXPIRY`` env var is set. Uses the operator-pinned
       value; skips the ContFuture IBKR call (useful during rollover or when
       the ContFuture qualify would fail).

    3. **Auto-resolve** - ``ES_EXPIRY`` is unset. Calls
       ``adapter.resolve_front_month_future`` once; the returned
       ``lastTradeDateOrContractMonth`` becomes the expiry. Any exception
       propagates to the caller (daemon exits and the container restart policy
       reconnects).

    In paths 2 and 3 the resolved expiry is written to Redis via
    ``write_front_month`` and exposed as ``alpha_assay_front_month_expiry``
    so consumers learn which stream is live without their own IBKR call.
    """
    es_sub_count = sum(
        1
        for s in manifest.subscriptions
        if s.kind == "bars"
        and s.contract is not None
        and s.contract.get("symbol") == "ES"
        and s.contract.get("exchange") == "CME"
    )
    if es_sub_count == 0:
        LOG.info("no ES bars subscriptions in manifest; skipping ContFuture resolve")
        return list(manifest.subscriptions)

    # Resolve (or pin) the ES front-month expiry.
    es_expiry_override = os.environ.get("ES_EXPIRY", "").strip()
    if es_expiry_override:
        # Emergency override: pin to the operator-specified contract.
        LOG.info("ES_EXPIRY override set to %s; skipping ContFuture", es_expiry_override)
        resolved_expiry = es_expiry_override
    else:
        # Auto-resolve via ContFuture. Exception propagates and exits the
        # daemon (watchdog restarts the container) - no error swallowing.
        fut = await adapter.resolve_front_month_future(symbol="ES", exchange="CME", currency="USD")
        resolved_expiry = fut.lastTradeDateOrContractMonth
        LOG.info(
            "ContFuture resolved ES@CME -> %s (%s)",
            resolved_expiry,
            fut.localSymbol,
        )

    # Patch any bars subscription for ES@CME to use the resolved expiry,
    # overriding whatever is hardcoded in the manifest file.
    patched_subs = []
    for sub in manifest.subscriptions:
        if (
            sub.kind == "bars"
            and sub.contract is not None
            and sub.contract.get("symbol") == "ES"
            and sub.contract.get("exchange") == "CME"
        ):
            patched_contract = {**sub.contract, "expiry": resolved_expiry}
            patched_subs.append(
                Subscription(
                    kind=sub.kind,
                    contract=patched_contract,
                    bar_size=sub.bar_size,
                    what_to_show=sub.what_to_show,
                )
            )
        else:
            patched_subs.append(sub)

    # Publish to Redis so consumers learn which stream is live.
    write_front_month(redis_client, symbol="ES", exchange="CME", expiry=resolved_expiry)

    # Expose as a Prometheus label for observability.
    M.front_month_expiry.labels(symbol="ES", exchange="CME").set(int(resolved_expiry))

    return patched_subs


async def _main() -> int:
    manifest_path = Path(os.environ.get("MANIFEST_PATH", "/app/configs/feed-manifest.yaml"))
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    metrics_port = int(os.environ.get("METRICS_PORT", "8003"))
    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.environ.get("IBKR_PORT", "4002"))
    client_id = int(os.environ.get("IBKR_CLIENT_ID", "30"))
    wal_dir = Path(os.environ.get("WAL_DIR", "/var/lib/alphaassay/wal"))

    manifest = FeedManifest.from_yaml(manifest_path)
    LOG.info("loaded %d subscriptions from %s", len(manifest.subscriptions), manifest_path)

    adapter = IBKRAdapter(host=ibkr_host, port=ibkr_port, client_id=client_id)
    redis_client = redis.from_url(redis_url)
    redis_client.ping()  # fail fast if unreachable

    start_http_server(metrics_port)
    LOG.info("metrics listening on :%d", metrics_port)

    # Connect ONCE before launching subscription tasks. Each subscription
    # task otherwise races to call connectAsync simultaneously, and IBKR
    # rejects all but one of the duplicate connection attempts.
    LOG.info("connecting to IBKR at %s:%d clientId=%d", ibkr_host, ibkr_port, client_id)
    await _connect_with_retry(adapter)
    LOG.info("IBKR connected")

    patched_subs = await _resolve_or_pin_es(manifest, adapter, redis_client)

    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=wal_dir)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    return await _run_subscriptions(daemon, adapter, patched_subs, stop)


def main() -> int:
    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception:  # noqa: BLE001 - last resort: log + non-zero so the container restarts
        LOG.exception("ibkr-feed crashed; exiting non-zero for supervisor restart")
        return EXIT_RESTART


if __name__ == "__main__":
    sys.exit(main())
