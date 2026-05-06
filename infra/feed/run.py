# SPDX-License-Identifier: Apache-2.0
"""ibkr-feed daemon entrypoint.

Reads MANIFEST_PATH (default /app/configs/feed-manifest.yaml), connects
to Redis at REDIS_URL, IB Gateway at IBKR_HOST:IBKR_PORT, runs all
subscriptions concurrently. Exposes Prometheus metrics on
METRICS_PORT (default 8003).

Exits non-zero on FeedLockHeldError or fatal Redis errors so the
container restart policy kicks in (docker compose up after operator
fixes the underlying issue).
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

from alpha_assay.data.ibkr_adapter import IBKRAdapter
from infra.feed.feed import FeedManifest, IBKRFeedDaemon

LOG = logging.getLogger("alpha_assay.ibkr_feed")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


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
    await adapter.connect_async()
    LOG.info("IBKR connected")

    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=wal_dir)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    tasks = [
        asyncio.create_task(daemon.run_subscription(sub), name=f"sub-{sub.kind}-{i}")
        for i, sub in enumerate(manifest.subscriptions)
    ]
    stop_task = asyncio.create_task(stop.wait(), name="stop")

    done, pending = await asyncio.wait(
        [*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED
    )
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    for t in done:
        if t is stop_task:
            continue
        exc = t.exception()
        if exc is not None:
            LOG.error("subscription task failed: %s", exc, exc_info=exc)
            return 2
    return 0


def main() -> int:
    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
