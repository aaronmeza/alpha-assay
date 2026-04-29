# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""PLACEHOLDER paper-trader entrypoint.

This module exists solely so the compose stack has a live
/metrics scrape target while the real paper-trader is built in .
It:

- Starts the Prometheus exporter on ``METRICS_PORT`` (default 8000).
- Increments ``alpha_assay_bars_processed_total{feed="stub"}`` every
  HEARTBEAT_INTERVAL_SECONDS.
- Logs a single heartbeat line to stdout per tick.
- Exits cleanly on SIGTERM / SIGINT, draining in at most ``DRAIN_TIMEOUT_SECONDS``.

replaces this entrypoint with the real ``alpha_assay paper``
command wired to IBKR and Databento. Do not add real trading logic here.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from types import FrameType

from alpha_assay.observability import metrics as M
from alpha_assay.observability import start_metrics_server

HEARTBEAT_INTERVAL_SECONDS = 5
DRAIN_TIMEOUT_SECONDS = 20

logger = logging.getLogger("alpha_assay.paper_trader_stub")


def heartbeat_once() -> None:
    """Emit one heartbeat tick. Unit-testable; no sleeps, no side-effects
    beyond a counter increment and a log line.
    """
    M.bars_processed_total.labels(feed="stub").inc()
    logger.info("paper-trader-stub heartbeat: bars_processed_total{feed=stub} += 1")


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        logger.info("paper-trader-stub received signal %d; requesting stop", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def run(metrics_port: int, stop_event: threading.Event | None = None) -> int:
    """Run the stub loop until stop_event fires. Returns a process exit code."""
    if stop_event is None:
        stop_event = threading.Event()
        _install_signal_handlers(stop_event)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "paper-trader-stub starting: metrics on :%d, heartbeat every %ds",
        metrics_port,
        HEARTBEAT_INTERVAL_SECONDS,
    )

    start_metrics_server(port=metrics_port)

    drain_deadline: float | None = None
    while True:
        if stop_event.is_set():
            if drain_deadline is None:
                drain_deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
                logger.info("paper-trader-stub draining (max %ds)", DRAIN_TIMEOUT_SECONDS)
            if time.monotonic() >= drain_deadline:
                logger.info("paper-trader-stub drain complete; exiting")
                return 0
        heartbeat_once()
        # Sleep in small slices so signal handling stays responsive.
        for _ in range(HEARTBEAT_INTERVAL_SECONDS):
            if stop_event.is_set() and drain_deadline is None:
                break
            time.sleep(1)


def main() -> int:
    raw_port = os.environ.get("METRICS_PORT", "8000")
    try:
        metrics_port = int(raw_port)
    except ValueError:
        print(f"invalid METRICS_PORT {raw_port!r}; must be int", file=sys.stderr)
        return 2
    return run(metrics_port=metrics_port)


if __name__ == "__main__":
    sys.exit(main())
