# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Shared IBKR connection-resilience helpers.

Single source for the connection-watchdog pattern used by every component
that holds an IB Gateway connection - the ``ibkr-feed`` producer
(``infra/feed/run.py``) and the paper-trader order path
(``scripts/paper_dryrun.py``). Extracting it here keeps the resilience
behaviour from being silently dropped when a new component starts holding
a connection: the bug that left the paper-trader without a watchdog for ten
days (it died on the first nightly IB Gateway restart and never reconnected)
was exactly that - the producer had the watchdog, the new consumer-turned-
order-submitter did not.

The watchdog is intentionally non-destructive: it only *observes* the
connection and returns when it drops. The caller decides what to do with the
news - the established pattern in this codebase is to exit ``EXIT_RESTART`` so
the container restart policy (``restart: unless-stopped``) brings the process
back with a fresh connect (the proven cold-start path), rather than attempting
an in-process reconnect (ib_insync does not auto-reconnect after a peer-closed
socket, which is what the nightly IBC restart produces).
"""

from __future__ import annotations

import asyncio
from typing import Protocol

# Process exit codes. A clean SIGTERM/SIGINT exits EXIT_OK; any condition that
# wants the supervisor (Docker ``restart: unless-stopped``) to recycle the
# process exits EXIT_RESTART.
EXIT_OK = 0
EXIT_RESTART = 2


class _Connectable(Protocol):
    """Minimal surface the watchdog needs: a live ``is_connected`` flag."""

    @property
    def is_connected(self) -> bool: ...


async def watch_connection(adapter: _Connectable, poll_seconds: float = 5.0) -> None:
    """Block until the IBKR connection is lost.

    Polls ``adapter.is_connected`` and returns as soon as it reads False.
    Cheap (one bool check per ``poll_seconds``) and side-effect free - the
    caller decides what to do with the news (see module docstring).
    """
    while adapter.is_connected:
        await asyncio.sleep(poll_seconds)
