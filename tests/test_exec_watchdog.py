# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Behaviour of the shared IBKR connection watchdog (alpha_assay.exec.watchdog).

Given a component that holds an IB Gateway connection, the watchdog must return
as soon as that connection drops, so the caller can exit EXIT_RESTART and let the
container restart policy reconnect. This is the resilience the paper-trader was
missing when its order path died silently for ten days across a nightly IB
Gateway restart (alphaassay-0n6).
"""

from __future__ import annotations

import asyncio

import pytest

from alpha_assay.exec.watchdog import EXIT_OK, EXIT_RESTART, watch_connection


class _FakeAdapter:
    def __init__(self, *, connected: bool) -> None:
        self.is_connected = connected


def test_exit_restart_is_nonzero_so_docker_recycles():
    # Given the restart contract, the OK code is 0 and the restart code is non-zero
    # (Docker `restart: unless-stopped` recycles only on a non-zero exit).
    assert EXIT_OK == 0
    assert EXIT_RESTART != 0


def test_watch_connection_returns_when_connection_drops():
    # Given a connected adapter, When the connection drops, Then the watchdog returns.
    adapter = _FakeAdapter(connected=True)

    async def _run():
        async def _drop_later():
            await asyncio.sleep(0.02)
            adapter.is_connected = False

        asyncio.create_task(_drop_later())
        await asyncio.wait_for(watch_connection(adapter, poll_seconds=0.01), timeout=2.0)

    asyncio.run(_run())


def test_watch_connection_returns_immediately_when_already_disconnected():
    # Given an already-dead socket, Then the watchdog returns without waiting a full poll.
    adapter = _FakeAdapter(connected=False)
    asyncio.run(asyncio.wait_for(watch_connection(adapter, poll_seconds=30.0), timeout=1.0))


def test_watch_connection_blocks_while_connected():
    # Given a healthy connection, Then the watchdog does not return.
    adapter = _FakeAdapter(connected=True)

    async def _run():
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(watch_connection(adapter, poll_seconds=0.01), timeout=0.1)

    asyncio.run(_run())
