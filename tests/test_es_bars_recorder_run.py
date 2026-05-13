# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for es-bars-recorder run.py _resolve_es_expiry / _resolve_es_expiry_with_wait.

Verifies the resolution contract:
  1. ES_EXPIRY env var set   -> use it (validated); do not read Redis.
  2. ES_EXPIRY unset         -> read from the Redis front-month metadata key.
  3. Neither present         -> FrontMonthMissingError after wait (fail closed).
  4. Cold-start race         -> polling sees the key once published.
"""

from __future__ import annotations

import pytest

from alpha_assay.data.front_month import FrontMonthMissingError, write_front_month
from infra.recorders.ibkr_es_bars.run import _resolve_es_expiry, _resolve_es_expiry_with_wait

# ---------------------------------------------------------------------------
# _resolve_es_expiry (delegates to _resolve_es_expiry_with_wait)
# ---------------------------------------------------------------------------


def test_resolve_expiry_prefers_env_override(fake_redis, monkeypatch):
    """If ES_EXPIRY is set, use it; do not read from Redis."""
    monkeypatch.setenv("ES_EXPIRY", "20260919")
    write_front_month(fake_redis, symbol="ES", exchange="CME", expiry="20260618")
    assert _resolve_es_expiry(fake_redis) == ("20260919", "ES_EXPIRY env")


def test_resolve_expiry_reads_redis_when_unset(fake_redis, monkeypatch):
    """If ES_EXPIRY unset, read from Redis."""
    monkeypatch.delenv("ES_EXPIRY", raising=False)
    write_front_month(fake_redis, symbol="ES", exchange="CME", expiry="20260619")
    assert _resolve_es_expiry(fake_redis) == ("20260619", "Redis metadata key")


def test_resolve_expiry_raises_when_neither_present(fake_redis, monkeypatch):
    """No env + no key -> fail closed after polling window expires."""
    monkeypatch.delenv("ES_EXPIRY", raising=False)
    # Use tiny timeouts so the test completes quickly.
    with pytest.raises(FrontMonthMissingError):
        _resolve_es_expiry_with_wait(
            fake_redis, max_wait_seconds=0.01, poll_interval_seconds=0.005
        )


# ---------------------------------------------------------------------------
# Cold-start polling: consumer waits for producer
# ---------------------------------------------------------------------------


def test_resolve_expiry_waits_for_redis_publication(fake_redis, monkeypatch):
    """Consumer boots before producer writes; polling sees the key once published."""
    import threading
    import time

    monkeypatch.delenv("ES_EXPIRY", raising=False)

    def _publish_after_delay():
        time.sleep(0.05)
        write_front_month(fake_redis, symbol="ES", exchange="CME", expiry="20260619")

    threading.Thread(target=_publish_after_delay, daemon=True).start()

    expiry, source = _resolve_es_expiry_with_wait(
        fake_redis, max_wait_seconds=2.0, poll_interval_seconds=0.01
    )
    assert expiry == "20260619"
    assert source == "Redis metadata key"
