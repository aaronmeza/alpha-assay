# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for es-bars-recorder run.py _resolve_es_expiry.

Verifies the three-case resolution contract:
  1. ES_EXPIRY env var set   -> use it; do not read Redis.
  2. ES_EXPIRY unset         -> read from the Redis front-month metadata key.
  3. Neither present         -> FrontMonthMissingError (fail closed).
"""

from __future__ import annotations

import pytest

from alpha_assay.data.front_month import FrontMonthMissingError, write_front_month
from infra.recorders.ibkr_es_bars.run import _resolve_es_expiry

# ---------------------------------------------------------------------------
# _resolve_es_expiry
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
    """No env + no key -> fail closed."""
    monkeypatch.delenv("ES_EXPIRY", raising=False)
    with pytest.raises(FrontMonthMissingError):
        _resolve_es_expiry(fake_redis)
