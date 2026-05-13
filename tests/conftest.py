# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import fakeredis
import pytest


@pytest.fixture
def fake_redis():
    """Return a fresh in-memory FakeRedis instance per test."""
    return fakeredis.FakeRedis()
