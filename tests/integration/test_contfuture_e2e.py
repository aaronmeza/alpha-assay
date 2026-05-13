# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: ibkr-feed publishes the front-month key to Redis at startup.

Opt-in: requires ``RUN_INTEGRATION=1`` and a reachable Redis at
``REDIS_URL`` (default ``redis://localhost:6379/0``). The test asserts
that the key was written by a live ibkr-feed instance, so it must run
*after* the feed has booted.

Example invocation against a reachable Redis (same pattern as
``test_bus_e2e.py``)::

    RUN_INTEGRATION=1 REDIS_URL=redis://127.0.0.1:6379/0 \\
        python -m pytest tests/integration/test_contfuture_e2e.py -v
"""

from __future__ import annotations

import os

import pytest
import redis as redis_pkg

from alpha_assay.data.front_month import read_front_month

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="integration test - set RUN_INTEGRATION=1 to run",
)


def test_front_month_key_present_after_feed_startup():
    """After ibkr-feed boots, the front-month key for ES@CME exists in Redis
    and is an 8-digit YYYYMMDD string."""
    r = redis_pkg.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    expiry = read_front_month(r, symbol="ES", exchange="CME")
    assert len(expiry) == 8 and expiry.isdigit(), f"expected YYYYMMDD, got {expiry!r}"
    year = int(expiry[:4])
    assert 2026 <= year <= 2028, f"expiry year {year} outside plausible window"
