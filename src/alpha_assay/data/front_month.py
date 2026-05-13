# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Front-month contract metadata key in Redis.

The producer (ibkr-feed) resolves the current front-month via
ContFuture + qualifyContracts, then writes the resolved expiry
(YYYYMMDD string) to a well-known Redis key. Consumers read this
key at startup and on a daily pre-open beat to know which
per-contract stream to subscribe to.
"""

from __future__ import annotations

import redis as redis_pkg


class FrontMonthMissingError(RuntimeError):
    """Raised when a consumer reads the front-month key before
    the producer has written it (cold-start race) or after Redis loss."""


def front_month_redis_key(symbol: str, exchange: str) -> str:
    """Deterministic Redis key for a (symbol, exchange) pair."""
    return f"alpha_assay:front_month:{symbol.lower()}.{exchange.lower()}"


def write_front_month(
    r: redis_pkg.Redis,
    *,
    symbol: str,
    exchange: str,
    expiry: str,
) -> None:
    """Producer-side: publish the resolved front-month expiry."""
    r.set(front_month_redis_key(symbol, exchange), expiry)


def read_front_month(
    r: redis_pkg.Redis,
    *,
    symbol: str,
    exchange: str,
) -> str:
    """Consumer-side: read the current front-month expiry."""
    key = front_month_redis_key(symbol, exchange)
    val = r.get(key)
    if val is None:
        raise FrontMonthMissingError(
            f"no front-month set for {symbol}@{exchange} (key {key!r}); "
            f"the producer (ibkr-feed) may not have started yet"
        )
    return val.decode() if isinstance(val, bytes) else str(val)
