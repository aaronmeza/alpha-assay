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

import re
from datetime import datetime

import redis as redis_pkg

_YYYYMMDD_RE = re.compile(r"^\d{8}$")


class FrontMonthMissingError(RuntimeError):
    """Raised when a consumer reads the front-month key before
    the producer has written it (cold-start race) or after Redis loss."""


class InvalidExpiryError(ValueError):
    """Raised when an expiry string is not a well-formed YYYYMMDD."""


def validate_yyyymmdd(s: str, *, source: str = "value") -> str:
    """Validate `s` is YYYYMMDD (8 digits, parseable as a date).

    Returns s unchanged on success. Raises InvalidExpiryError otherwise.

    The `source` arg is included in the error message so the operator
    knows WHERE the bad value came from (env override, IBKR, Redis read).
    """
    if not isinstance(s, str) or not _YYYYMMDD_RE.match(s):
        raise InvalidExpiryError(f"invalid expiry {s!r} from {source}: expected 8 digits (YYYYMMDD)")
    try:
        # Real-date check: ensures e.g. "20260230" (Feb 30) is rejected.
        datetime.strptime(s, "%Y%m%d")
    except ValueError as e:
        raise InvalidExpiryError(f"invalid expiry {s!r} from {source}: {e}") from e
    return s


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
    validate_yyyymmdd(expiry, source="write")
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
    raw = val.decode() if isinstance(val, bytes) else str(val)
    validate_yyyymmdd(raw, source="redis read")
    return raw
