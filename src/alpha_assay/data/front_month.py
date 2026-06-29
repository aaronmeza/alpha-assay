# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Front-month contract metadata key in Redis.

The producer (ibkr-feed) resolves the current front-month via
ContFuture + qualifyContracts, then writes the resolved expiry
(YYYYMMDD string) to a well-known Redis key. Consumers read this
key at startup and on a daily pre-open beat to know which
per-contract stream to subscribe to.

This module is the single, symbol-parameterized home for that
consumer-side resolution:

- :func:`read_front_month_with_wait` resolves the front-month expiry,
  honouring an operator env-pin first and otherwise polling Redis for
  up to a deadline (the cold-start race where a consumer boots before
  the producer has written the key). It replaces the per-consumer
  ``_resolve_*_expiry_with_wait`` copies that previously duplicated this
  logic in each recorder / paper-trader entrypoint.
- :class:`FrontMonthWatcher` lets a *long-lived* consumer notice a
  quarterly contract roll (the producer re-resolves and starts
  publishing to a new per-contract stream) and rebind its Redis Stream
  subscription to the new stream WITHOUT a process restart. Pinning the
  expiry to two contracts' streams is unsound (mixed-contract bars), so
  each consumer keeps exactly one single-expiry subscription and the
  watcher signals when that subscription has gone stale.

Everything is parameterized by ``(symbol, exchange)`` so a second
recorder instance covering a different futures root (e.g. NQ) shares
the identical code path - no hard-coded ``es``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime

import redis as redis_pkg

_LOG = logging.getLogger(__name__)

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


def read_front_month_with_wait(
    redis_client: redis_pkg.Redis | None,
    *,
    symbol: str,
    exchange: str,
    env_override: str = "",
    env_override_source: str = "env override",
    max_wait_seconds: float = 60.0,
    poll_interval_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str]:
    """Resolve the front-month expiry, polling Redis up to a deadline.

    This is the single shared resolver behind every consumer's startup
    front-month resolution (es-bars recorder, nq-bars recorder,
    paper-trader). The env-var *policy* (which env names pin which root)
    stays in the consumer: the caller resolves its own operator pin and
    passes it in as ``env_override`` so the data layer stays generic.

    Resolution order:

    1. ``env_override`` set -> validate and use it (instant, no Redis).
       This is the operator emergency pin (``BARS_EXPIRY`` / ``ES_EXPIRY``).
    2. ``redis_client`` provided -> poll the per-root metadata key for up
       to ``max_wait_seconds`` (handles the cold-start race where the
       consumer boots before the producer has published the key).
    3. Neither -> :class:`FrontMonthMissingError` (fail closed).

    Args:
        redis_client: Bus Redis handle, or ``None`` in direct-IBKR mode.
        symbol: Futures root (e.g. ``ES``, ``NQ``). Not lowercased here;
            :func:`front_month_redis_key` normalises casing.
        exchange: Futures exchange (e.g. ``CME``).
        env_override: Operator expiry pin (``YYYYMMDD``), or ``""``.
        env_override_source: Human label for the pin's origin, used in the
            returned source string for operator traceability.
        max_wait_seconds: Total Redis poll budget before failing closed.
        poll_interval_seconds: Sleep between Redis polls.
        sleep: Injectable sleep (tests pass a no-op to avoid real waits).

    Returns:
        ``(expiry, source_label)`` - log the source for traceability.

    Raises:
        InvalidExpiryError: ``env_override`` is set but malformed.
        FrontMonthMissingError: no override and the key never appears.
    """
    if env_override:
        validated = validate_yyyymmdd(env_override, source=env_override_source)
        return validated, env_override_source
    if redis_client is None:
        raise FrontMonthMissingError(
            f"no expiry override is set and no Redis client - cannot resolve the "
            f"{symbol}@{exchange} front-month. Set an expiry env override (YYYYMMDD) "
            f"or ensure the bus Redis URL is set and the producer has published the "
            f"front-month metadata key."
        )
    deadline = time.monotonic() + max_wait_seconds
    last_err: Exception | None = None
    while True:
        try:
            value = read_front_month(redis_client, symbol=symbol, exchange=exchange)
            return value, "Redis metadata key"
        except FrontMonthMissingError as e:
            last_err = e
            if time.monotonic() >= deadline:
                break
            sleep(poll_interval_seconds)
    raise FrontMonthMissingError(
        f"{symbol}@{exchange} front-month not in Redis after {max_wait_seconds:.0f}s wait; "
        f"the producer may not have started or failed to resolve. Last error: {last_err}"
    )


class FrontMonthWatcher:
    """Observe the front-month key for a long-lived consumer across rolls.

    A bus consumer (recorder or paper-trader) binds its Redis Stream
    subscription to ``bars.<root>.<venue>.<expiry>`` at startup. When the
    producer rolls to the next quarterly contract it begins publishing to
    a *new* per-contract stream; a consumer that pinned the old expiry at
    startup then reads a now-dead stream forever (the silent roll-day stall
    this fix addresses). This watcher lets the consumer detect the roll on
    a cheap poll and rebind, with no process restart.

    Atomic key+stream flip (why a plain key-watch is safe)
    ------------------------------------------------------
    The producer flips the front-month KEY and its publishing stream
    together, atomically: on a detected rollover the feed daemon exits
    ``EXIT_RESTART`` and cold-starts, which re-resolves, writes the new key,
    and bakes the new per-contract stream in one step (see
    ``infra/feed/run.py`` ``_pre_open_requalify_loop`` + ``_run_subscriptions``).
    So when this watcher observes the key flip to a new expiry, the producer
    is already publishing that stream - a key-watch rebind is safe, with no
    pre-open-to-restart window for the key to lead the live stream.

    State:

    - ``bound_expiry`` - the expiry the consumer's stream is attached to and
      reading live data from. The consumer calls :meth:`mark_bound` after it
      rebinds.
    - ``resolved_expiry`` - the most recent good read of the front-month key
      (what the producer is publishing now); equals ``bound_expiry`` except
      in the brief instant between observing the flip and rebinding.

    ``diverged`` is ``True`` while ``resolved_expiry != bound_expiry`` - the
    transient between observing the new front month and the consumer
    rebinding to it. The paper-trader uses ``diverged`` as a trade-refusal
    backstop so it never acts on a rolled-off contract during that instant.

    Resilience: a missing key, an invalid value, or any Redis error returns
    ``None`` from :meth:`poll` and does NOT advance state - a transient blip
    must never tear down a healthy subscription. An operator env-pin
    (``pinned=True``) freezes the watcher entirely.
    """

    def __init__(
        self,
        redis_client: redis_pkg.Redis | None,
        *,
        symbol: str,
        exchange: str,
        current_expiry: str,
        pinned: bool = False,
    ) -> None:
        self._redis = redis_client
        self._symbol = symbol
        self._exchange = exchange
        self._bound = current_expiry
        self._resolved = current_expiry
        self._pinned = pinned

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def exchange(self) -> str:
        return self._exchange

    @property
    def bound_expiry(self) -> str:
        """Expiry the consumer's stream is currently attached to."""
        return self._bound

    @property
    def resolved_expiry(self) -> str:
        """Most recent good read of the front-month key."""
        return self._resolved

    @property
    def diverged(self) -> bool:
        """True while the bound stream is not yet the resolved front month."""
        return self._resolved != self._bound

    def mark_bound(self, expiry: str) -> None:
        """Record that the consumer has rebound its stream to ``expiry``."""
        self._bound = expiry

    def poll(self) -> str | None:
        """Re-read the key; return the new expiry when a cutover is due.

        Returns the new expiry when the key has flipped. The producer flips
        the key and its publishing stream together (it exits EXIT_RESTART on
        a rollover and cold-starts onto the new stream), so a flipped key
        means the new stream is already live. The caller rebinds its stream
        and then calls :meth:`mark_bound`. Returns ``None`` when no cutover
        is due: unchanged, pinned, no Redis handle, or a transient
        missing/invalid key (hold the current binding rather than tear down
        a healthy subscription on a blip).
        """
        if self._pinned or self._redis is None:
            return None
        try:
            candidate = read_front_month(self._redis, symbol=self._symbol, exchange=self._exchange)
        except (FrontMonthMissingError, InvalidExpiryError):
            # Transient: key briefly absent (producer restart mid-write) or a
            # malformed value. Hold the current binding rather than tearing
            # down a healthy subscription on a blip.
            return None
        except Exception:  # noqa: BLE001 - any Redis error is a transient blip
            _LOG.warning("front-month watch poll failed for %s@%s; holding binding", self._symbol, self._exchange)
            return None
        self._resolved = candidate
        if candidate == self._bound:
            return None
        # The key flipped: the producer rolled (key + stream flip atomically),
        # so the new stream is live. Signal the consumer to rebind.
        _LOG.info(
            "front-month rolled %s -> %s for %s@%s; rebinding consumer stream",
            self._bound,
            candidate,
            self._symbol,
            self._exchange,
        )
        return candidate
