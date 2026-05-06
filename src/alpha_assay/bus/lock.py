# SPDX-License-Identifier: Apache-2.0
"""Redis-backed singleton lock for ibkr-feed.

Pattern: SET NX PX. The lock value is a unique fencing token per
acquirer (uuid4 + pid) so release only removes our own key.
"""

from __future__ import annotations

import os
import uuid

import redis as redis_pkg


class FeedLockHeldError(RuntimeError):
    """Raised when SET NX fails because another instance holds the lock."""


class FeedLock:
    """SET NX PX lock with refresh + safe release."""

    def __init__(
        self,
        redis_client: redis_pkg.Redis,
        key: str,
        ttl_ms: int = 60_000,
    ) -> None:
        self._redis = redis_client
        self._key = key
        self._ttl_ms = ttl_ms
        self._token = f"{os.getpid()}-{uuid.uuid4().hex}"

    def acquire(self) -> None:
        """SET key value NX PX ttl_ms. Raise on conflict."""
        ok = self._redis.set(self._key, self._token, nx=True, px=self._ttl_ms)
        if not ok:
            held_by = self._redis.get(self._key)
            raise FeedLockHeldError(
                f"feed lock {self._key!r} already held by {held_by!r}; refusing to start a duplicate producer"
            )

    def refresh(self) -> bool:
        """Extend our TTL only if we still hold the key.

        Returns True iff refresh succeeded; False means we lost the lock
        (another process stole it; caller should exit).

        Atomic via Lua eval - GET-then-PEXPIRE has a race window where
        an expired-then-reacquired key could let us extend a lock we no
        longer own.
        """
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "  return redis.call('pexpire', KEYS[1], ARGV[2]) "
            "else "
            "  return 0 "
            "end"
        )
        result = self._redis.eval(script, 1, self._key, self._token, self._ttl_ms)
        return bool(result)

    def release(self) -> None:
        """Delete the key only if it's still ours (token match).

        Atomic via Lua eval - GET-then-DEL has a race window where an
        expired-then-reacquired key could let us delete another holder's
        lock.
        """
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "  return redis.call('del', KEYS[1]) "
            "else "
            "  return 0 "
            "end"
        )
        self._redis.eval(script, 1, self._key, self._token)
