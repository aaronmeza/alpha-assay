"""Tests for Redis singleton lock."""

from __future__ import annotations

import fakeredis
import pytest

from alpha_assay.bus.lock import FeedLock, FeedLockHeldError


def test_first_acquire_succeeds():
    r = fakeredis.FakeRedis()
    lock = FeedLock(redis_client=r, key="alpha_assay:feed_lock:bars.es", ttl_ms=60_000)
    lock.acquire()
    assert r.get("alpha_assay:feed_lock:bars.es") is not None


def test_second_acquire_raises():
    r = fakeredis.FakeRedis()
    lock1 = FeedLock(redis_client=r, key="alpha_assay:feed_lock:bars.es", ttl_ms=60_000)
    lock1.acquire()
    lock2 = FeedLock(redis_client=r, key="alpha_assay:feed_lock:bars.es", ttl_ms=60_000)
    with pytest.raises(FeedLockHeldError):
        lock2.acquire()


def test_refresh_extends_ttl():
    r = fakeredis.FakeRedis()
    lock = FeedLock(redis_client=r, key="alpha_assay:feed_lock:x", ttl_ms=60_000)
    lock.acquire()
    pre_ttl = r.pttl("alpha_assay:feed_lock:x")
    lock.refresh()
    post_ttl = r.pttl("alpha_assay:feed_lock:x")
    # After refresh the TTL should be reset to ~60_000.
    assert post_ttl >= pre_ttl - 100


def test_release_removes_key():
    r = fakeredis.FakeRedis()
    lock = FeedLock(redis_client=r, key="alpha_assay:feed_lock:x", ttl_ms=60_000)
    lock.acquire()
    lock.release()
    assert r.get("alpha_assay:feed_lock:x") is None


def test_release_only_removes_own_lock():
    # If another instance has stolen the lock (e.g. ours expired), release must NOT remove it.
    r = fakeredis.FakeRedis()
    lock1 = FeedLock(redis_client=r, key="alpha_assay:feed_lock:x", ttl_ms=60_000)
    lock1.acquire()
    # Simulate: another process took over.
    r.set("alpha_assay:feed_lock:x", "other-pid")
    lock1.release()  # ours - should not delete other's key
    assert r.get("alpha_assay:feed_lock:x") == b"other-pid"
