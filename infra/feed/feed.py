# SPDX-License-Identifier: Apache-2.0
"""ibkr-feed daemon: holds the single IBKR subscription per contract,
publishes to Redis Streams via the alpha_assay.bus module."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import redis as redis_pkg
import yaml

from alpha_assay.bus import metrics as BM
from alpha_assay.bus.lock import FeedLock, FeedLockHeldError
from alpha_assay.bus.producer import Producer
from alpha_assay.bus.streams import CURRENT_VERSION, Message, stream_name_for_bars, stream_name_for_ticks
from alpha_assay.bus.streams import pack as pack_msg
from alpha_assay.bus.wal import WALAppender

LOG = logging.getLogger(__name__)


class ManifestError(ValueError):
    """Raised on invalid feed manifest."""


@dataclass(frozen=True)
class Subscription:
    """One feed subscription decoded from the manifest.

    ``required`` (default True) controls failure isolation: a required
    subscription that errors exits the daemon (the historical behaviour -
    the container restart policy reconnects everything), while an
    optional (``required: false``) subscription that errors is logged
    loudly and dropped without disturbing the other feeds. Mark feeds
    that depend on a market-data entitlement the account may lack as
    optional so a permissions error cannot take down the core streams.

    ``exchange`` / ``currency`` apply to ticks subscriptions only and
    route the IBKR Index contract (defaults NYSE/USD for back-compat
    with manifests that predate multi-venue breadth).
    """

    kind: str  # "bars" or "ticks"
    contract: dict[str, Any] | None = None  # for bars
    symbol: str | None = None  # for ticks
    bar_size: str = "1 min"
    what_to_show: str = "TRADES"
    exchange: str = "NYSE"  # for ticks: IBKR Index routing exchange
    currency: str = "USD"  # for ticks: IBKR Index currency
    required: bool = True

    @property
    def stream(self) -> str:
        """Deterministic Redis stream name for this subscription."""
        if self.kind == "bars":
            return stream_name_for_bars(self.contract or {})
        return stream_name_for_ticks(self.symbol or "")


@dataclass(frozen=True)
class FeedManifest:
    subscriptions: list[Subscription] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> FeedManifest:
        raw = yaml.safe_load(Path(path).read_text())
        subs_raw = raw.get("subscriptions", [])
        subs: list[Subscription] = []
        for s in subs_raw:
            kind = s.get("kind")
            required = bool(s.get("required", True))
            if kind == "bars":
                subs.append(
                    Subscription(
                        kind="bars",
                        contract=dict(s["contract"]),
                        bar_size=s.get("bar_size", "1 min"),
                        what_to_show=s.get("what_to_show", "TRADES"),
                        required=required,
                    )
                )
            elif kind == "ticks":
                subs.append(
                    Subscription(
                        kind="ticks",
                        symbol=str(s["symbol"]),
                        exchange=str(s.get("exchange", "NYSE")),
                        currency=str(s.get("currency", "USD")),
                        required=required,
                    )
                )
            else:
                raise ManifestError(f"unknown subscription kind: {kind!r}")
        return cls(subscriptions=subs)


class FreshnessTracker:
    """Monotonic last-publish time per stream, read by the staleness watchdog.

    The connection watchdog only catches a dropped socket. A *silent* socket -
    IBKR connected but pushing no bars/ticks - leaves the publish loop idle
    with ``is_connected`` still True (the 2026-06-17 14:06-14:24 CT stall). This
    tracker records, at the publish boundary, the monotonic time each stream
    last landed an event in Redis, so the staleness watchdog can notice a feed
    has gone quiet despite a live connection and restart the daemon for a fresh
    subscription.

    Uses a monotonic clock (not wall-clock) so a system-clock step never yields
    a spurious age. ``clock`` is injectable for deterministic tests.
    """

    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._last: dict[str, float] = {}

    def seed(self, stream: str) -> None:
        """Establish a baseline at subscription start so a feed that has not
        yet produced its first event is not seen as infinitely stale."""
        self._last[stream] = self._clock()

    def mark(self, stream: str) -> None:
        """Record that *stream* just published an event."""
        self._last[stream] = self._clock()

    def age(self, stream: str) -> float | None:
        """Seconds since *stream* last published, or None if never seeded."""
        last = self._last.get(stream)
        if last is None:
            return None
        return max(0.0, self._clock() - last)


class IBKRFeedDaemon:
    """Single producer for one or more IBKR subscriptions.

    Per subscription, the receive->WAL-append->publish->watermark-advance
    sequence guarantees that every IBKR event durably lands in either
    Redis (immediately) or the WAL (for replay on restart).
    """

    def __init__(
        self,
        adapter,  # IBKRAdapter (untyped to keep this file mock-test-friendly)
        redis_client: redis_pkg.Redis,
        wal_dir: Path,
        maxlen: int = 3600,
        *,
        freshness: FreshnessTracker | None = None,
    ) -> None:
        self._adapter = adapter
        self._redis = redis_client
        self._producer = Producer(redis_client=redis_client, maxlen=maxlen)
        self._wal_dir = Path(wal_dir)
        self._wal_dir.mkdir(parents=True, exist_ok=True)
        # Shared with the staleness watchdog (None in unit tests that don't
        # exercise it). Stamped at the publish boundary below.
        self._freshness = freshness

    async def run_subscription(self, sub: Subscription) -> None:
        """Run one subscription end-to-end. Blocks until cancelled or stream ends."""
        if sub.kind == "bars":
            stream = stream_name_for_bars(sub.contract or {})
            await self._connect_if_needed()
            gen = self._adapter.subscribe_bars(
                sub.contract,
                bar_size=sub.bar_size,
                what_to_show=sub.what_to_show,
            )
        elif sub.kind == "ticks":
            stream = stream_name_for_ticks(sub.symbol or "")
            await self._connect_if_needed()
            gen = self._adapter.subscribe_breadth(
                symbol=sub.symbol,
                exchange=sub.exchange,
                currency=sub.currency,
            )
        else:
            raise ValueError(f"unsupported subscription kind: {sub.kind!r}")

        # Acquire singleton lock for this contract.
        lock = FeedLock(
            redis_client=self._redis,
            key=f"alpha_assay:feed_lock:{stream}",
            ttl_ms=60_000,
        )
        try:
            lock.acquire()
        except FeedLockHeldError:
            BM.feed_lock_state.labels(contract=stream).set(0)
            LOG.error("feed lock held for %s; refusing to start", stream)
            raise
        BM.feed_lock_state.labels(contract=stream).set(1)

        # Baseline the staleness clock at subscription start so a feed that has
        # not yet produced its first event is not instantly "infinitely stale"
        # (the watchdog's warmup guard gives it time to start flowing).
        if self._freshness is not None:
            self._freshness.seed(stream)

        # Drain any uncommitted WAL records first. Per-stream WAL subdirs
        # so concurrent subscriptions don't share state (otherwise each
        # task's drain would republish messages from other subscriptions
        # to its own Redis stream - the cause of the cross-stream contam
        # bug observed during the first SER9 deploy).
        wal_subdir = self._wal_dir / stream
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        wal = WALAppender(directory=wal_subdir, day=day)
        for record in wal.read_uncommitted():
            self._redis.xadd(stream, {"data": record.msg_bytes}, maxlen=3600, approximate=True)
            wal.advance_committed(record.seq)

        seq_counter = 0
        try:
            async for event in gen:
                seq = seq_counter
                seq_counter += 1
                # Build payload.
                ts_event_ns = self._extract_event_ts_ns(event)
                payload = self._payload_for(sub, event)
                msg = Message(
                    v=CURRENT_VERSION,
                    seq=seq,
                    ts_recv_ns=time.time_ns(),
                    ts_event_ns=ts_event_ns,
                    stream=stream,
                    payload=payload,
                )
                msg_bytes = pack_msg(msg)
                wal.append(seq=seq, msg_bytes=msg_bytes)
                # Direct XADD so we control the bytes; bypass Producer to
                # keep WAL/publish/advance-committed atomic at this level.
                self._redis.xadd(stream, {"data": msg_bytes}, maxlen=3600, approximate=True)
                BM.bus_publish_total.labels(stream=stream).inc()
                wal.advance_committed(seq)
                # Publish boundary: data is flowing for this stream.
                if self._freshness is not None:
                    self._freshness.mark(stream)
        finally:
            wal.close()
            BM.feed_lock_state.labels(contract=stream).set(0)
            lock.release()

    async def _connect_if_needed(self) -> None:
        if not getattr(self._adapter, "is_connected", False):
            await self._adapter.connect_async()

    @staticmethod
    def _extract_event_ts_ns(event: dict) -> int:
        ts = event.get("timestamp")
        if ts is None:
            return time.time_ns()
        if isinstance(ts, (int, float)):
            return int(ts * 1e9) if ts < 1e12 else int(ts)
        # pandas / ISO string
        return int(pd.Timestamp(ts).value)  # nanoseconds since epoch

    @staticmethod
    def _payload_for(sub: Subscription, event: dict) -> dict:
        if sub.kind == "bars":
            return {
                "open": float(event["open"]),
                "high": float(event["high"]),
                "low": float(event["low"]),
                "close": float(event["close"]),
                "volume": int(event["volume"]),
                "ts_minute_utc": int(pd.Timestamp(event["timestamp"]).value // 10**9),
            }
        else:  # ticks
            return {
                "value": float(event["value"]),
                "symbol": str(event["symbol"]),
            }
