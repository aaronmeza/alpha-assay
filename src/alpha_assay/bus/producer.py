# SPDX-License-Identifier: Apache-2.0
"""Bus Producer - XADD with MAXLEN cap, monotonic per-stream sequence."""

from __future__ import annotations

import time
from typing import Any

import redis as redis_pkg

from alpha_assay.bus import metrics as BM
from alpha_assay.bus.streams import CURRENT_VERSION, Message, pack


class Producer:
    """Publishes messages to Redis Streams with a per-stream monotonic seq.

    Sequence is local to this Producer instance. On restart, seq resets;
    consumers identify ordering via stream-id (server-assigned) and
    timestamps (event + receive). The seq is for WAL recovery dedup, not
    consumer ordering.
    """

    def __init__(
        self,
        redis_client: redis_pkg.Redis,
        maxlen: int = 3600,
    ) -> None:
        self._redis = redis_client
        self._maxlen = maxlen
        self._seq_per_stream: dict[str, int] = {}

    def publish(
        self,
        stream: str,
        payload: dict[str, Any],
        ts_event_ns: int,
        ts_recv_ns: int | None = None,
    ) -> int:
        """Publish a message to ``stream``. Returns the seq assigned.

        Records publish lag in :data:`bus_publish_lag_seconds`.
        Increments :data:`bus_publish_total`.
        """
        if ts_recv_ns is None:
            ts_recv_ns = time.time_ns()
        seq = self._seq_per_stream.get(stream, 0)
        self._seq_per_stream[stream] = seq + 1
        msg = Message(
            v=CURRENT_VERSION,
            seq=seq,
            ts_recv_ns=ts_recv_ns,
            ts_event_ns=ts_event_ns,
            stream=stream,
            payload=payload,
        )
        raw = pack(msg)
        publish_start = time.monotonic()
        self._redis.xadd(
            stream,
            {"data": raw},
            maxlen=self._maxlen,
            approximate=True,
        )
        BM.bus_publish_lag_seconds.labels(stream=stream).observe(
            time.monotonic() - publish_start
        )
        BM.bus_publish_total.labels(stream=stream).inc()
        return seq
