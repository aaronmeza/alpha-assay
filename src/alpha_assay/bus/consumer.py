# SPDX-License-Identifier: Apache-2.0
"""Bus Consumer - XREAD BLOCK 0 over a single stream.

start_id="0" replays from the beginning of the stream (bounded by MAXLEN).
start_id="$" starts from the next new entry (drops history).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import redis as redis_pkg

from alpha_assay.bus import metrics as BM
from alpha_assay.bus.streams import Message, unpack

_VALID_START_IDS = {"0", "$"}


class Consumer:
    """Single-stream Redis Streams consumer using ``XREAD BLOCK``.

    Maintains a cursor across calls so iter_messages() can be called
    repeatedly. ``last_id`` is the last delivered Redis stream-id.
    """

    def __init__(
        self,
        redis_client: redis_pkg.Redis,
        stream: str,
        consumer_id: str,
        start_id: str = "$",
    ) -> None:
        if start_id not in _VALID_START_IDS:
            raise ValueError(f"start_id must be '0' (replay) or '$' (latest); got {start_id!r}")
        self._redis = redis_client
        self._stream = stream
        self._consumer_id = consumer_id
        # Cursor: when starting from $, ask Redis for the current $ once;
        # subsequent calls track the highest delivered id.
        self._cursor = start_id

    def iter_messages(
        self,
        max_messages: int | None = None,
        block_ms: int = 0,
    ) -> Iterator[Message]:
        """Yield messages until ``max_messages`` reached or stream closes.

        ``block_ms=0`` blocks indefinitely on XREAD (the typical loop case).
        """
        delivered = 0
        while True:
            if max_messages is not None and delivered >= max_messages:
                return
            entries = self._redis.xread(
                {self._stream: self._cursor},
                count=100,
                block=block_ms,
            )
            if not entries:
                # Block timed out (only happens when block_ms > 0).
                return
            # entries is [(stream_name, [(id, {fields}), ...])]
            _stream_name, items = entries[0]
            for entry_id, fields in items:
                raw: bytes = fields[b"data"]
                msg = unpack(raw)
                BM.bus_consume_total.labels(stream=self._stream, consumer=self._consumer_id).inc()
                lag_s = max(0.0, (time.time_ns() - msg.ts_event_ns) / 1e9)
                BM.bus_consume_lag_seconds.labels(stream=self._stream, consumer=self._consumer_id).observe(lag_s)
                self._cursor = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                yield msg
                delivered += 1
                if max_messages is not None and delivered >= max_messages:
                    return
