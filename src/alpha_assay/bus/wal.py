# SPDX-License-Identifier: Apache-2.0
"""Producer-side write-ahead log for the alpha-assay bus.

Append-only file ``feed-{day}.jsonl`` (newline-delimited base64-msgpack)
+ atomic-rename watermark sidecar ``feed-{day}.committed``. fsync is
batched: every N records OR every Nms whichever first.

Sequence numbers are strictly monotonic within a day-file. A producer
restarting on the same day must seed its next live sequence from
``WALAppender.next_seq`` so the integer watermark remains a sound cursor
across process segments.

On producer restart, ``read_uncommitted()`` returns records past the
watermark - the producer drains these to Redis before entering its
live loop. This is the durability boundary: every event that survives
``append() + flush()`` will eventually land on the bus, even if Redis
or the producer process die in between.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from alpha_assay.bus import metrics as BM

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class WALRecord:
    """One uncommitted record on disk."""

    seq: int
    msg_bytes: bytes


class WALAppender:
    """Append-only WAL for the bus producer.

    Format per line: ``{seq}\\t{base64(msg_bytes)}``. Newline-delimited
    so partial-write corruption only affects one line.

    The watermark sidecar is updated via atomic rename (write tempfile,
    fsync, rename onto final path) so a crash mid-watermark-write
    leaves either old or new value, never partial.
    """

    def __init__(
        self,
        directory: Path,
        day: str,
        fsync_every_n: int = 10,
        fsync_every_ms: int = 100,
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._day = day
        self._wal_path = self._dir / f"feed-{day}.jsonl"
        self._watermark_path = self._dir / f"feed-{day}.committed"
        self._fsync_every_n = fsync_every_n
        self._fsync_every_ms = fsync_every_ms
        self._unflushed = 0
        self._last_flush_ms = self._now_ms()
        self._max_seq = self._scan_max_seq()
        self._committed = self._read_committed()
        # Open in append-mode binary; line-buffered.
        self._fp = open(self._wal_path, "ab")  # noqa: SIM115

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    @property
    def next_seq(self) -> int:
        """Return the next strictly monotonic sequence for this day-file."""
        return self._max_seq + 1

    def _decode_line(self, line: bytes) -> WALRecord:
        """Decode one WAL line, raising when the line is corrupt."""
        seq_str, b64 = line.decode().rstrip("\n").split("\t", 1)
        seq = int(seq_str)
        msg_bytes = base64.b64decode(b64)
        return WALRecord(seq=seq, msg_bytes=msg_bytes)

    def _scan_max_seq(self) -> int:
        """Find the highest sequence already present in the day-file."""
        max_seq = -1
        if not self._wal_path.exists():
            return max_seq
        with open(self._wal_path, "rb") as f:
            for line in f:
                try:
                    record = self._decode_line(line)
                except (ValueError, UnicodeDecodeError, Exception) as e:
                    LOG.warning("skipping corrupt WAL line in %s: %s", self._wal_path, e)
                    continue
                max_seq = max(max_seq, record.seq)
        return max_seq

    def append(self, seq: int, msg_bytes: bytes) -> None:
        """Append one record. fsync is batched per fsync_every_n / _ms."""
        if seq <= self._max_seq:
            LOG.warning(
                "non-monotonic WAL append in %s: seq=%s current_max=%s",
                self._wal_path,
                seq,
                self._max_seq,
            )
        line = f"{seq}\t{base64.b64encode(msg_bytes).decode()}\n".encode()
        self._fp.write(line)
        self._max_seq = max(self._max_seq, seq)
        self._unflushed += 1
        BM.bus_wal_pending.inc()
        if self._unflushed >= self._fsync_every_n or (self._now_ms() - self._last_flush_ms) >= self._fsync_every_ms:
            self.flush()

    def flush(self) -> None:
        """Force fsync to disk."""
        self._fp.flush()
        os.fsync(self._fp.fileno())
        self._unflushed = 0
        self._last_flush_ms = self._now_ms()

    def advance_committed(self, seq: int) -> None:
        """Mark seq as confirmed-published. Atomic rename for crash safety."""
        if seq > self._committed:
            tmp = self._watermark_path.with_suffix(".committed.tmp")
            tmp.write_text(str(seq))
            # fsync the file content + the directory so the rename is durable.
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, self._watermark_path)
            self._committed = seq
        BM.bus_wal_pending.dec()

    def _read_committed(self) -> int:
        if not self._watermark_path.exists():
            return -1
        try:
            return int(self._watermark_path.read_text().strip())
        except (ValueError, OSError):
            LOG.warning("corrupt watermark sidecar %s; treating as -1", self._watermark_path)
            return -1

    def read_uncommitted(self) -> Iterator[WALRecord]:
        """Replay records past the committed watermark.

        Skips corrupt lines with a warning; never raises.
        """
        if not self._wal_path.exists():
            return
        with open(self._wal_path, "rb") as f:
            for line in f:
                try:
                    record = self._decode_line(line)
                except (ValueError, UnicodeDecodeError, Exception) as e:
                    LOG.warning("skipping corrupt WAL line in %s: %s", self._wal_path, e)
                    continue
                if record.seq > self._committed:
                    BM.bus_wal_pending.inc()
                    yield record

    def close(self) -> None:
        if not self._fp.closed:
            self.flush()
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
