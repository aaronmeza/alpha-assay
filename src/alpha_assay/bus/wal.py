# SPDX-License-Identifier: Apache-2.0
"""Producer-side write-ahead log for the alpha-assay bus.

Append-only file ``feed-{day}.jsonl`` (newline-delimited base64-msgpack)
+ atomic-rename watermark sidecar ``feed-{day}.committed``. ``append()``
flushes and fsyncs each record before returning, so the caller's publish
step always happens after the event is durable on disk.

Sequence numbers are strictly monotonic within a day-file. A producer
restarting on the same day must seed its next live sequence from
``WALAppender.next_seq`` so the integer watermark remains a sound cursor
across process segments.

On producer restart, ``read_uncommitted()`` returns records past the
watermark - the producer drains these to Redis before entering its
live loop. This is the durability boundary: every event that survives
``append()`` will eventually land on the bus, even if Redis or the
producer process die in between.
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


@dataclass(frozen=True)
class WALScan:
    """Startup facts collected from one pass over the day-file."""

    max_seq: int
    strictly_increasing: bool


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
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._day = day
        self._wal_path = self._dir / f"feed-{day}.jsonl"
        self._watermark_path = self._dir / f"feed-{day}.committed"
        self._migrated_path = self._dir / f"feed-{day}.migrated"
        wal_existed = self._wal_path.exists()
        scan = self._scan_wal()
        self._max_seq = scan.max_seq
        self._strictly_increasing = scan.strictly_increasing
        self._needs_full_drain = not self._strictly_increasing and not self._migrated_path.exists()
        if not self._strictly_increasing:
            LOG.warning(
                "dirty WAL day-file detected: path=%s migrated=%s needs_full_drain=%s",
                self._wal_path,
                self._migrated_path.exists(),
                self._needs_full_drain,
            )
        self._committed = self._read_committed()
        if self._committed > self._max_seq:
            LOG.warning(
                "WAL committed watermark exceeds day-file max seq: path=%s committed=%s max_seq=%s",
                self._wal_path,
                self._committed,
                self._max_seq,
            )
        # Open in append-mode binary; line-buffered.
        self._fp = open(self._wal_path, "ab")  # noqa: SIM115
        if not wal_existed:
            self._fsync_directory()

    def _fsync_directory(self) -> None:
        """Durably link a newly created WAL day-file into its directory."""
        dir_fd = os.open(self._dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    @property
    def next_seq(self) -> int:
        """Next seq strictly above BOTH the durable log and the watermark.

        Legacy files can have a committed watermark ahead of the durable
        day-file. Seeding from the log alone would hand back a seq the
        watermark has already passed, so an unpublished record could be
        filtered out of replay. Seeding above both keeps every new record
        strictly above the cursor. Seqs need only be strictly increasing,
        not contiguous, so skipping the gap is free.
        """
        return max(self._max_seq, self._committed) + 1

    @property
    def needs_full_drain(self) -> bool:
        """Return True when a legacy dirty file needs one conservative drain."""
        return self._needs_full_drain

    @property
    def path(self) -> Path:
        """Return the WAL day-file path for diagnostic logging."""
        return self._wal_path

    @property
    def committed(self) -> int:
        """Highest seq confirmed published, per the durable watermark sidecar."""
        return self._committed

    @property
    def max_seq(self) -> int:
        """Highest seq durably on disk in the day-file (-1 when the file is empty).

        Exposed alongside ``committed`` so the cursor invariant this class now
        upholds - ``committed <= max_seq``, because ``append()`` fsyncs before
        the caller can publish and advance - is assertable from outside without
        reaching into private state. Legacy files written by the pre-fsync
        producer can still violate it; see ``next_seq``.
        """
        return self._max_seq

    def _decode_line(self, line: bytes) -> WALRecord:
        """Decode one WAL line, raising when the line is corrupt."""
        seq_str, b64 = line.decode().rstrip("\n").split("\t", 1)
        seq = int(seq_str)
        msg_bytes = base64.b64decode(b64)
        return WALRecord(seq=seq, msg_bytes=msg_bytes)

    def _scan_wal(self) -> WALScan:
        """Find the highest sequence and whether seqs strictly increase."""
        max_seq = -1
        prev_seq: int | None = None
        strictly_increasing = True
        if not self._wal_path.exists():
            return WALScan(max_seq=max_seq, strictly_increasing=strictly_increasing)
        with open(self._wal_path, "rb") as f:
            for line in f:
                try:
                    record = self._decode_line(line)
                except (ValueError, UnicodeDecodeError, Exception) as e:
                    LOG.warning("skipping corrupt WAL line in %s: %s", self._wal_path, e)
                    continue
                if prev_seq is not None and record.seq <= prev_seq:
                    strictly_increasing = False
                prev_seq = record.seq
                max_seq = max(max_seq, record.seq)
        return WALScan(max_seq=max_seq, strictly_increasing=strictly_increasing)

    def append(self, seq: int, msg_bytes: bytes) -> None:
        """Append one record and fsync it before returning."""
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
        start = time.perf_counter()
        self.flush()
        BM.bus_wal_fsync_seconds.observe(time.perf_counter() - start)
        BM.bus_wal_pending.inc()

    def flush(self) -> None:
        """Force any buffered WAL bytes to disk."""
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def advance_committed(self, seq: int) -> None:
        """Mark seq as confirmed-published. Atomic rename for crash safety."""
        if seq > self._committed:
            self._write_int_sidecar(self._watermark_path, seq)
            self._committed = seq
        BM.bus_wal_pending.dec()

    def mark_migrated(self) -> None:
        """Record that this dirty day-file has been conservatively drained."""
        self._write_int_sidecar(self._migrated_path, self._max_seq)
        self._needs_full_drain = False

    def _write_int_sidecar(self, path: Path, value: int) -> None:
        """Atomically write an integer sidecar and fsync its file content."""
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        tmp.write_text(str(value))
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _read_committed(self) -> int:
        """Read the durable committed watermark sidecar."""
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
        committed = -1 if self._needs_full_drain else self._committed
        with open(self._wal_path, "rb") as f:
            for line in f:
                try:
                    record = self._decode_line(line)
                except (ValueError, UnicodeDecodeError, Exception) as e:
                    LOG.warning("skipping corrupt WAL line in %s: %s", self._wal_path, e)
                    continue
                if record.seq > committed:
                    yield record

    def close(self) -> None:
        if not self._fp.closed:
            self.flush()
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
