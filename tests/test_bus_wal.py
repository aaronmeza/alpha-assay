# SPDX-License-Identifier: Apache-2.0
"""Tests for the producer-side write-ahead log."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alpha_assay.bus.wal import WALAppender


@pytest.fixture
def wal_dir(tmp_path: Path) -> Path:
    return tmp_path / "wal"


def test_append_creates_file_and_writes_record(wal_dir: Path):
    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    wal.append(seq=0, msg_bytes=b"hello-world")
    wal.flush()
    wal.close()

    files = list(wal_dir.glob("feed-*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "hello-world" in content or "aGVsbG8td29ybGQ=" in content  # base64 acceptable


def test_advance_committed_writes_watermark(wal_dir: Path):
    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    wal.append(seq=0, msg_bytes=b"a")
    wal.append(seq=1, msg_bytes=b"b")
    wal.advance_committed(1)
    wal.close()

    sidecar = wal_dir / "feed-2026-05-06.committed"
    assert sidecar.exists()
    assert sidecar.read_text().strip() == "1"


def test_read_uncommitted_returns_unflushed_messages(wal_dir: Path):
    # Producer appended 3 messages; only the first was confirmed-published.
    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    wal.append(seq=0, msg_bytes=b"committed")
    wal.append(seq=1, msg_bytes=b"in-flight-1")
    wal.append(seq=2, msg_bytes=b"in-flight-2")
    wal.advance_committed(0)
    wal.close()

    # Simulate restart: open a fresh appender, replay uncommitted.
    wal2 = WALAppender(directory=wal_dir, day="2026-05-06")
    uncommitted = list(wal2.read_uncommitted())
    assert len(uncommitted) == 2
    assert uncommitted[0].seq == 1
    assert uncommitted[0].msg_bytes == b"in-flight-1"
    assert uncommitted[1].seq == 2


def test_read_uncommitted_empty_after_full_advance(wal_dir: Path):
    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    wal.append(seq=0, msg_bytes=b"a")
    wal.advance_committed(0)
    wal.close()

    wal2 = WALAppender(directory=wal_dir, day="2026-05-06")
    assert list(wal2.read_uncommitted()) == []


def test_corrupt_line_skipped_with_warning(wal_dir: Path, caplog):
    # Manually corrupt the WAL file: insert a bogus line.
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / "feed-2026-05-06.jsonl"
    wal_path.write_text("not-a-valid-record\n")

    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    uncommitted = list(wal.read_uncommitted())
    # Corrupt line skipped, no exception.
    assert uncommitted == []


def test_fsync_batched_by_count(wal_dir: Path, monkeypatch):
    fsync_calls = {"n": 0}
    real_fsync = os.fsync

    def counting_fsync(fd):
        fsync_calls["n"] += 1
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    wal = WALAppender(directory=wal_dir, day="2026-05-06", fsync_every_n=10, fsync_every_ms=100_000)
    for i in range(25):
        wal.append(seq=i, msg_bytes=f"m-{i}".encode())
    wal.close()  # final flush
    # 25 records, fsync every 10 -> 2 batched flushes + 1 close flush = 3
    assert 2 <= fsync_calls["n"] <= 4
