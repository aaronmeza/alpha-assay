# SPDX-License-Identifier: Apache-2.0
"""Tests for the producer-side write-ahead log."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alpha_assay.bus import metrics as BM
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


def test_every_append_is_durable(wal_dir: Path, monkeypatch):
    fsync_calls = {"n": 0}
    real_fsync = os.fsync

    def counting_fsync(fd):
        """Count fsync calls while preserving real durability semantics."""
        fsync_calls["n"] += 1
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    append_count = 25
    for i in range(append_count):
        wal.append(seq=i, msg_bytes=f"m-{i}".encode())

    assert fsync_calls["n"] >= append_count


def _hard_kill_wal_writer(wal_dir: Path, day: str, body: str) -> int:
    """Drive WAL calls in a child interpreter that dies WITHOUT unwinding.

    ``os._exit()`` skips finally-blocks, atexit handlers, and - the point here -
    the flush of Python's userspace write buffer. That makes this a true hard
    kill (SIGKILL / OOM / power loss), which an in-process test cannot simulate:
    anything that lets the interpreter tear down cleanly flushes the buffer and
    hides the very bug under test.

    ``body`` runs against a ``wal`` already bound to *wal_dir* / *day*. The
    child imports from this checkout's ``src`` so it exercises the code under
    test, not whatever an editable install happens to point at.
    """
    src_path = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import os\n"
        "from pathlib import Path\n"
        "from alpha_assay.bus.wal import WALAppender\n"
        f"wal = WALAppender(directory=Path({str(wal_dir)!r}), day={day!r})\n"
        f"{body}\n"
        "os._exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(src_path)},
        check=False,
    )
    return result.returncode


def test_appended_record_survives_hard_kill_without_flush_or_close(tmp_path: Path):
    """A record appended but never flushed/closed is still replayable after a hard kill.

    This is the whole point of a write-AHEAD log, and it is what the batched
    fsync silently broke: the record died in the write buffer, was never
    published, and no replay could recover it.
    """
    wal_dir = tmp_path / "wal"

    returncode = _hard_kill_wal_writer(
        wal_dir,
        "2026-05-06",
        "wal.append(seq=0, msg_bytes=b'hard-kill-survivor')",
    )

    assert returncode == 1
    restarted = WALAppender(directory=wal_dir, day="2026-05-06")
    uncommitted = list(restarted.read_uncommitted())
    assert [(record.seq, record.msg_bytes) for record in uncommitted] == [(0, b"hard-kill-survivor")]


def test_committed_never_exceeds_max_seq_after_hard_kill(tmp_path: Path):
    """The watermark can never outrun the log, even when the producer is hard-killed.

    Driving this through a hard kill is what gives it teeth. Closing the WAL
    normally flushes the day-file, so an in-process append/advance/close
    sequence upholds ``committed <= max_seq`` even on the batched-fsync code -
    the assertion passes for the wrong reason and proves nothing.

    Killed mid-flight, the batched-fsync producer left the eagerly-fsynced
    watermark pointing past a day-file whose tail was still buffered. That is
    the exact state measured on the production host (watermark 1349 vs max_seq
    1340) and the one the seq cursor then has to defend against. An fsync on
    every append makes it unreachable.
    """
    day = "2026-05-06"
    wal_dir = tmp_path / "wal"
    body = "\n".join(
        f"wal.append(seq={seq}, msg_bytes=b'published-{seq}')\nwal.advance_committed({seq})" for seq in range(5)
    )

    returncode = _hard_kill_wal_writer(wal_dir, day, body)

    assert returncode == 1
    restarted = WALAppender(directory=wal_dir, day=day)
    assert restarted.committed == 4
    assert restarted.committed <= restarted.max_seq


def test_legacy_committed_above_max_seq_warns_and_seeds_above_both(wal_dir: Path, caplog):
    """A legacy watermark-ahead file stays open and seeds above both cursors."""
    day = "2026-05-06"
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / f"feed-{day}.jsonl"
    lines = [f"{seq}\t{base64.b64encode(f'flushed-{seq}'.encode()).decode()}\n" for seq in range(5)]
    wal_path.write_text("".join(lines))
    (wal_dir / f"feed-{day}.committed").write_text("9")

    wal = WALAppender(directory=wal_dir, day=day)

    assert "committed watermark exceeds day-file max seq" in caplog.text
    assert wal.next_seq == 10


def test_same_day_restart_keeps_seq_monotonic_and_replays_only_unpublished(wal_dir: Path):
    """A same-day restart continues seqs and replays only records above the watermark."""
    day = "2026-07-01"
    first_count = 5
    second_count = 4

    wal = WALAppender(directory=wal_dir, day=day)
    assert wal.next_seq == 0
    seq_counter = wal.next_seq
    for _ in range(first_count):
        wal.append(seq=seq_counter, msg_bytes=f"first-{seq_counter}".encode())
        wal.advance_committed(seq_counter)
        seq_counter += 1
    wal.close()

    restarted = WALAppender(directory=wal_dir, day=day)
    assert list(restarted.read_uncommitted()) == []
    assert restarted.next_seq == first_count
    seq_counter = restarted.next_seq
    unpublished_seq = seq_counter + second_count - 1
    for idx in range(second_count):
        restarted.append(seq=seq_counter, msg_bytes=f"second-{idx}".encode())
        if seq_counter != unpublished_seq:
            restarted.advance_committed(seq_counter)
        seq_counter += 1
    restarted.close()

    restarted_again = WALAppender(directory=wal_dir, day=day)
    uncommitted = list(restarted_again.read_uncommitted())
    assert [record.seq for record in uncommitted] == [unpublished_seq]
    assert [record.msg_bytes for record in uncommitted] == [b"second-3"]

    seqs = []
    wal_path = wal_dir / f"feed-{day}.jsonl"
    for line in wal_path.read_text().splitlines():
        seq_str, _ = line.split("\t", 1)
        seqs.append(int(seq_str))
    assert len(seqs) == len(set(seqs))


def test_advance_committed_never_moves_watermark_backwards(wal_dir: Path):
    """A stale advance call cannot regress the durable watermark sidecar."""
    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    wal.advance_committed(10)
    wal.advance_committed(3)
    wal.close()

    sidecar = wal_dir / "feed-2026-05-06.committed"
    assert sidecar.read_text().strip() == "10"


def test_dirty_duplicate_segment_replays_unpublished_below_watermark(wal_dir: Path):
    """A dirty legacy file must not lose an unpublished duplicate seq."""
    day = "2026-07-13"
    wal = WALAppender(directory=wal_dir, day=day)
    for seq in range(5):
        wal.append(seq=seq, msg_bytes=f"segment-1-{seq}".encode())
        wal.advance_committed(seq)
    wal.close()

    dirty_writer = WALAppender(directory=wal_dir, day=day)
    dirty_writer.append(seq=0, msg_bytes=b"segment-2-published-0")
    dirty_writer.append(seq=1, msg_bytes=b"segment-2-published-1")
    dirty_writer.append(seq=2, msg_bytes=b"segment-2-unpublished-2")
    dirty_writer.close()

    restarted = WALAppender(directory=wal_dir, day=day)
    replayed = [record.msg_bytes for record in restarted.read_uncommitted()]

    assert b"segment-2-unpublished-2" in replayed


def test_dirty_day_file_full_drains_exactly_once(wal_dir: Path):
    """The dirty-file migration drains once, then the marker restores cursor replay."""
    day = "2026-07-13"
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / f"feed-{day}.jsonl"
    lines = [f"{seq}\t{base64.b64encode(f'segment-1-{seq}'.encode()).decode()}\n" for seq in range(5)]
    lines.extend(f"{seq}\t{base64.b64encode(f'segment-2-{seq}'.encode()).decode()}\n" for seq in range(3))
    wal_path.write_text("".join(lines))
    (wal_dir / f"feed-{day}.committed").write_text("4")

    wal = WALAppender(directory=wal_dir, day=day)
    assert wal.needs_full_drain is True
    replayed = list(wal.read_uncommitted())
    assert [record.seq for record in replayed] == [0, 1, 2, 3, 4, 0, 1, 2]
    for record in replayed:
        wal.advance_committed(record.seq)
    wal.mark_migrated()
    wal.close()

    restarted = WALAppender(directory=wal_dir, day=day)
    assert restarted.needs_full_drain is False
    assert list(restarted.read_uncommitted()) == []


def test_post_migration_unpublished_record_is_still_replayed(wal_dir: Path):
    """A record appended AFTER migration, then lost to a crash, is still recovered.

    Adversarial-review claim (round 2): once ``.migrated`` exists, the dirty
    file falls back to the scalar watermark cursor while its duplicate seqs are
    still on disk - so a post-migration record that is appended and then lost to
    a crash before its publish would be skipped forever.

    It is not, and this test pins down why. The full drain advances the
    watermark for EVERY record it replays, so at migration the watermark equals
    the file's max seq. Live appends then resume at ``next_seq`` == max + 1, i.e.
    strictly ABOVE the watermark. So every post-migration record - published or
    not - sits above the cursor, and an unpublished one is always replayed. The
    legacy duplicates below the watermark were all published by the full drain.
    The scalar cursor is therefore sound for everything written after migration,
    even though the file itself never becomes strictly increasing again.
    """
    day = "2026-07-13"
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / f"feed-{day}.jsonl"

    # A legacy dirty file: segment 1 wrote 0..4, segment 2 restarted at 0.
    lines = [f"{seq}\t{base64.b64encode(f'seg1-{seq}'.encode()).decode()}\n" for seq in range(5)]
    lines.extend(f"{seq}\t{base64.b64encode(f'seg2-{seq}'.encode()).decode()}\n" for seq in range(3))
    wal_path.write_text("".join(lines))
    (wal_dir / f"feed-{day}.committed").write_text("4")

    # The migrating process full-drains, then starts publishing live.
    wal = WALAppender(directory=wal_dir, day=day)
    assert wal.needs_full_drain is True
    for record in wal.read_uncommitted():
        wal.advance_committed(record.seq)
    wal.mark_migrated()

    # The drain leaves the watermark at the file's high-water mark, which is
    # the invariant the whole argument rests on.
    assert (wal_dir / f"feed-{day}.committed").read_text().strip() == "4"

    # Live loop resumes above it, publishes one record, then dies mid-record:
    # seq 6 is appended but never published (no advance_committed).
    seq_counter = wal.next_seq
    assert seq_counter == 5
    wal.append(seq=5, msg_bytes=b"post-migration-published")
    wal.advance_committed(5)
    wal.append(seq=6, msg_bytes=b"post-migration-LOST")
    wal.close()

    # Restart. The file is still not strictly increasing and the marker is
    # present, so it uses the scalar cursor - and still recovers the lost record.
    restarted = WALAppender(directory=wal_dir, day=day)
    assert restarted.needs_full_drain is False
    replayed = [record.msg_bytes for record in restarted.read_uncommitted()]
    assert replayed == [b"post-migration-LOST"]


def test_marker_is_not_written_when_the_drain_fails_midway(wal_dir: Path):
    """A drain that dies partway leaves no marker, so the next boot re-drains.

    Adversarial-review claim (round 2): the migration marker is not atomic with
    the replay, so a crash could leave ``.migrated`` behind while records were
    still unpublished, permanently hiding them.

    It cannot. ``mark_migrated()`` runs only AFTER the drain loop completes, and
    the loop publishes every record before it advances. A publish that raises
    propagates out before the marker is ever written, so the next boot still
    sees a dirty, unmigrated file and conservatively drains it again. The
    failure mode is a repeated over-replay, never a skipped record.
    """
    day = "2026-07-13"
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / f"feed-{day}.jsonl"
    lines = [f"{seq}\t{base64.b64encode(f'seg1-{seq}'.encode()).decode()}\n" for seq in range(3)]
    lines.extend(f"{seq}\t{base64.b64encode(f'seg2-{seq}'.encode()).decode()}\n" for seq in range(2))
    wal_path.write_text("".join(lines))
    (wal_dir / f"feed-{day}.committed").write_text("2")

    wal = WALAppender(directory=wal_dir, day=day)
    assert wal.needs_full_drain is True

    # Simulate the publish blowing up on the third record, the way a Redis
    # outage would take the drain loop down before mark_migrated() is reached.
    published: list[bytes] = []
    with pytest.raises(RuntimeError):
        for i, record in enumerate(wal.read_uncommitted()):
            if i == 2:
                raise RuntimeError("redis is down")
            published.append(record.msg_bytes)
            wal.advance_committed(record.seq)
    wal.close()

    assert not (wal_dir / f"feed-{day}.migrated").exists()

    # Next boot: still dirty, still unmigrated, so the whole file replays again.
    restarted = WALAppender(directory=wal_dir, day=day)
    assert restarted.needs_full_drain is True
    assert len(list(restarted.read_uncommitted())) == 5


def test_clean_day_file_uses_watermark_cursor_without_full_drain(wal_dir: Path):
    """A strictly increasing file only replays records above the watermark."""
    day = "2026-07-13"
    wal = WALAppender(directory=wal_dir, day=day)
    for seq in range(5):
        wal.append(seq=seq, msg_bytes=f"clean-{seq}".encode())
        if seq <= 2:
            wal.advance_committed(seq)
    wal.close()

    restarted = WALAppender(directory=wal_dir, day=day)
    assert restarted.needs_full_drain is False
    assert [record.seq for record in restarted.read_uncommitted()] == [3, 4]


def test_seeded_same_day_restarts_remain_strictly_increasing(wal_dir: Path):
    """A file written by fixed producers across restarts is never dirty."""
    day = "2026-07-13"

    first = WALAppender(directory=wal_dir, day=day)
    seq_counter = first.next_seq
    for _ in range(3):
        first.append(seq=seq_counter, msg_bytes=f"first-{seq_counter}".encode())
        first.advance_committed(seq_counter)
        seq_counter += 1
    first.close()

    second = WALAppender(directory=wal_dir, day=day)
    assert second.needs_full_drain is False
    seq_counter = second.next_seq
    for _ in range(3):
        second.append(seq=seq_counter, msg_bytes=f"second-{seq_counter}".encode())
        second.advance_committed(seq_counter)
        seq_counter += 1
    second.close()

    restarted = WALAppender(directory=wal_dir, day=day)
    assert restarted.needs_full_drain is False
    assert list(restarted.read_uncommitted()) == []


def test_next_seq_is_zero_for_absent_file_and_max_plus_one_for_populated_file(wal_dir: Path):
    """The caller can seed live seqs from the existing day-file high-water mark."""
    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    assert wal.next_seq == 0
    wal.append(seq=3, msg_bytes=b"late")
    wal.append(seq=7, msg_bytes=b"latest")
    wal.close()

    restarted = WALAppender(directory=wal_dir, day="2026-05-06")
    assert restarted.next_seq == 8


def test_next_seq_clears_a_watermark_that_outran_the_log(wal_dir: Path):
    """A watermark ahead of the last durable record must not be re-used.

    The watermark sidecar is fsynced on every advance_committed(); the day-file
    is fsynced in batches. So a hard kill drops the buffered tail and leaves a
    committed value HIGHER than any seq on disk. This is not hypothetical - it
    was measured on the live feed on 2026-07-13, where the ES bars WAL sat at
    watermark=1349 with max_seq=1340 because the last nine published bars were
    still in the write buffer.

    Seeding from the log alone hands back a seq the watermark has already
    passed. Combined with the monotonic advance_committed guard, that record's
    advance is a no-op and an unpublished one is filtered out of the replay as
    seq <= committed - the exact under-replay this branch exists to prevent,
    arriving through a different door.
    """
    day = "2026-07-13"
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / f"feed-{day}.jsonl"

    # Five records made it to disk; the watermark says nine were published,
    # because records 5..9 were still buffered when the process was killed.
    lines = [f"{seq}\t{base64.b64encode(f'flushed-{seq}'.encode()).decode()}\n" for seq in range(5)]
    wal_path.write_text("".join(lines))
    (wal_dir / f"feed-{day}.committed").write_text("9")

    wal = WALAppender(directory=wal_dir, day=day)
    # The file is strictly increasing, so this takes the ordinary cursor path -
    # the dirty-file full drain is not what saves us here.
    assert wal.needs_full_drain is False
    # Not 5: that would collide with a watermark that has already reached 9.
    assert wal.next_seq == 10

    # Prove the record stays recoverable: append at the seeded seq, then die
    # before publishing it (no advance_committed).
    wal.append(seq=wal.next_seq, msg_bytes=b"appended-never-published")
    wal.close()

    restarted = WALAppender(directory=wal_dir, day=day)
    replayed = [record.msg_bytes for record in restarted.read_uncommitted()]
    assert replayed == [b"appended-never-published"]


def test_next_seq_ignores_corrupt_lines(wal_dir: Path):
    """Corrupt day-file lines do not stop seq discovery."""
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / "feed-2026-05-06.jsonl"
    good_line = f"4\t{base64.b64encode(b'ok').decode()}\n"
    wal_path.write_text(f"not-a-valid-record\n{good_line}")

    wal = WALAppender(directory=wal_dir, day="2026-05-06")
    assert wal.next_seq == 5


def test_pending_gauge_never_goes_negative_when_replay_is_seeded(wal_dir: Path):
    """Replay drain decrements balance a caller-seeded pending count."""
    start_value = BM.bus_wal_pending._value.get()
    BM.bus_wal_pending._value.set(0)
    try:
        wal = WALAppender(directory=wal_dir, day="2026-05-06")
        wal.append(seq=0, msg_bytes=b"unpublished-0")
        wal.append(seq=1, msg_bytes=b"unpublished-1")
        wal.close()
        BM.bus_wal_pending._value.set(0)

        restarted = WALAppender(directory=wal_dir, day="2026-05-06")
        BM.bus_wal_pending.inc(2)
        seen_values = []
        for record in restarted.read_uncommitted():
            seen_values.append(BM.bus_wal_pending._value.get())
            restarted.advance_committed(record.seq)
            seen_values.append(BM.bus_wal_pending._value.get())

        assert seen_values
        assert min(seen_values) >= 0
        assert BM.bus_wal_pending._value.get() == 0
    finally:
        BM.bus_wal_pending._value.set(start_value)


def test_legacy_dirty_day_file_recovers_to_a_monotonic_cursor(wal_dir: Path):
    """A day-file already poisoned by the l6f bug converges to a sound cursor.

    This is the migration case, not a hypothetical: every day-file on the box
    was written by the buggy producer, so it holds DUPLICATE seqs (one 0..N
    block per process segment) with a watermark that regressed below the true
    max. The fix must (a) seed the next seq from the real high-water mark
    rather than from a duplicate, (b) replay the above-watermark tail exactly
    once, and (c) never write another duplicate seq afterwards - so the file
    heals on the first restart instead of staying poisoned.
    """
    day = "2026-07-13"
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / f"feed-{day}.jsonl"

    # Segment 1 wrote seqs 0..4, then the process restarted and segment 2
    # started over at 0 (the bug) and got as far as seq 2 before dying.
    lines = [f"{seq}\tc2VnMS0=\n" for seq in range(5)] + [f"{seq}\tc2VnMi0=\n" for seq in range(3)]
    wal_path.write_text("".join(lines))
    # The watermark regressed to segment 2's last advance_committed(2).
    (wal_dir / f"feed-{day}.committed").write_text("2")

    wal = WALAppender(directory=wal_dir, day=day)

    # (a) Seeded from the true max seq (4) across BOTH segments, not from
    # segment 2's trailing 2 - which is what let the counter collide before.
    assert wal.next_seq == 5

    # (b) The dirty file full-drains once so duplicate-seq records below the
    # watermark cannot be silently skipped.
    replayed = [record.seq for record in wal.read_uncommitted()]
    assert replayed == [0, 1, 2, 3, 4, 0, 1, 2]
    for seq in replayed:
        wal.advance_committed(seq)
    wal.mark_migrated()

    # (c) Live appends resume above the high-water mark, so no NEW duplicate
    # seq is written and the watermark only ever moves forward.
    seq_counter = wal.next_seq
    for _ in range(3):
        wal.append(seq=seq_counter, msg_bytes=b"post-fix")
        wal.advance_committed(seq_counter)
        seq_counter += 1
    wal.close()

    # The file has healed: a further restart replays nothing at all.
    healed = WALAppender(directory=wal_dir, day=day)
    assert list(healed.read_uncommitted()) == []
    assert healed.next_seq == 8
    assert (wal_dir / f"feed-{day}.committed").read_text().strip() == "7"

    # Only the pre-existing duplicates remain; the fix added none.
    seqs = [int(line.split("\t", 1)[0]) for line in wal_path.read_text().splitlines()]
    assert seqs == [0, 1, 2, 3, 4, 0, 1, 2, 5, 6, 7]
