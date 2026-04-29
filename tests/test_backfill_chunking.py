# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Unit tests for the backfill chunk-schedule + helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import click
import pandas as pd
import pytest

from alpha_assay.cli.backfill import (
    ChunkWindow,
    _format_ibkr_end_datetime,
    _group_bars_by_day,
    _is_in_rth_ct,
    _parse_duration_seconds,
    build_chunk_schedule,
)


def test_parse_duration_seconds_known_units():
    assert _parse_duration_seconds("1 D") == 86_400
    assert _parse_duration_seconds("1 W") == 7 * 86_400
    assert _parse_duration_seconds("2 D") == 2 * 86_400
    assert _parse_duration_seconds("3600 S") == 3600


def test_parse_duration_seconds_rejects_garbage():
    with pytest.raises(click.BadParameter):
        _parse_duration_seconds("garbage")
    with pytest.raises(click.BadParameter):
        _parse_duration_seconds("0 D")
    with pytest.raises(click.BadParameter):
        _parse_duration_seconds("1 X")


def test_format_ibkr_end_datetime_renders_utc_string():
    ts = datetime(2026, 4, 21, 14, 30, 0, tzinfo=UTC)
    assert _format_ibkr_end_datetime(ts) == "20260421 14:30:00 UTC"


def test_format_ibkr_end_datetime_normalizes_naive_to_utc():
    ts = datetime(2026, 4, 21, 14, 30, 0)
    assert _format_ibkr_end_datetime(ts) == "20260421 14:30:00 UTC"


def test_chunk_schedule_walks_backwards_and_covers_window():
    now = datetime(2026, 4, 21, 0, 0, 0, tzinfo=UTC)
    chunks = build_chunk_schedule(now=now, days=14, chunk_duration="1 W")

    assert len(chunks) == 2
    assert chunks[0].end == now
    assert chunks[0].start == now - timedelta(days=7)
    # Walks backwards: the second chunk's end is the first chunk's start.
    assert chunks[1].end == chunks[0].start
    assert chunks[1].start == now - timedelta(days=14)
    for c in chunks:
        assert c.duration_str == "1 W"


def test_chunk_schedule_overshoots_when_days_not_multiple():
    """A 10-day request with 1W chunks should produce 2 chunks; the
    older one extends past the requested left edge by design.
    """
    now = datetime(2026, 4, 21, 0, 0, 0, tzinfo=UTC)
    chunks = build_chunk_schedule(now=now, days=10, chunk_duration="1 W")
    assert len(chunks) == 2
    assert chunks[-1].start <= now - timedelta(days=10)


def test_chunk_schedule_180_day_default_is_about_26_chunks():
    now = datetime(2026, 4, 21, 0, 0, 0, tzinfo=UTC)
    chunks = build_chunk_schedule(now=now, days=180, chunk_duration="1 W")
    # 180 / 7 == 25.7, walked-backwards that rounds up to 26.
    assert len(chunks) == 26


def test_chunk_schedule_rejects_zero_or_negative_days():
    now = datetime(2026, 4, 21, 0, 0, 0, tzinfo=UTC)
    with pytest.raises(click.BadParameter):
        build_chunk_schedule(now=now, days=0, chunk_duration="1 W")
    with pytest.raises(click.BadParameter):
        build_chunk_schedule(now=now, days=-3, chunk_duration="1 W")


def test_group_bars_by_day_buckets_on_chicago_calendar():
    """Bars are bucketed by America/Chicago calendar day so the shard
    layout matches the recorder.

    All input bars are inside RTH (08:30-15:00 CT) on weekdays so the
    RTH gate added in the weekend-bars fix is a pass-through here -
    this test pins the per-day grouping invariant only.
    """
    bars = [
        # 13:31 UTC on Apr 21 (Tue) = 08:31 CT, in RTH.
        {
            "timestamp": pd.Timestamp("2026-04-21T13:31:00", tz="UTC"),
            "open": 2.0,
            "high": 2.0,
            "low": 2.0,
            "close": 2.0,
            "volume": 2,
        },
        # 19:59 UTC on Apr 21 (Tue) = 14:59 CT, last RTH minute.
        {
            "timestamp": pd.Timestamp("2026-04-21T19:59:00", tz="UTC"),
            "open": 4.0,
            "high": 4.0,
            "low": 4.0,
            "close": 4.0,
            "volume": 4,
        },
        # 13:30 UTC on Apr 22 (Wed) = 08:30 CT, RTH open.
        {
            "timestamp": pd.Timestamp("2026-04-22T13:30:00", tz="UTC"),
            "open": 3.0,
            "high": 3.0,
            "low": 3.0,
            "close": 3.0,
            "volume": 3,
        },
    ]
    grouped = _group_bars_by_day(bars)
    assert sorted(grouped.keys()) == ["2026-04-21", "2026-04-22"]
    assert len(grouped["2026-04-21"]) == 2
    assert len(grouped["2026-04-22"]) == 1


def test_is_in_rth_ct_drops_weekend_and_off_session_bars():
    """RTH gate: 08:30-15:00 CT on weekdays only.

    Sunday-evening Globex (17:00 CT) and Saturday CT bars are outside
    RTH and must be dropped so the backfill never writes a shard for a
    non-trading day. ES Globex re-opens at 17:00 CT Sunday and runs
    through 16:00 CT Friday with a 16:00-17:00 CT halt; the recorder's
    RTH gate keeps only 08:30-15:00 CT weekday bars.
    """
    # 22:00 UTC on Sun 2026-04-26 == 17:00 CT (Sunday-evening Globex
    # open). This is exactly the timestamp range that produced the
    # bogus 2026-04-26.parquet shard before the fix.
    sunday_globex = pd.Timestamp("2026-04-26T22:00:00", tz="UTC")
    assert _is_in_rth_ct(sunday_globex) is False

    # 18:00 UTC on Sat 2026-04-25 == 13:00 CT Saturday (no trading).
    saturday_ct = pd.Timestamp("2026-04-25T18:00:00", tz="UTC")
    assert _is_in_rth_ct(saturday_ct) is False

    # 03:00 UTC on Mon 2026-04-27 == 22:00 CT Sunday (still Sunday CT).
    sunday_late = pd.Timestamp("2026-04-27T03:00:00", tz="UTC")
    assert _is_in_rth_ct(sunday_late) is False

    # 13:30 UTC on Mon 2026-04-27 == 08:30 CT, RTH open.
    rth_open = pd.Timestamp("2026-04-27T13:30:00", tz="UTC")
    assert _is_in_rth_ct(rth_open) is True

    # 19:59 UTC on Mon 2026-04-27 == 14:59 CT, last RTH minute.
    rth_close = pd.Timestamp("2026-04-27T19:59:00", tz="UTC")
    assert _is_in_rth_ct(rth_close) is True

    # 20:00 UTC on Mon 2026-04-27 == 15:00 CT, exclusive end.
    after_rth = pd.Timestamp("2026-04-27T20:00:00", tz="UTC")
    assert _is_in_rth_ct(after_rth) is False

    # 13:29 UTC on Mon 2026-04-27 == 08:29 CT, before RTH.
    before_rth = pd.Timestamp("2026-04-27T13:29:00", tz="UTC")
    assert _is_in_rth_ct(before_rth) is False


def test_group_bars_by_day_drops_sunday_globex_open():
    """Bars from the Sunday-evening Globex re-open must not produce a
    Sunday shard.

    Pins the actual 2026-04-26 bug: 420 bars at 17:00-23:59 CT Sunday
    were landing in 2026-04-26.parquet. After the RTH filter, those
    bars are silently dropped (they are part of the upcoming Monday's
    Globex session by CME convention but are out of RTH for the
    recorder's purposes).
    """
    bars = [
        # 22:00 UTC Sun 2026-04-26 == 17:00 CT Sunday (Globex re-open).
        {
            "timestamp": pd.Timestamp("2026-04-26T22:00:00", tz="UTC"),
            "open": 7185.0,
            "high": 7191.0,
            "low": 7181.25,
            "close": 7182.5,
            "volume": 1504,
        },
        # 04:59 UTC Mon 2026-04-27 == 23:59 CT Sunday (still Sunday CT).
        {
            "timestamp": pd.Timestamp("2026-04-27T04:59:00", tz="UTC"),
            "open": 7200.75,
            "high": 7200.75,
            "low": 7200.5,
            "close": 7200.75,
            "volume": 27,
        },
        # 13:30 UTC Mon 2026-04-27 == 08:30 CT Monday RTH open. Kept.
        {
            "timestamp": pd.Timestamp("2026-04-27T13:30:00", tz="UTC"),
            "open": 7210.0,
            "high": 7211.0,
            "low": 7209.0,
            "close": 7210.5,
            "volume": 200,
        },
    ]
    grouped = _group_bars_by_day(bars)
    assert "2026-04-26" not in grouped  # no Sunday shard
    assert sorted(grouped.keys()) == ["2026-04-27"]
    assert len(grouped["2026-04-27"]) == 1


def test_group_bars_by_day_drops_friday_evening_through_sunday_session():
    """Friday 15:30 CT through Sunday 18:00 CT input produces only a
    Friday shard.

    Spans the full weekend gap CME observes (Fri 16:00 CT halt -> Sun
    17:00 CT re-open) plus a few minutes before / after. Only bars
    inside Friday's 08:30-15:00 CT RTH window survive the gate; every
    other timestamp is non-trading per the recorder's convention.
    """
    bars = [
        # Fri 14:59 CT (last RTH minute) - kept.
        {
            "timestamp": pd.Timestamp("2026-04-24T19:59:00", tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        },
        # Fri 15:30 CT - past RTH end, before the 16:00 CT Globex halt.
        {
            "timestamp": pd.Timestamp("2026-04-24T20:30:00", tz="UTC"),
            "open": 2.0,
            "high": 2.0,
            "low": 2.0,
            "close": 2.0,
            "volume": 2,
        },
        # Sat 12:00 CT - exchange closed; defensive gate test in case
        # IBKR or a fixture ever returns one.
        {
            "timestamp": pd.Timestamp("2026-04-25T17:00:00", tz="UTC"),
            "open": 3.0,
            "high": 3.0,
            "low": 3.0,
            "close": 3.0,
            "volume": 3,
        },
        # Sun 17:00 CT - Globex re-open, treated as Monday session.
        {
            "timestamp": pd.Timestamp("2026-04-26T22:00:00", tz="UTC"),
            "open": 4.0,
            "high": 4.0,
            "low": 4.0,
            "close": 4.0,
            "volume": 4,
        },
        # Sun 18:00 CT - first hour of the Sunday-evening session.
        {
            "timestamp": pd.Timestamp("2026-04-26T23:00:00", tz="UTC"),
            "open": 5.0,
            "high": 5.0,
            "low": 5.0,
            "close": 5.0,
            "volume": 5,
        },
    ]
    grouped = _group_bars_by_day(bars)
    # Fri 2026-04-24 is the only date in the span that contains an
    # in-RTH bar; Sat / Sun bars are dropped wholesale.
    assert sorted(grouped.keys()) == ["2026-04-24"]
    assert len(grouped["2026-04-24"]) == 1


def test_group_bars_by_day_drops_saturday_ct_bars():
    """Bars whose CT timestamp is on Saturday must never produce a
    Saturday shard. ES does not trade Saturday CT.
    """
    bars = [
        # 09:00 UTC Sat 2026-04-25 == 04:00 CT Saturday.
        {
            "timestamp": pd.Timestamp("2026-04-25T09:00:00", tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        },
        # 21:59 UTC Sat 2026-04-25 == 16:59 CT Saturday.
        {
            "timestamp": pd.Timestamp("2026-04-25T21:59:00", tz="UTC"),
            "open": 2.0,
            "high": 2.0,
            "low": 2.0,
            "close": 2.0,
            "volume": 2,
        },
    ]
    grouped = _group_bars_by_day(bars)
    assert grouped == {}


def test_chunk_window_is_a_frozen_dataclass():
    """ChunkWindow is exposed as part of the CLI public surface; lock
    its shape so refactors don't break import-site consumers.
    """
    cw = ChunkWindow(
        end=datetime(2026, 4, 21, tzinfo=UTC),
        start=datetime(2026, 4, 14, tzinfo=UTC),
        duration_str="1 W",
    )
    with pytest.raises(FrozenInstanceError):
        cw.end = datetime.now(tz=UTC)  # type: ignore[misc]
