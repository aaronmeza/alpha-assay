# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Unit tests for the incremental joined-frame builder.

Covers the causal aggregation contract (a breadth minute closes only
when a later-minute tick arrives), the join-completeness rule (a row
appends only when ES bar + every breadth close are present), strict
index monotonicity, and the session-date reset.
"""

from __future__ import annotations

import pandas as pd

from alpha_assay.paper.frame import FRAME_COLUMNS, MinuteCloseAggregator, SessionFrameBuilder

# A mid-session weekday minute (Tuesday 10:00 CT == 16:00 UTC).
T0 = pd.Timestamp("2026-06-02 16:00:00", tz="UTC")


def _bar(minute: pd.Timestamp, close: float = 5000.0) -> dict:
    return {
        "timestamp": minute,
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 2.0,
        "close": close,
        "volume": 100,
    }


# --- MinuteCloseAggregator ----------------------------------------------


def test_aggregator_emits_close_on_minute_rollover():
    agg = MinuteCloseAggregator("TICK-NYSE")

    assert agg.ingest(T0, 100.0) is None
    assert agg.ingest(T0 + pd.Timedelta(seconds=30), 250.0) is None
    emitted = agg.ingest(T0 + pd.Timedelta(minutes=1), -50.0)

    assert emitted == (T0, 250.0)  # last value in the minute is the close


def test_aggregator_skips_gap_minutes_without_fabricating_closes():
    """A 3-minute tick gap yields exactly one close (the pre-gap minute);
    the empty minutes produce nothing (inner-join parity with the
    backtest loader, which drops breadth-less minutes)."""
    agg = MinuteCloseAggregator("AD-NYSE")

    agg.ingest(T0, 10.0)
    emitted = agg.ingest(T0 + pd.Timedelta(minutes=3), 40.0)

    assert emitted == (T0, 10.0)


def test_aggregator_drops_out_of_rth_ticks():
    agg = MinuteCloseAggregator("TICK-NYSE")

    # 05:00 CT is pre-open; the recorder drops these before aggregation.
    pre_open = pd.Timestamp("2026-06-02 11:00:00", tz="UTC")
    assert agg.ingest(pre_open, 500.0) is None
    # The pre-open tick must not have started a bucket.
    assert agg.ingest(T0, 100.0) is None
    assert agg.ingest(T0 + pd.Timedelta(minutes=1), 200.0) == (T0, 100.0)


def test_aggregator_ignores_out_of_order_ticks():
    agg = MinuteCloseAggregator("TICK-NYSE")

    agg.ingest(T0 + pd.Timedelta(minutes=2), 100.0)
    assert agg.ingest(T0, 999.0) is None  # straggler: dropped
    emitted = agg.ingest(T0 + pd.Timedelta(minutes=3), 300.0)
    assert emitted == (T0 + pd.Timedelta(minutes=2), 100.0)


# --- SessionFrameBuilder --------------------------------------------------


def test_row_appends_only_when_all_components_present():
    builder = SessionFrameBuilder()

    assert builder.add_es_bar(_bar(T0)) == []
    assert builder.add_breadth_close("TICK", T0, 150.0) == []
    completed = builder.add_breadth_close("ADD", T0, 900.0)

    assert completed == [T0.tz_convert("America/Chicago")]
    frame = builder.frame
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert str(frame.index.tz) == "America/Chicago"
    assert frame.index.name == "timestamp"
    row = frame.iloc[0]
    assert row["close"] == 5000.0
    assert row["TICK"] == 150.0
    assert row["ADD"] == 900.0


def test_rows_append_in_timestamp_order_across_arrival_orderings():
    """Minute 1's components arriving before minute 0 completes must not
    produce an out-of-order frame: rows flush oldest-first."""
    builder = SessionFrameBuilder()
    t1 = T0 + pd.Timedelta(minutes=1)

    builder.add_es_bar(_bar(T0))
    builder.add_es_bar(_bar(t1, close=5010.0))
    builder.add_breadth_close("TICK", t1, 50.0)
    builder.add_breadth_close("ADD", t1, 100.0)
    # Minute 1 completed first, so it appended first; minute 0 completing
    # afterwards is a late straggler and must be dropped, not inserted
    # out of order.
    builder.add_breadth_close("TICK", T0, 25.0)
    completed = builder.add_breadth_close("ADD", T0, 75.0)

    frame = builder.frame
    assert frame.index.is_monotonic_increasing
    assert completed == []  # late row dropped, never appended out of order


def test_empty_frame_is_typed():
    builder = SessionFrameBuilder()
    frame = builder.frame
    assert frame.empty
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert str(frame.index.tz) == "America/Chicago"


def test_frame_resets_on_session_date_rollover():
    builder = SessionFrameBuilder()
    next_day = T0 + pd.Timedelta(days=1)

    builder.add_es_bar(_bar(T0))
    builder.add_breadth_close("TICK", T0, 1.0)
    builder.add_breadth_close("ADD", T0, 2.0)
    assert len(builder.frame) == 1

    builder.add_es_bar(_bar(next_day))
    builder.add_breadth_close("TICK", next_day, 3.0)
    builder.add_breadth_close("ADD", next_day, 4.0)

    frame = builder.frame
    assert len(frame) == 1  # prior session dropped
    assert frame.index[0] == next_day.tz_convert("America/Chicago")


def test_unknown_breadth_column_rejected():
    builder = SessionFrameBuilder()
    try:
        builder.add_breadth_close("VOLD", T0, 1.0)
    except ValueError as exc:
        assert "VOLD" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown breadth column")
