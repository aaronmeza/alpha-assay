# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Incremental joined-frame construction for the paper runner.

Builds, minute by minute, the same frame shape that
`alpha_assay.data.joined_loader.load_es_with_breadth` produces for
backtests:

    DatetimeIndex name='timestamp' tz='America/Chicago', columns:
    [open, high, low, close, volume, TICK, ADD]

Inputs arrive as bus messages: the primary instrument as completed
1-min bars, breadth indices as raw ticks. `MinuteCloseAggregator`
mirrors the recorder's causal aggregation (a minute's close is known
only when a tick with a later minute arrives - no wall-clock heuristics)
but keeps just the close, which is the only breadth field the joined
frame carries. `SessionFrameBuilder` joins the three per-minute series
and appends a row exactly once, when all components for that minute are
present, in strictly increasing timestamp order - so a strategy
evaluating the frame's last row can never see a partial or out-of-order
bar (lookahead-safe by construction).
"""

from __future__ import annotations

import logging

import pandas as pd

from alpha_assay.filters.session_mask import session_mask

_LOG = logging.getLogger(__name__)

FRAME_TZ = "America/Chicago"
_ES_COLUMNS = ("open", "high", "low", "close", "volume")
FRAME_COLUMNS = (*_ES_COLUMNS, "TICK", "ADD")


def _minute_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize any tz-aware timestamp to its UTC minute floor."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").floor("1min")


def _in_rth(ts: pd.Timestamp) -> bool:
    """True iff `ts` falls inside US-equity core hours (recorder parity).

    Uses the canonical session bounds from `filters.session_mask` with
    zero trim so the live frame matches the recorder shards that feed
    `joined_loader` (the recorders drop out-of-RTH ticks before
    aggregation).
    """
    idx = pd.DatetimeIndex([ts])
    return bool(session_mask(idx, minutes_after_open=0, minutes_before_close=0).iloc[0])


class MinuteCloseAggregator:
    """Causal per-minute close aggregation for one breadth feed.

    `ingest(ts, value)` returns `(minute_utc, close)` when the incoming
    tick's minute is later than the in-flight bucket's (the bucket is
    then complete), else None. Out-of-RTH ticks are dropped; ticks older
    than the in-flight minute are ignored defensively (same policy as
    the recorder).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._minute: pd.Timestamp | None = None
        self._close: float = 0.0

    def ingest(self, ts: pd.Timestamp, value: float) -> tuple[pd.Timestamp, float] | None:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if not _in_rth(ts):
            return None
        minute = _minute_utc(ts)
        emitted: tuple[pd.Timestamp, float] | None = None
        if self._minute is None:
            self._minute = minute
        elif minute > self._minute:
            emitted = (self._minute, self._close)
            self._minute = minute
        elif minute < self._minute:
            # Out-of-order straggler; the feed normally serializes by time.
            _LOG.warning("%s: dropping out-of-order tick at %s (bucket %s)", self._name, ts, self._minute)
            return None
        self._close = float(value)
        return emitted


class SessionFrameBuilder:
    """Joins per-minute ES bars + breadth closes into the canonical frame.

    A row appends exactly once, when its ES bar and every breadth close
    are present, and only in strictly increasing timestamp order. The
    frame resets on session-date rollover (America/Chicago calendar day)
    so rolling-window strategy state never bleeds across sessions -
    matching the per-session shards a backtest run loads.
    """

    def __init__(self, breadth_columns: tuple[str, ...] = ("TICK", "ADD")) -> None:
        self._breadth_columns = tuple(breadth_columns)
        self._pending: dict[pd.Timestamp, dict[str, object]] = {}
        self._rows: list[dict[str, object]] = []
        self._index: list[pd.Timestamp] = []
        self._last_appended: pd.Timestamp | None = None
        self._session_date: object | None = None

    @property
    def frame(self) -> pd.DataFrame:
        """Current joined frame. Empty (correctly typed) before first row."""
        if not self._rows:
            idx = pd.DatetimeIndex([], tz=FRAME_TZ, name="timestamp")
            return pd.DataFrame({c: pd.Series(dtype=float) for c in FRAME_COLUMNS}, index=idx)
        idx = pd.DatetimeIndex(self._index, name="timestamp")
        return pd.DataFrame(self._rows, index=idx, columns=list(FRAME_COLUMNS))

    def add_es_bar(self, bar: dict[str, object]) -> list[pd.Timestamp]:
        """Ingest one completed ES 1-min bar (canonical bus-shaped dict
        with `timestamp` + OHLCV). Returns timestamps of any rows that
        completed as a result, in append order.
        """
        minute = _minute_utc(pd.Timestamp(bar["timestamp"]))
        slot = self._pending.setdefault(minute, {})
        for col in _ES_COLUMNS:
            slot[col] = float(bar[col])  # type: ignore[arg-type]
        return self._flush_completed()

    def add_breadth_close(self, column: str, minute_utc: pd.Timestamp, close: float) -> list[pd.Timestamp]:
        """Ingest one completed breadth minute close for `column`
        (e.g. "TICK" / "ADD"). Returns timestamps of completed rows.
        """
        if column not in self._breadth_columns:
            raise ValueError(f"unknown breadth column {column!r}; expected one of {self._breadth_columns}")
        slot = self._pending.setdefault(_minute_utc(minute_utc), {})
        slot[column] = float(close)
        return self._flush_completed()

    # --- internals --------------------------------------------------------

    def _is_complete(self, slot: dict[str, object]) -> bool:
        return all(c in slot for c in _ES_COLUMNS) and all(c in slot for c in self._breadth_columns)

    def _flush_completed(self) -> list[pd.Timestamp]:
        """Append every completed pending minute, oldest first, keeping
        the index strictly increasing. A completed minute older than the
        newest appended row is dropped loudly rather than appended out of
        order (the strategy contract requires monotonic timestamps).
        """
        appended: list[pd.Timestamp] = []
        for minute in sorted(self._pending):
            slot = self._pending[minute]
            if not self._is_complete(slot):
                continue
            del self._pending[minute]
            if self._last_appended is not None and minute <= self._last_appended:
                _LOG.warning("frame: dropping late row %s (frame already at %s)", minute, self._last_appended)
                continue
            local = minute.tz_convert(FRAME_TZ)
            if self._session_date is not None and local.date() != self._session_date:
                # Session rollover: start a fresh frame for the new day.
                self._rows.clear()
                self._index.clear()
            self._session_date = local.date()
            self._rows.append({c: slot[c] for c in FRAME_COLUMNS})
            self._index.append(local)
            self._last_appended = minute
            appended.append(local)
        if self._last_appended is not None:
            # Prune incomplete slots the frame has moved past (e.g. an ES
            # bar whose breadth minute never materialized) so the pending
            # map cannot grow unbounded across a long session.
            for minute in [m for m in self._pending if m <= self._last_appended]:
                del self._pending[minute]
        return appended
