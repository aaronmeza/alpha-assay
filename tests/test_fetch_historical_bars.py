# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/fetch_historical_bars.py.

All tests use a fake IB object — no network, no gateway. Coverage:
shard write/resume logic, RTH + session filtering, recorder-schema
dtypes, the quarterly roll schedule, and retry/no-data error handling.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from ib_insync import RequestError

from scripts import fetch_historical_bars as mod

LOG = logging.getLogger("test_fetch_historical_bars")


def _bar(ts: datetime, price: float = 100.0, volume: int = 10) -> SimpleNamespace:
    return SimpleNamespace(date=ts, open=price, high=price + 0.5, low=price - 0.5, close=price + 0.25, volume=volume)


def _session_bars(session: date, n: int = 5) -> list[SimpleNamespace]:
    """n bars starting at 08:30 CT (13:30 UTC during CDT) on ``session``."""
    start = datetime(session.year, session.month, session.day, 13, 30, tzinfo=UTC)
    return [_bar(start + timedelta(minutes=i)) for i in range(n)]


class FakeIB:
    """Minimal stand-in for ib_insync.IB used by run_fetch."""

    def __init__(self, bars_by_end=None, errors=None, qualify_ok=True):
        self.bars_by_end = bars_by_end or {}
        self.errors = list(errors or [])
        self.qualify_ok = qualify_ok
        self.qualify_calls: list[str] = []
        self.request_ends: list[str] = []

    async def qualifyContractsAsync(self, contract):
        self.qualify_calls.append(contract.lastTradeDateOrContractMonth)
        return [contract] if self.qualify_ok else []

    async def reqHistoricalDataAsync(self, contract, *, endDateTime, **kwargs):
        self.request_ends.append(endDateTime)
        if self.errors:
            raise self.errors.pop(0)
        return self.bars_by_end.get(endDateTime, [])


def _run(ib, sessions, out_dir, **overrides):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    kwargs = dict(
        sessions=sessions,
        out_dir=out_dir,
        symbol="ES",
        exchange="CME",
        currency="USD",
        contract_month_fn=lambda s: mod.front_contract_month(s, roll_days=8),
        pace_seconds=12.0,
        max_retries=2,
        backoff_cap=600.0,
        sleep=fake_sleep,
        log=LOG,
    )
    kwargs.update(overrides)
    summary = asyncio.run(mod.run_fetch(ib, **kwargs))
    return summary, sleeps


# --- roll schedule ---------------------------------------------------------


def test_third_friday_known_dates():
    assert mod.third_friday(2026, 6) == date(2026, 6, 19)
    assert mod.third_friday(2026, 9) == date(2026, 9, 18)
    assert mod.third_friday(2024, 3) == date(2024, 3, 15)


def test_front_contract_month_rolls_eight_days_before_expiry():
    # ESM6 expiry 2026-06-19; roll cutoff with roll_days=8 is 2026-06-11.
    assert mod.front_contract_month(date(2026, 6, 10), roll_days=8) == "202606"
    assert mod.front_contract_month(date(2026, 6, 11), roll_days=8) == "202609"
    # After the December roll the front month is March of the next year.
    assert mod.front_contract_month(date(2026, 12, 28), roll_days=8) == "202703"


def test_front_contract_month_roll_days_zero_holds_until_expiry():
    assert mod.front_contract_month(date(2026, 6, 18), roll_days=0) == "202606"
    assert mod.front_contract_month(date(2026, 6, 19), roll_days=0) == "202609"


# --- frame normalization and shard schema ----------------------------------


def test_bars_to_frame_filters_rth_and_session():
    session = date(2026, 6, 1)
    keep = _session_bars(session, n=3)
    stray = [
        _bar(datetime(2026, 6, 1, 13, 29, tzinfo=UTC)),  # 08:29 CT, pre-RTH
        _bar(datetime(2026, 6, 1, 20, 0, tzinfo=UTC)),  # 15:00 CT, post-RTH
        _bar(datetime(2026, 6, 2, 13, 30, tzinfo=UTC)),  # wrong session
    ]
    df = mod.bars_to_frame(keep + stray, session)
    assert list(df.columns) == list(mod.OHLCV_COLUMNS)
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing


def test_bars_to_frame_dedupes_keeping_last():
    session = date(2026, 6, 1)
    ts = datetime(2026, 6, 1, 13, 30, tzinfo=UTC)
    df = mod.bars_to_frame([_bar(ts, price=100.0), _bar(ts, price=200.0)], session)
    assert len(df) == 1
    assert df.loc[0, "open"] == 200.0


def test_write_shard_matches_recorder_dtypes(tmp_path):
    session = date(2026, 6, 1)
    df = mod.bars_to_frame(_session_bars(session), session)
    path = mod.shard_path(tmp_path, session)
    assert mod.write_shard(df, path) == 5

    back = pd.read_parquet(path)
    assert list(back.columns) == list(mod.OHLCV_COLUMNS)
    assert str(back["timestamp"].dtype) == "datetime64[ms, UTC]"
    assert str(back["volume"].dtype) == "int64"


# --- run_fetch orchestration -----------------------------------------------


def test_run_fetch_skips_existing_shards(tmp_path):
    done = date(2026, 6, 1)
    todo = date(2026, 6, 2)
    mod.write_shard(mod.bars_to_frame(_session_bars(done), done), mod.shard_path(tmp_path, done))

    ib = FakeIB(bars_by_end={mod.format_ibkr_end_datetime(todo): _session_bars(todo)})
    summary, _ = _run(ib, [done, todo], tmp_path)

    assert summary.skipped_existing == 1
    assert summary.fetched == 1
    assert ib.request_ends == [mod.format_ibkr_end_datetime(todo)]
    assert mod.shard_path(tmp_path, todo).exists()


def test_run_fetch_paces_between_requests(tmp_path):
    days = [date(2026, 6, 1), date(2026, 6, 2)]
    bars = {mod.format_ibkr_end_datetime(d): _session_bars(d) for d in days}
    summary, sleeps = _run(FakeIB(bars_by_end=bars), days, tmp_path)

    assert summary.fetched == 2
    assert sleeps == [12.0]  # no sleep before the first request, paced after


def test_run_fetch_retries_with_backoff_then_succeeds(tmp_path):
    day = date(2026, 6, 1)
    ib = FakeIB(
        bars_by_end={mod.format_ibkr_end_datetime(day): _session_bars(day)},
        errors=[RequestError(1, 162, "Historical Market Data Service error message:pacing violation")],
    )
    summary, sleeps = _run(ib, [day], tmp_path)

    assert summary.fetched == 1
    assert summary.failed_days == []
    assert sleeps == [24.0]  # one exponential backoff (12 * 2**1)
    assert len(ib.request_ends) == 2


def test_run_fetch_gives_up_after_max_retries(tmp_path):
    day = date(2026, 6, 1)
    errs = [RequestError(1, 162, "pacing violation")] * 3
    summary, _ = _run(FakeIB(errors=errs), [day], tmp_path, max_retries=2)

    assert summary.fetched == 0
    assert summary.failed_days == [day.isoformat()]
    assert not mod.shard_path(tmp_path, day).exists()


def test_run_fetch_treats_no_data_as_empty_day(tmp_path):
    day = date(2026, 6, 1)
    err = RequestError(1, 162, "Historical Market Data Service error message:HMDS query returned no data")
    summary, sleeps = _run(FakeIB(errors=[err]), [day], tmp_path)

    assert summary.empty_days == 1
    assert summary.failed_days == []
    assert sleeps == []  # no retry, no backoff
    assert not mod.shard_path(tmp_path, day).exists()


def test_run_fetch_unqualified_contract_marks_days_failed(tmp_path):
    days = [date(2026, 6, 1), date(2026, 6, 2)]
    ib = FakeIB(qualify_ok=False)
    summary, _ = _run(ib, days, tmp_path)

    assert summary.fetched == 0
    assert summary.failed_days == [d.isoformat() for d in days]
    assert ib.qualify_calls == ["202606"]  # qualified once, then cached as bad
    assert ib.request_ends == []


# --- CLI parsing ------------------------------------------------------------


def test_parse_argv_clamps_pace_floor(tmp_path):
    args = mod.parse_argv(
        ["--start", "2026-06-01", "--end", "2026-06-02", "--out", str(tmp_path), "--pace-seconds", "3"]
    )
    assert args.pace_seconds == mod.MIN_PACE_SECONDS
    assert args.client_id == 30


def test_parse_argv_rejects_inverted_range(tmp_path):
    with pytest.raises(SystemExit):
        mod.parse_argv(["--start", "2026-06-02", "--end", "2026-06-01", "--out", str(tmp_path)])
