#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generic, paced, resumable IBKR historical 1-min bar fetcher.

Pulls one RTH session per request via ``ib_insync.IB.reqHistoricalDataAsync``
and writes per-day parquet shards matching the live recorder schema
(``timestamp`` tz-aware UTC, ``open``, ``high``, ``low``, ``close``,
``volume``), one file per session at ``{out}/YYYY-MM-DD.parquet``. A day
whose shard already exists is skipped, so the script is resumable and safe
to re-run after an interruption.

Contract selection: stitched explicit expiries, not CONTFUT
-----------------------------------------------------------
For a multi-year futures pull there are two candidate approaches:

1. ``ContFuture`` (secType ``CONTFUT``). In ib_insync this is a thin
   passthrough (``ib_insync.contract.ContFuture`` just sets
   ``secType='CONTFUT'``); the stitching policy lives entirely on IBKR's
   servers. IBKR serves CONTFUT history as front-month data concatenated at
   each contract's *expiration* date, unadjusted, and CONTFUT is only valid
   for historical data / contract details. That means the final ~week of
   every quarter is served from the dying contract after open interest has
   already migrated to the next one, and the splice date is neither
   controllable nor visible per bar.
2. Explicit per-quarter expiries (``includeExpired=True``) stitched by a
   documented roll rule.

This script uses approach 2. Each session date is mapped to the contract
that was the *liquid* front month that day: the nearest quarterly expiry
(third Friday of Mar/Jun/Sep/Dec) whose roll cutoff -- expiry minus
``--roll-days`` calendar days, default 8, the conventional equity-index
volume roll -- is still ahead of the session. The mapping is deterministic
and reproducible, and every session's bars come from a single, known
contract.

Known limits of this choice:

- The built-in roll schedule assumes quarterly third-Friday expiries
  (equity-index style: ES, NQ, RTY, YM). For other products pass an
  explicit ``--expiry`` per pull.
- The stitched series is unadjusted. On the first session after a roll the
  overnight gap versus the prior session's close includes the calendar
  spread between the two contracts. Studies that anchor returns on the
  prior session close should drop or flag roll-boundary sessions (the log
  reports each session's contract; a contract change between consecutive
  sessions is a roll boundary).
- IBKR only serves historical data for futures that expired within roughly
  the last two years. Sessions older than that come back empty and are
  logged, not retried.

Pacing and error handling
-------------------------
IBKR's historical pacing limits are strict (no more than ~60 historical
requests per 10 minutes; identical-request cooldowns). The script issues
one request per session day and sleeps ``--pace-seconds`` (default 12,
floor 10) between requests, which stays at or under 5 requests/minute.
``IB.RaiseRequestErrors`` is enabled so a failed request raises
``ib_insync.RequestError`` instead of silently resolving empty; retryable
failures (pacing violations and other transient errors) back off
exponentially from the pace interval up to ``--backoff-cap`` seconds for
``--max-retries`` attempts. Error 162 with an "HMDS query returned no
data" message is treated as a legitimately empty day (holiday, or data
older than IBKR retains) -- logged, no shard written, no retry. Ctrl-C
disconnects cleanly and exits 130; already-written shards are kept, so a
rerun resumes where the interrupted run stopped.

Usage::

    python scripts/fetch_historical_bars.py \\
        --start 2024-06-12 --end 2026-06-10 \\
        --out data/historical/es_1min \\
        --host 127.0.0.1 --port 4002 --client-id 30

    # explicit single contract instead of the quarterly roll schedule:
    python scripts/fetch_historical_bars.py \\
        --start 2026-03-16 --end 2026-06-10 \\
        --out data/historical/es_1min --expiry 202606

    # print the session -> contract plan without connecting:
    python scripts/fetch_historical_bars.py --start 2026-06-01 --out /tmp/x --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Future, RequestError

LOG = logging.getLogger("fetch_historical_bars")

CHICAGO = ZoneInfo("America/Chicago")

# RTH session gate in America/Chicago. Mirrors the live ES-bars recorder
# (08:30 <= t < 15:00 CT, weekdays) so fetched shards are interchangeable
# with recorded ones.
RTH_START_CT = time(8, 30)
RTH_END_CT = time(15, 0)

# Canonical shard column order (recorder schema).
OHLCV_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")

# Quarterly expiry months for equity-index futures (third-Friday cycle).
QUARTERLY_MONTHS: tuple[int, ...] = (3, 6, 9, 12)

# Minimum allowed pacing interval. IBKR allows ~60 historical requests
# per 10 minutes; 10s spacing = 6/min, right at the limit, so it is the
# floor rather than the default.
MIN_PACE_SECONDS = 10.0

# Substrings that mark IBKR error 162 as "no data for this query" rather
# than a pacing violation (both share the 162 code).
_NO_DATA_MARKERS = ("no data", "query returned no data")


def third_friday(year: int, month: int) -> date:
    """Return the third Friday of ``year``/``month`` (CME equity-index expiry)."""
    first = date(year, month, 1)
    first_friday = 1 + (4 - first.weekday()) % 7
    return date(year, month, first_friday + 14)


def front_contract_month(session: date, *, roll_days: int = 8) -> str:
    """Map a session date to its liquid front contract month (``YYYYMM``).

    The front contract is the earliest quarterly expiry whose roll cutoff
    (third-Friday expiry minus ``roll_days`` calendar days) is strictly
    after ``session``. ``roll_days=0`` holds each contract until its
    expiry date.
    """
    if roll_days < 0:
        raise ValueError("roll_days must be >= 0")
    year = session.year
    for y in (year, year + 1):
        for m in QUARTERLY_MONTHS:
            cutoff = third_friday(y, m) - timedelta(days=roll_days)
            if session < cutoff:
                return f"{y}{m:02d}"
    raise RuntimeError(f"no quarterly contract found for session {session}")  # pragma: no cover


def iter_weekday_sessions(start: date, end: date) -> Iterator[date]:
    """Yield candidate session dates (Mon-Fri) in ``[start, end]`` inclusive.

    Holidays are not filtered here; they come back from IBKR as empty
    days and are skipped at fetch time.
    """
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            yield cur
        cur += timedelta(days=1)


def shard_path(out_dir: Path, session: date) -> Path:
    return out_dir / f"{session.isoformat()}.parquet"


def bars_to_frame(bars: list[Any], session: date) -> pd.DataFrame:
    """Normalize raw ib_insync ``BarData`` into the canonical shard frame.

    Keeps only bars whose America/Chicago timestamp falls on ``session``
    and inside the RTH window, then sorts, dedupes by timestamp (last
    write wins) and casts to the recorder dtypes (``datetime64[ms, UTC]``
    timestamp, ``int64`` volume).
    """
    rows = [
        {
            "timestamp": bar.date,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": bar.volume,
        }
        for bar in bars
    ]
    if not rows:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ts_ct = df["timestamp"].dt.tz_convert(CHICAGO)
    in_session = ts_ct.dt.date == session
    in_rth = (ts_ct.dt.time >= RTH_START_CT) & (ts_ct.dt.time < RTH_END_CT)
    df = df[in_session & in_rth]
    if df.empty:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("datetime64[ms, UTC]")
    df["volume"] = df["volume"].round().astype("int64")
    return df[list(OHLCV_COLUMNS)]


def write_shard(df: pd.DataFrame, path: Path) -> int:
    """Write a session frame to its parquet shard. Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)


def format_ibkr_end_datetime(session: date) -> str:
    """IBKR ``endDateTime`` string just past the session close, in UTC."""
    end_ct = datetime.combine(session, time(15, 10), tzinfo=CHICAGO)
    return end_ct.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d %H:%M:%S UTC")


def is_no_data_error(exc: BaseException) -> bool:
    """True if ``exc`` is IBKR error 162 reporting an empty query result."""
    if not isinstance(exc, RequestError) or exc.code != 162:
        return False
    message = exc.message.lower()
    return any(marker in message for marker in _NO_DATA_MARKERS)


@dataclass
class FetchSummary:
    fetched: int = 0
    skipped_existing: int = 0
    empty_days: int = 0
    failed_days: list[str] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.fetched + self.empty_days + len(self.failed_days)


async def run_fetch(
    ib: Any,
    *,
    sessions: list[date],
    out_dir: Path,
    symbol: str,
    exchange: str,
    currency: str,
    contract_month_fn: Callable[[date], str],
    what_to_show: str = "TRADES",
    pace_seconds: float = 12.0,
    max_retries: int = 3,
    backoff_cap: float = 600.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    log: logging.Logger = LOG,
) -> FetchSummary:
    """Walk ``sessions`` oldest-first, fetching and writing one shard per day.

    ``ib`` is anything that provides ``qualifyContractsAsync`` and
    ``reqHistoricalDataAsync`` (the real ``ib_insync.IB`` in production, a
    fake in tests). Existing shards are skipped before any request is
    made, so resumption costs zero IBKR quota.
    """
    summary = FetchSummary()
    contracts: dict[str, Any] = {}
    unqualified: set[str] = set()
    last_month = None
    made_request = False

    for session in sorted(sessions):
        path = shard_path(out_dir, session)
        if path.exists():
            summary.skipped_existing += 1
            continue

        month = contract_month_fn(session)
        if month in unqualified:
            summary.failed_days.append(session.isoformat())
            continue
        if month not in contracts:
            contract = Future(symbol, lastTradeDateOrContractMonth=month, exchange=exchange, currency=currency)
            contract.includeExpired = True
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified or qualified[0] is None:
                log.warning("contract %s %s could not be qualified; skipping its sessions", symbol, month)
                unqualified.add(month)
                summary.failed_days.append(session.isoformat())
                continue
            contracts[month] = qualified[0]
        if last_month is not None and month != last_month:
            log.info("roll boundary: %s -> %s at session %s", last_month, month, session)
        last_month = month

        bars: list[Any] | None = None
        for attempt in range(max_retries + 1):
            if made_request:
                delay = pace_seconds if attempt == 0 else min(pace_seconds * (2**attempt), backoff_cap)
                await sleep(delay)
            try:
                made_request = True
                bars = await ib.reqHistoricalDataAsync(
                    contracts[month],
                    endDateTime=format_ibkr_end_datetime(session),
                    durationStr="1 D",
                    barSizeSetting="1 min",
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=2,
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if is_no_data_error(exc):
                    log.info("session %s: no data (holiday or beyond IBKR retention)", session)
                    bars = []
                    break
                if attempt >= max_retries:
                    log.error("session %s: failed after %d retries: %s", session, max_retries, exc)
                    summary.failed_days.append(session.isoformat())
                    bars = None
                    break
                log.warning("session %s: attempt %d failed (%s); backing off", session, attempt + 1, exc)

        if bars is None:
            continue

        df = bars_to_frame(bars, session)
        if df.empty:
            log.info("session %s: 0 RTH bars; no shard written", session)
            summary.empty_days += 1
            continue

        rows = write_shard(df, path)
        summary.fetched += 1
        log.info("session %s: wrote %d bars (contract %s%s) -> %s", session, rows, symbol, month, path)

    return summary


def parse_argv(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--symbol", default="ES", help="Futures root symbol (default: ES)")
    parser.add_argument("--exchange", default="CME", help="Futures exchange (default: CME)")
    parser.add_argument("--currency", default="USD", help="Quote currency (default: USD)")
    parser.add_argument(
        "--expiry",
        default=None,
        help="Explicit contract month (YYYYMM or YYYYMMDD) for ALL sessions; disables the quarterly roll schedule",
    )
    parser.add_argument(
        "--roll-days",
        type=int,
        default=8,
        help="Calendar days before third-Friday expiry to roll to the next quarterly contract (default: 8)",
    )
    parser.add_argument("--start", required=True, help="First session date, YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--end",
        default=None,
        help="Last session date, YYYY-MM-DD (inclusive; default: today in America/Chicago)",
    )
    parser.add_argument("--out", required=True, help="Output directory for per-day parquet shards")
    parser.add_argument("--host", default="127.0.0.1", help="TWS / IB Gateway host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4002, help="TWS / IB Gateway port (default: 4002)")
    parser.add_argument(
        "--client-id",
        type=int,
        default=30,
        help="IBKR clientId (default: 30; pick one not used by any live consumer)",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=12.0,
        help=f"Seconds between historical requests (default: 12, floor {MIN_PACE_SECONDS:.0f})",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per session on transient errors")
    parser.add_argument("--backoff-cap", type=float, default=600.0, help="Max backoff sleep in seconds")
    parser.add_argument("--what-to-show", default="TRADES", help="IBKR whatToShow (default: TRADES)")
    parser.add_argument("--dry-run", action="store_true", help="Print the session -> contract plan and exit")
    args = parser.parse_args(argv)

    args.start_date = date.fromisoformat(args.start)
    args.end_date = date.fromisoformat(args.end) if args.end else datetime.now(tz=CHICAGO).date()
    if args.end_date < args.start_date:
        parser.error("--end must be on or after --start")
    if args.pace_seconds < MIN_PACE_SECONDS:
        LOG.warning("pace %.1fs below floor; clamping to %.0fs", args.pace_seconds, MIN_PACE_SECONDS)
        args.pace_seconds = MIN_PACE_SECONDS
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_argv(argv)

    sessions = list(iter_weekday_sessions(args.start_date, args.end_date))
    if args.expiry:
        contract_month_fn: Callable[[date], str] = lambda _session: args.expiry  # noqa: E731
    else:
        contract_month_fn = lambda session: front_contract_month(session, roll_days=args.roll_days)  # noqa: E731

    out_dir = Path(args.out)
    if args.dry_run:
        for session in sessions:
            done = " [shard exists]" if shard_path(out_dir, session).exists() else ""
            print(f"{session}  {args.symbol}{contract_month_fn(session)}{done}")
        print(f"{len(sessions)} candidate sessions")
        return 0

    LOG.info(
        "fetch: %s %s..%s sessions=%d out=%s gateway=%s:%d clientId=%d pace=%.0fs",
        args.symbol,
        args.start_date,
        args.end_date,
        len(sessions),
        out_dir,
        args.host,
        args.port,
        args.client_id,
        args.pace_seconds,
    )

    ib = IB()
    ib.RaiseRequestErrors = True

    async def _amain() -> FetchSummary:
        await ib.connectAsync(args.host, args.port, clientId=args.client_id, readonly=True)
        try:
            return await run_fetch(
                ib,
                sessions=sessions,
                out_dir=out_dir,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                contract_month_fn=contract_month_fn,
                what_to_show=args.what_to_show,
                pace_seconds=args.pace_seconds,
                max_retries=args.max_retries,
                backoff_cap=args.backoff_cap,
            )
        finally:
            ib.disconnect()

    try:
        summary = asyncio.run(_amain())
    except KeyboardInterrupt:
        print("fetch: interrupted; completed shards are kept, rerun to resume", file=sys.stderr)
        return 130

    LOG.info(
        "fetch: done. wrote=%d skipped_existing=%d empty=%d failed=%d%s",
        summary.fetched,
        summary.skipped_existing,
        summary.empty_days,
        len(summary.failed_days),
        f" failed_days={summary.failed_days}" if summary.failed_days else "",
    )
    return 1 if summary.failed_days else 0


if __name__ == "__main__":
    raise SystemExit(main())
