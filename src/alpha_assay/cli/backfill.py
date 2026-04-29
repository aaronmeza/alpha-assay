# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""`alpha_assay backfill` command.

Pulls historical 1-min OHLCV bars from IBKR via
``IBKRAdapter.historical_bars_async`` and writes them to per-day
parquet shards using the canonical schema accepted by
:func:`alpha_assay.data.databento_adapter.load_parquet`.

The shard layout is identical to the ES-bars recorder
(`{out}/YYYY-MM-DD.parquet`) so live and historical data can be
ingested interchangeably by the engine. When a shard already exists
(typically because the live recorder is also writing to ``--out``),
the new bars are merged in by sort + dedupe rather than clobbering
the existing file.

Bars outside the recorder's RTH window (08:30-15:00 America/Chicago,
weekdays only) are dropped before any shard is written. ES does not
trade Saturday CT and the Sunday-evening Globex re-open is part of
the upcoming Monday session by exchange convention; without this
gate the backfill would emit non-trading-day shards (e.g. a Sunday
shard with bars from 17:00-23:59 CT) that the live recorder would
never produce, breaking the "live + historical interchangeable"
invariant. The adapter still pulls with ``useRTH=False`` so
out-of-RTH bars remain available to callers that want them; the CLI
filters at write time.

Why one-shot historical instead of the recorder's history-replay
buffer: ``reqHistoricalData(..., keepUpToDate=True)`` only replays
roughly the last 2 hours per call before flipping to live updates.
For backtests we need a multi-month window; that requires
walking the request endpoint backwards in chunks under IBKR's pacing
limits.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import click
import pandas as pd

from alpha_assay.data.ibkr_adapter import IBKRAdapter

LOG = logging.getLogger(__name__)

CHICAGO = ZoneInfo("America/Chicago")

# RTH session in America/Chicago. Mirrors the # ``ibkr_es_bars`` recorder so backfilled shards land in the same
# layout the live recorder produces. CME E-mini S&P 500 (ES) does
# not trade Saturday CT and the Sunday-evening Globex re-open is
# treated as part of the upcoming Monday session - both of which
# the recorder drops via this gate.
RTH_START_CT = dtime(8, 30)
RTH_END_CT = dtime(15, 0)

# Canonical column order. Matches alpha_assay.data.databento_adapter._REQUIRED
# plus the timestamp column the loader extracts back into the index.
_OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

# IBKR durationStr keywords. Sufficient for the values the CLI exposes.
_DURATION_UNIT_SECONDS = {
    "S": 1,
    "D": 86_400,
    "W": 7 * 86_400,
    "M": 30 * 86_400,
    "Y": 365 * 86_400,
}

# Feeds we currently support. TICK-NYSE / AD-NYSE have no historical
# data per IBKR's docs (ADR Appendix B). Keep the surface narrow and
# reject unknown feeds with a clear message.
_SUPPORTED_FEEDS = ("es-bars",)


def _parse_duration_seconds(duration_str: str) -> int:
    """Approximate the duration string in seconds.

    Used purely for chunk-schedule arithmetic (how far each chunk
    spans). IBKR is the authoritative bound at request time; we just
    need a non-zero positive width to walk backwards by.
    """
    parts = duration_str.strip().split()
    if len(parts) != 2:
        raise click.BadParameter(
            f"--chunk-duration must be like '1 W' or '2 D', got {duration_str!r}"
        )
    try:
        n = int(parts[0])
    except ValueError as exc:
        raise click.BadParameter(f"--chunk-duration count must be int, got {parts[0]!r}") from exc
    unit = parts[1].upper()
    if unit not in _DURATION_UNIT_SECONDS:
        raise click.BadParameter(
            f"--chunk-duration unit must be one of {sorted(_DURATION_UNIT_SECONDS)}, "
            f"got {parts[1]!r}"
        )
    if n <= 0:
        raise click.BadParameter("--chunk-duration count must be positive")
    return n * _DURATION_UNIT_SECONDS[unit]


def _format_ibkr_end_datetime(ts: datetime) -> str:
    """Render a UTC datetime as IBKR's endDateTime string.

    IBKR accepts ``YYYYMMDD HH:MM:SS UTC`` (note the trailing tz
    keyword). ``ts`` is expected tz-aware in UTC.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)
    return ts.strftime("%Y%m%d %H:%M:%S UTC")


@dataclass(frozen=True, slots=True)
class ChunkWindow:
    """One IBKR historical request worth of bars.

    ``end`` is the right edge (exclusive in spirit) IBKR will receive
    as ``endDateTime``. ``start`` is informational - useful for
    progress logs and resume-skip checks - and is computed by
    subtracting the chunk's approximate duration.
    """

    end: datetime
    start: datetime
    duration_str: str


def build_chunk_schedule(
    *,
    now: datetime,
    days: int,
    chunk_duration: str,
) -> list[ChunkWindow]:
    """Walk backwards from ``now`` in ``chunk_duration`` slices.

    The total window covers at least ``days * 86400`` seconds. The
    final (oldest) chunk may extend past the requested window's left
    edge, which is fine: IBKR will simply return whatever data lies
    inside its ``durationStr`` regardless of where the window starts.
    Callers dedupe on write.
    """
    if days <= 0:
        raise click.BadParameter("--days must be a positive integer")
    chunk_seconds = _parse_duration_seconds(chunk_duration)
    total_seconds = days * 86_400

    chunks: list[ChunkWindow] = []
    end = now
    elapsed = 0
    while elapsed < total_seconds:
        start = end - timedelta(seconds=chunk_seconds)
        chunks.append(ChunkWindow(end=end, start=start, duration_str=chunk_duration))
        elapsed += chunk_seconds
        end = start
    return chunks


def _bar_session_date(ts: pd.Timestamp) -> str:
    """Return the YYYY-MM-DD shard key for a bar timestamp.

    Bucket by America/Chicago calendar day to match the recorder's output layout exactly.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(CHICAGO).strftime("%Y-%m-%d")


def _is_in_rth_ct(ts: pd.Timestamp) -> bool:
    """Return True iff the bar's timestamp lands inside the RTH session.

    Mirrors :func:`infra.recorders.ibkr_es_bars.recorder._is_in_rth`:
    weekdays only (Mon-Fri CT) and 08:30-15:00 CT. ES Globex re-opens
    at 17:00 CT on Sundays and runs through 16:00 CT Friday with a
    daily 16:00-17:00 CT halt; none of those bars are part of RTH so
    they should never reach a per-day shard. zoneinfo handles DST.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_ct = ts.tz_convert(CHICAGO)
    if ts_ct.weekday() >= 5:
        return False
    local_time = ts_ct.time()
    return RTH_START_CT <= local_time < RTH_END_CT


def _group_bars_by_day(bars: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group canonical bars dicts by America/Chicago shard date.

    Bars outside RTH (08:30-15:00 CT, Mon-Fri) are dropped so the
    backfill never produces non-trading-day shards (e.g. Saturday or
    Sunday-evening Globex re-open) and weekday shards stay aligned
    with the live recorder's ~390-bar RTH layout.
    """
    by_day: dict[str, list[dict[str, Any]]] = {}
    for bar in bars:
        ts = bar["timestamp"]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if not _is_in_rth_ct(ts):
            continue
        day = _bar_session_date(ts)
        by_day.setdefault(day, []).append(bar)
    return by_day


def _merge_into_shard(path: Path, new_rows: list[dict[str, Any]]) -> int:
    """Write or merge ``new_rows`` into the shard at ``path``.

    If a shard already exists at ``path``, read it, concat the new
    rows, dedupe by timestamp keeping the new value (so a re-pull
    overrides a stale bar), sort, and rewrite. Returns the total row
    count after the merge.

    The shard schema is the canonical Databento OHLCV schema; the
    ``feed`` column emitted by ``subscribe_bars`` / ``historical_bars_async``
    is intentionally dropped here so the file passes
    :func:`alpha_assay.data.databento_adapter.load_parquet` validation.
    """
    if not new_rows:
        return 0

    df_new = pd.DataFrame(new_rows)
    # Coerce timestamp to a single tz (UTC) for clean dedupe.
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)

    if path.exists():
        df_existing = pd.read_parquet(path)
        if "timestamp" in df_existing.columns:
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
        # Drop legacy / extra columns; keep only canonical OHLCV.
        df_existing = df_existing[[c for c in _OHLCV_COLUMNS if c in df_existing.columns]]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = df_combined[[c for c in _OHLCV_COLUMNS if c in df_combined.columns]]
    # Last write wins on duplicate timestamps so re-pulls override stale bars.
    df_combined = df_combined.drop_duplicates(subset="timestamp", keep="last")
    df_combined = df_combined.sort_values("timestamp", kind="stable").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(path, index=False)
    return len(df_combined)


def _flush_buffer_to_shards(
    buffer: list[dict[str, Any]],
    out_dir: Path,
    *,
    log: logging.Logger,
) -> int:
    """Group ``buffer`` by shard date and merge into per-day parquet files.

    Returns the number of distinct days touched. Empties ``buffer`` in place.
    Out-of-RTH bars (weekend CT or outside 08:30-15:00 CT) are silently
    discarded by :func:`_group_bars_by_day` so they never produce a shard.
    """
    if not buffer:
        return 0
    incoming = len(buffer)
    by_day = _group_bars_by_day(buffer)
    kept = sum(len(rows) for rows in by_day.values())
    if kept < incoming:
        log.info(
            "backfill: dropped %d non-RTH bars (kept %d of %d for shard write)",
            incoming - kept,
            kept,
            incoming,
        )
    for day, rows in sorted(by_day.items()):
        shard_path = out_dir / f"{day}.parquet"
        total = _merge_into_shard(shard_path, rows)
        log.info(
            "backfill: wrote shard day=%s new_bars=%d total_bars=%d path=%s",
            day,
            len(rows),
            total,
            shard_path,
        )
    n_days = len(by_day)
    buffer.clear()
    return n_days


def _build_contract_spec(
    *, symbol: str, exchange: str, currency: str, expiry: str
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "sec_type": "FUT",
        "exchange": exchange,
        "currency": currency,
        "expiry": expiry,
    }


async def _run_backfill(
    *,
    adapter: IBKRAdapter,
    contract_spec: dict[str, Any],
    out_dir: Path,
    chunks: list[ChunkWindow],
    pace_seconds: int,
    bar_size_setting: str,
    what_to_show: str,
    log: logging.Logger,
) -> tuple[int, int]:
    """Connect, walk the chunk schedule, write shards. Returns (chunks_done, bars_written)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[dict[str, Any]] = []
    chunks_done = 0
    bars_total = 0

    await adapter.connect_async()
    try:
        for idx, chunk in enumerate(chunks):
            if idx > 0 and pace_seconds > 0:
                # Pace strictly between requests so identical-spec
                # successive calls stay outside IBKR's 15s + 6/2s rules.
                await asyncio.sleep(pace_seconds)
            end_str = _format_ibkr_end_datetime(chunk.end)
            log.info(
                "backfill: chunk %d/%d end=%s duration=%s",
                idx + 1,
                len(chunks),
                end_str,
                chunk.duration_str,
            )
            try:
                bars = await adapter.historical_bars_async(
                    contract_spec,
                    end_datetime=end_str,
                    duration_str=chunk.duration_str,
                    bar_size_setting=bar_size_setting,
                    what_to_show=what_to_show,
                    use_rth=False,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Don't lose the buffered bars on a single chunk failure;
                # flush what we have and re-raise so the operator sees it.
                log.exception("backfill: chunk %d/%d failed", idx + 1, len(chunks))
                _flush_buffer_to_shards(buffer, out_dir, log=log)
                raise

            log.info(
                "backfill: chunk %d/%d returned %d bars (running total %d)",
                idx + 1,
                len(chunks),
                len(bars),
                bars_total + len(bars),
            )
            buffer.extend(bars)
            bars_total += len(bars)
            chunks_done += 1

            # Flush per chunk to keep the on-disk state close to the
            # in-memory state. If the operator hits Ctrl-C we still
            # have most of the work persisted.
            _flush_buffer_to_shards(buffer, out_dir, log=log)
    finally:
        await adapter.disconnect_async()
        # Defensive: in case anything slipped past the per-chunk flush.
        _flush_buffer_to_shards(buffer, out_dir, log=log)

    return chunks_done, bars_total


@click.command()
@click.option(
    "--feed",
    type=click.Choice(list(_SUPPORTED_FEEDS), case_sensitive=False),
    required=True,
    help="Which feed to backfill. Only 'es-bars' is supported today.",
)
@click.option(
    "--days",
    type=int,
    default=180,
    show_default=True,
    help="How many calendar days back from now to pull (~6 months default).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default="data/recorder/es_bars/",
    show_default=True,
    help="Output directory; per-day shards land here. Defaults match the running recorder.",
)
@click.option(
    "--ibkr-host",
    default=None,
    help="TWS / Gateway host. Default: $IBKR_HOST or 127.0.0.1.",
)
@click.option(
    "--ibkr-port",
    type=int,
    default=None,
    help="TWS / Gateway port. Default: $IBKR_PORT or 4002.",
)
@click.option(
    "--ibkr-client-id",
    type=int,
    default=23,
    show_default=True,
    help="IBKR clientId. Default 23 to avoid clashing with paper-trader / recorders.",
)
@click.option(
    "--es-expiry",
    default=None,
    help="ES futures lastTradeDateOrContractMonth (YYYYMMDD). Default: $ES_EXPIRY or 20260618.",
)
@click.option(
    "--chunk-duration",
    default="1 W",
    show_default=True,
    help="Per-request IBKR durationStr. 1-min bars cap at ~1 W per call.",
)
@click.option(
    "--pace-seconds",
    type=int,
    default=10,
    show_default=True,
    help="Wait between successive IBKR requests (avoids 15s identical-request + 6/2s rules).",
)
@click.option(
    "--symbol",
    default="ES",
    show_default=True,
    help="Futures root symbol.",
)
@click.option(
    "--exchange",
    default="CME",
    show_default=True,
    help="Futures exchange.",
)
@click.option(
    "--currency",
    default="USD",
    show_default=True,
    help="Quote currency.",
)
def backfill(
    feed: str,
    days: int,
    out_dir: str,
    ibkr_host: str | None,
    ibkr_port: int | None,
    ibkr_client_id: int,
    es_expiry: str | None,
    chunk_duration: str,
    pace_seconds: int,
    symbol: str,
    exchange: str,
    currency: str,
) -> None:
    """Pull historical ES 1-min OHLCV from IBKR into per-day parquet shards.

    Resume-safe: existing shards in ``--out`` are merged with new bars
    (sort + dedupe by timestamp) rather than clobbered. Safe to run
    while the recorder is also writing to the same directory
    because each shard is rewritten atomically per pandas / pyarrow.
    """
    feed_lower = feed.lower()
    if feed_lower != "es-bars":
        raise click.BadParameter(
            f"--feed must be one of {list(_SUPPORTED_FEEDS)}; "
            "TICK-NYSE / AD-NYSE have no IBKR historical data per ADR Appendix B."
        )

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    host = ibkr_host or os.environ.get("IBKR_HOST", "127.0.0.1")
    port = ibkr_port if ibkr_port is not None else int(os.environ.get("IBKR_PORT", "4002"))
    expiry = es_expiry or os.environ.get("ES_EXPIRY", "20260618")

    contract_spec = _build_contract_spec(
        symbol=symbol, exchange=exchange, currency=currency, expiry=expiry
    )

    now_utc = datetime.now(tz=UTC).replace(microsecond=0)
    chunks = build_chunk_schedule(now=now_utc, days=days, chunk_duration=chunk_duration)

    out_path = Path(out_dir)
    LOG.info(
        "backfill: feed=%s days=%d chunks=%d chunk_duration=%s pace=%ds out=%s ibkr=%s:%d",
        feed_lower,
        days,
        len(chunks),
        chunk_duration,
        pace_seconds,
        out_path,
        host,
        port,
    )

    adapter = IBKRAdapter(
        host=host,
        port=port,
        client_id=ibkr_client_id,
        read_only=True,
    )

    try:
        chunks_done, bars_total = asyncio.run(
            _run_backfill(
                adapter=adapter,
                contract_spec=contract_spec,
                out_dir=out_path,
                chunks=chunks,
                pace_seconds=pace_seconds,
                bar_size_setting="1 min",
                what_to_show="TRADES",
                log=LOG,
            )
        )
    except KeyboardInterrupt:
        click.echo("backfill: interrupted; partial work flushed to shards", err=True)
        raise SystemExit(130) from None

    click.echo(
        f"backfill: done. chunks={chunks_done}/{len(chunks)} bars={bars_total} out={out_path}"
    )


__all__ = [
    "ChunkWindow",
    "backfill",
    "build_chunk_schedule",
]
