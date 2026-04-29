# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""IBKR breadth recorder .

Long-running service that consumes the `IBKRAdapter.subscribe_breadth`
read-only stream for TICK-NYSE and AD-NYSE, aggregates ticks causally into
1-min bars, and writes rolling daily parquet shards to
    {out_dir}/{symbol_sanitized}/{YYYY-MM-DD}.parquet

Canonical bar schema per day shard:
    {
        "timestamp": pd.Timestamp,   # tz-aware UTC, minute-floored
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "n_ticks": int,
        "symbol": str,
    }

Design invariants
-----------------
- **Causal aggregation only**: a minute bucket is emitted exactly when
  a tick arrives with a later minute. No wall-clock heuristics close a
  bucket early. On shutdown, the *completed* pending bucket (if its
  minute has already rolled over and the close tick was mid-flight)
  is flushed; an in-flight partial bucket is discarded.
- **RTH session filter**: ticks outside 08:30-15:00 America/Chicago
  are dropped silently before aggregation. Session membership uses
  timezone-aware datetimes so DST transitions are handled by tzdata,
  never by a hardcoded offset.
- **Reconnect**: the recorder catches adapter-side disconnects, logs
  loudly, and retries with exponential backoff 1,2,4,8,16,60s (capped).
  In-memory buckets and day buffers are preserved across reconnects.
- **Flush policy**: day-buffered bars are persisted to parquet every
  5 minutes, on day rollover, and on graceful shutdown. Each flush
  writes the full day file (overwrite) for idempotency - shards cap
  at ~450 bars/symbol so the cost is negligible.


"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from alpha_assay.data.ibkr_adapter import IBKRAdapter
from alpha_assay.observability import recorder_metrics as RM

LOG = logging.getLogger(__name__)

CHICAGO: tzinfo = ZoneInfo("America/Chicago")

# RTH session in America/Chicago (09:30-16:00 America/New_York == 08:30-15:00 CT).
RTH_START_CT = dtime(8, 30)
RTH_END_CT = dtime(15, 0)

# Reconnect backoff schedule: 1,2,4,8,16,60 seconds capped.
_BACKOFF_SEQUENCE: tuple[int, ...] = (1, 2, 4, 8, 16, 60)

# Periodic flush cadence (seconds). Separate from the 5-min retro test hook.
FLUSH_PERIOD_SECONDS = 300


def _sanitize_symbol(symbol: str) -> str:
    """Map ``TICK-NYSE`` -> ``TICK_NYSE`` for filesystem safety."""
    return symbol.replace("-", "_").replace("/", "_")


def _is_in_rth(ts_utc: pd.Timestamp) -> bool:
    """Return True iff the UTC timestamp falls inside the 08:30-15:00 CT
    session window. DST transitions are handled by zoneinfo.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_ct = ts_utc.tz_convert(CHICAGO)
    local_time = ts_ct.time()
    # Weekday gate: Monday=0 ... Friday=4.
    if ts_ct.weekday() >= 5:
        return False
    return RTH_START_CT <= local_time < RTH_END_CT


@dataclass
class _Bucket:
    """In-flight 1-min aggregation state for one symbol."""

    minute: pd.Timestamp  # UTC, minute-floored
    open: float
    high: float
    low: float
    close: float
    n_ticks: int = 1


@dataclass
class _DayBuffer:
    """Completed bars for one symbol for one shard-date."""

    date: str  # YYYY-MM-DD in America/Chicago local time
    rows: list[dict[str, Any]] = field(default_factory=list)


class BreadthRecorder:
    """Aggregate IBKR breadth ticks into rolling 1-min parquet bars.

    Parameters
    ----------
    adapter:
        ``IBKRAdapter`` instance. The recorder only calls
        ``connect`` / ``disconnect`` / ``subscribe_breadth`` / ``is_connected``.
    out_dir:
        Root directory for shard output. Created on first flush.
    symbols:
        Breadth symbols to subscribe to. Default: ``TICK-NYSE`` + ``AD-NYSE``.
    flush_period_seconds:
        Periodic flush cadence (default 300). Overridable for tests.
    shutdown_timeout_seconds:
        Max time budget for graceful shutdown drain + flush.
    """

    def __init__(
        self,
        *,
        adapter: IBKRAdapter,
        out_dir: Path,
        symbols: tuple[str, ...] = ("TICK-NYSE", "AD-NYSE"),
        flush_period_seconds: int = FLUSH_PERIOD_SECONDS,
        shutdown_timeout_seconds: int = 30,
    ) -> None:
        self._adapter = adapter
        self._out_dir = Path(out_dir)
        self._symbols = tuple(symbols)
        self._flush_period_seconds = flush_period_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

        # Per-symbol aggregation state. Preserved across reconnects.
        self._buckets: dict[str, _Bucket | None] = {s: None for s in self._symbols}
        self._day_buffers: dict[str, _DayBuffer | None] = {s: None for s in self._symbols}

        # Track last in-RTH tick timestamps for session-gap gauge.
        self._last_rth_tick: dict[str, pd.Timestamp | None] = {s: None for s in self._symbols}

        # Installed by run(); used by signal handler.
        self._shutdown_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- public surface -------------------------------------------------

    def ingest_tick(self, tick: dict[str, Any]) -> dict[str, Any] | None:
        """Ingest a single raw breadth tick. Return an emitted bar dict
        (if a minute bucket closed) or ``None``.

        Synchronous and side-effect-only wrt the recorder's internal
        state + day buffer + metrics. The caller decides when to flush
        the day buffer to parquet.
        """
        symbol = tick["symbol"]
        if symbol not in self._buckets:
            # Unknown symbol: silently ignore (defensive; adapter should not
            # send symbols we didn't subscribe to).
            return None

        RM.recorder_ticks_received_total.labels(symbol=symbol).inc()

        ts = pd.Timestamp(tick["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        if not _is_in_rth(ts):
            return None

        value = float(tick["value"])
        minute = ts.floor("1min")
        emitted: dict[str, Any] | None = None

        current = self._buckets[symbol]
        if current is None:
            self._buckets[symbol] = _Bucket(
                minute=minute,
                open=value,
                high=value,
                low=value,
                close=value,
                n_ticks=1,
            )
        elif minute == current.minute:
            current.high = max(current.high, value)
            current.low = min(current.low, value)
            current.close = value
            current.n_ticks += 1
        elif minute > current.minute:
            # Minute rolled over. Emit the completed bucket and start a new one.
            emitted = self._emit(symbol, current)
            self._buckets[symbol] = _Bucket(
                minute=minute,
                open=value,
                high=value,
                low=value,
                close=value,
                n_ticks=1,
            )
        # else: out-of-order tick with an earlier minute than the live bucket.
        # Ignore defensively; IBKR pendingTickers normally serialize by time.

        self._last_rth_tick[symbol] = ts
        RM.recorder_session_gap_seconds.labels(symbol=symbol).set(0.0)
        return emitted

    def flush(self) -> dict[str, int]:
        """Persist every non-empty day buffer to parquet. Return a map
        ``{symbol: rows_written}``.
        """
        written: dict[str, int] = {}
        for symbol, buf in self._day_buffers.items():
            if buf is None or not buf.rows:
                written[symbol] = 0
                continue
            try:
                path = self._shard_path(symbol, buf.date)
                path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(buf.rows)
                df.to_parquet(path, index=False)
            except Exception as exc:  # noqa: BLE001 - recorder must not crash on IO
                RM.recorder_write_errors_total.labels(
                    symbol=symbol,
                    error_class=type(exc).__name__,
                ).inc()
                LOG.exception("breadth-recorder: parquet write failed for %s", symbol)
                written[symbol] = 0
                continue
            RM.recorder_bars_written_total.labels(symbol=symbol).inc(len(buf.rows))
            written[symbol] = len(buf.rows)
        return written

    async def run(self) -> None:
        """Main loop. Connect, subscribe to every symbol, aggregate until
        SIGTERM. On adapter-side failure, reconnect with exponential
        backoff and resume.
        """
        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()

        # Register POSIX signal handlers (not available on Windows, but the
        # Docker image is Linux).
        for sig in (signal.SIGTERM, signal.SIGINT):
            with suppress(NotImplementedError):
                self._loop.add_signal_handler(sig, self._request_shutdown)

        # Periodic flush task.
        flush_task = self._loop.create_task(self._periodic_flush_loop())

        try:
            await self._connect_and_stream_forever()
        finally:
            flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await flush_task
            self._drain_and_flush()

    # --- internals ------------------------------------------------------

    def _request_shutdown(self) -> None:
        LOG.info("breadth-recorder: shutdown requested")
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    async def _periodic_flush_loop(self) -> None:
        """Flush every `flush_period_seconds`. Exits when shutdown is set."""
        assert self._shutdown_event is not None
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._flush_period_seconds,
                )
                return
            except TimeoutError:
                self.flush()

    async def _connect_and_stream_forever(self) -> None:
        """Outer loop: connect, stream, on failure back off and retry.

        Connects via :meth:`IBKRAdapter.connect_async` so the underlying
        ``ib_insync.IB.connectAsync`` runs as a normal awaitable inside
        this coroutine. The sync ``connect()`` would call
        ``loop.run_until_complete`` and crash with ``RuntimeError: This
        event loop is already running``.
        """
        assert self._shutdown_event is not None
        backoff_idx = 0
        while not self._shutdown_event.is_set():
            try:
                if not self._adapter.is_connected:
                    await self._adapter.connect_async()
                await self._stream_all_symbols()
                # Clean return means every subscription exited normally.
                return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - top-level supervisor
                LOG.exception("breadth-recorder: stream crashed; will reconnect")
                RM.recorder_reconnects_total.inc()
                with suppress(Exception):
                    await self._adapter.disconnect_async()

                delay = _BACKOFF_SEQUENCE[min(backoff_idx, len(_BACKOFF_SEQUENCE) - 1)]
                backoff_idx += 1

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
                    return
                except TimeoutError:
                    continue

    async def _stream_all_symbols(self) -> None:
        """Fan out one subscription task per symbol; exits when all done
        or on shutdown.
        """
        assert self._shutdown_event is not None
        tasks = [asyncio.create_task(self._stream_symbol(sym)) for sym in self._symbols]
        shutdown_waiter = asyncio.create_task(self._shutdown_event.wait())
        try:
            done, _pending = await asyncio.wait(
                [*tasks, shutdown_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                if t is shutdown_waiter:
                    continue
                # Re-raise exceptions from crashed symbol tasks to trigger
                # the supervisor reconnect path.
                exc = t.exception()
                if exc is not None:
                    raise exc
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                with suppress(asyncio.CancelledError, Exception):
                    await t
            if not shutdown_waiter.done():
                shutdown_waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await shutdown_waiter

    async def _stream_symbol(self, symbol: str) -> None:
        assert self._shutdown_event is not None
        async for tick in self._adapter.subscribe_breadth(symbol):
            if self._shutdown_event.is_set():
                return
            self.ingest_tick(tick)

    def _emit(self, symbol: str, bucket: _Bucket) -> dict[str, Any]:
        """Move a completed bucket into the day buffer and return it."""
        local_minute = bucket.minute.tz_convert(CHICAGO)
        day = local_minute.strftime("%Y-%m-%d")
        buf = self._day_buffers.get(symbol)
        if buf is None or buf.date != day:
            # Day rollover: flush the outgoing day before starting a new one.
            if buf is not None and buf.rows:
                self._flush_one(symbol, buf)
            buf = _DayBuffer(date=day)
            self._day_buffers[symbol] = buf

        row = {
            "timestamp": bucket.minute,
            "open": bucket.open,
            "high": bucket.high,
            "low": bucket.low,
            "close": bucket.close,
            "n_ticks": bucket.n_ticks,
            "symbol": symbol,
        }
        buf.rows.append(row)
        return row

    def _flush_one(self, symbol: str, buf: _DayBuffer) -> None:
        if not buf.rows:
            return
        try:
            path = self._shard_path(symbol, buf.date)
            path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(buf.rows)
            df.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001
            RM.recorder_write_errors_total.labels(
                symbol=symbol,
                error_class=type(exc).__name__,
            ).inc()
            LOG.exception("breadth-recorder: parquet write failed for %s", symbol)
            return
        RM.recorder_bars_written_total.labels(symbol=symbol).inc(len(buf.rows))

    def _drain_and_flush(self) -> None:
        """Graceful shutdown: any bucket whose minute has already closed
        is emitted; then flush every day buffer. Partial in-flight
        buckets (minute not yet rolled over) are discarded.
        """
        budget = timedelta(seconds=self._shutdown_timeout_seconds)
        started = datetime.now(tz=CHICAGO)

        now_utc = pd.Timestamp.now(tz="UTC")
        for symbol in self._symbols:
            bucket = self._buckets.get(symbol)
            if bucket is None:
                continue
            # A bucket is considered "closed" if wall-clock has advanced past
            # its minute boundary (we have no more same-minute ticks coming).
            if now_utc.floor("1min") > bucket.minute:
                self._emit(symbol, bucket)
            self._buckets[symbol] = None

        # Flush every buffered day - bounded to ~450 rows per symbol so
        # this fits comfortably within the 30s shutdown budget.
        for symbol, buf in list(self._day_buffers.items()):
            if buf is not None and buf.rows:
                self._flush_one(symbol, buf)
            if datetime.now(tz=CHICAGO) - started > budget:
                LOG.warning("breadth-recorder: shutdown flush exceeded budget")
                break

    def _shard_path(self, symbol: str, day: str) -> Path:
        return self._out_dir / _sanitize_symbol(symbol) / f"{day}.parquet"
