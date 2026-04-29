# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""IBKR ES 1-min bar recorder .

Long-running service that consumes the ``IBKRAdapter.subscribe_bars`` stream for an ES futures contract and
writes rolling daily parquet shards to
    {out_dir}/{YYYY-MM-DD}.parquet

Canonical shard schema (matches
:func:`alpha_assay.data.databento_adapter.load_parquet`):

    {
        "timestamp": pd.Timestamp,   # tz-aware UTC, minute-floored
        "open":   float,
        "high":   float,
        "low":    float,
        "close":  float,
        "volume": int,
    }

Design invariants
-----------------
- **No client-side aggregation**: ``subscribe_bars`` already returns
  per-minute closed bars (``reqHistoricalData(..., keepUpToDate=True)``
  emits one event per minute close). The recorder only buffers and
  flushes; OHLC clamps live in :func:`_normalize_bar` upstream.
- **RTH session filter**: bars whose UTC timestamp falls outside
  08:30-15:00 America/Chicago are dropped before buffering. The
  adapter requests ``useRTH=True``, so out-of-session bars normally
  never arrive; the gate is defense-in-depth for backfills,
  pre-open warmup, and DST corner cases.
- **OHLC clamp**: every accepted bar must satisfy
  ``high >= max(open, close)`` and ``low <= min(open, close)``.
  Bars violating the invariant are silently re-clamped (idempotent
  with the upstream clamp in ``_normalize_bar``) so a malformed
  bar never lands on disk.
- **Reconnect**: on adapter-side disconnect / subscription crash, the
  recorder logs loudly, increments the reconnects counter, and
  retries with exponential backoff 1,2,4,8,16,60s capped. In-memory
  day buffer is preserved across reconnects.
- **Flush policy**: the day buffer is persisted to parquet every 5
  minutes, on day rollover, and on graceful shutdown. Each flush
  rewrites the full day file (overwrite) for idempotency - shards
  cap at ~390 bars/session so the cost is negligible.

is responsible for the equivalent live IBKR validation against
real TWS for the always-flat paper-trader. The recorder is
deliberately strategy-free: it only collects.
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
from infra.recorders.ibkr_es_bars import recorder_metrics as RM

LOG = logging.getLogger(__name__)

CHICAGO: tzinfo = ZoneInfo("America/Chicago")

# RTH session in America/Chicago (09:30-16:00 America/New_York == 08:30-15:00 CT).
RTH_START_CT = dtime(8, 30)
RTH_END_CT = dtime(15, 0)

# Reconnect backoff schedule: 1,2,4,8,16,60 seconds capped.
_BACKOFF_SEQUENCE: tuple[int, ...] = (1, 2, 4, 8, 16, 60)

# Periodic flush cadence (seconds).
FLUSH_PERIOD_SECONDS = 300

# Canonical column order for the parquet shard - matches
# alpha_assay.data.databento_adapter._REQUIRED.
_OHLCV_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


def _is_in_rth(ts_utc: pd.Timestamp) -> bool:
    """Return True iff the UTC timestamp falls inside the 08:30-15:00 CT
    session window. DST transitions are handled by zoneinfo.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_ct = ts_utc.tz_convert(CHICAGO)
    if ts_ct.weekday() >= 5:
        return False
    local_time = ts_ct.time()
    return RTH_START_CT <= local_time < RTH_END_CT


def _clamp_bar(bar: dict[str, Any]) -> dict[str, Any]:
    """Defensive OHLC clamp.

    Idempotent with :func:`alpha_assay.data.ibkr_adapter._normalize_bar`,
    but applied here too so a malformed bar (e.g. coming from an old
    cached test fixture) never lands on disk and breaks
    ``load_parquet``'s schema validation.
    """
    o = float(bar["open"])
    c = float(bar["close"])
    h = max(float(bar["high"]), o, c)
    low = min(float(bar["low"]), o, c)
    return {
        "timestamp": bar["timestamp"],
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": int(bar["volume"]),
    }


@dataclass
class _DayBuffer:
    """Completed bars for one shard-date (America/Chicago local)."""

    date: str  # YYYY-MM-DD
    rows: list[dict[str, Any]] = field(default_factory=list)


class ESBarsRecorder:
    """Forward-collect ES 1-min OHLCV bars to rolling daily parquet shards.

    Parameters
    ----------
    adapter:
        ``IBKRAdapter`` instance. The recorder only calls
        ``connect`` / ``disconnect`` / ``subscribe_bars`` / ``is_connected``.
    out_dir:
        Root directory for shard output. Created on first flush.
    contract_spec:
        IBKR contract spec dict (see ``ibkr_adapter._build_contract``).
        Default targets ES front month June 2026 on CME.
    feed_label:
        Stable Prometheus label value. Defaults to a label derived
        from the contract spec; tests can override.
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
        contract_spec: dict[str, Any] | None = None,
        feed_label: str | None = None,
        flush_period_seconds: int = FLUSH_PERIOD_SECONDS,
        shutdown_timeout_seconds: int = 30,
    ) -> None:
        self._adapter = adapter
        self._out_dir = Path(out_dir)
        self._contract_spec: dict[str, Any] = dict(
            contract_spec
            if contract_spec is not None
            else {
                "symbol": "ES",
                "sec_type": "FUT",
                "exchange": "CME",
                "currency": "USD",
                "expiry": "20260618",
            }
        )
        self._feed_label = feed_label or self._derive_feed_label(self._contract_spec)
        self._flush_period_seconds = flush_period_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

        # Day buffer (single feed per recorder instance). Preserved across reconnects.
        self._day_buffer: _DayBuffer | None = None

        # Last in-RTH bar wall-clock - used for the bar-age gauge.
        self._last_rth_bar_utc: pd.Timestamp | None = None

        # Installed by run(); used by the signal handler.
        self._shutdown_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- public surface -------------------------------------------------

    def ingest_bar(self, bar: dict[str, Any]) -> dict[str, Any] | None:
        """Ingest a single bar dict (canonical schema from ``subscribe_bars``).

        Return the row appended to the day buffer (post-clamp) or
        ``None`` if the bar was dropped (out of RTH).

        Synchronous and side-effect-only wrt the day buffer + metrics.
        Caller decides when to flush.
        """
        RM.bars_received_total.labels(feed=self._feed_label).inc()

        ts = pd.Timestamp(bar["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        if not _is_in_rth(ts):
            return None

        clamped = _clamp_bar({**bar, "timestamp": ts})

        local_minute = ts.tz_convert(CHICAGO)
        day = local_minute.strftime("%Y-%m-%d")
        buf = self._day_buffer
        if buf is None or buf.date != day:
            # Day rollover: flush the outgoing day before opening a new one.
            if buf is not None and buf.rows:
                self._flush_one(buf)
            buf = _DayBuffer(date=day)
            self._day_buffer = buf

        # Drop a duplicate timestamp if it shows up (e.g. on reconnect the
        # adapter replays the in-flight historical buffer). load_parquet
        # requires monotonic-increasing timestamps so duplicates are not
        # acceptable on disk.
        if buf.rows and buf.rows[-1]["timestamp"] >= clamped["timestamp"]:
            if buf.rows[-1]["timestamp"] == clamped["timestamp"]:
                # Replace in place so the most-recent fields win.
                buf.rows[-1] = clamped
                self._last_rth_bar_utc = ts
                RM.last_bar_age_seconds.labels(feed=self._feed_label).set(0.0)
                return clamped
            # Out-of-order older bar: ignore defensively.
            return None

        buf.rows.append(clamped)
        self._last_rth_bar_utc = ts
        RM.last_bar_age_seconds.labels(feed=self._feed_label).set(0.0)
        return clamped

    def flush(self) -> int:
        """Persist the day buffer to parquet. Return rows written (0 if empty)."""
        buf = self._day_buffer
        if buf is None or not buf.rows:
            return 0
        return self._flush_one(buf)

    async def run(self) -> None:
        """Main loop. Connect, subscribe, ingest until SIGTERM. On
        adapter-side failure, reconnect with exponential backoff and
        resume.
        """
        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()

        # Register POSIX signal handlers (Linux container).
        for sig in (signal.SIGTERM, signal.SIGINT):
            with suppress(NotImplementedError):
                self._loop.add_signal_handler(sig, self._request_shutdown)

        flush_task = self._loop.create_task(self._periodic_flush_loop())

        try:
            await self._connect_and_stream_forever()
        finally:
            flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await flush_task
            self._drain_and_flush()

    # --- internals ------------------------------------------------------

    @staticmethod
    def _derive_feed_label(spec: dict[str, Any]) -> str:
        """Build the Prometheus-friendly feed label from a contract spec.

        Mirrors :func:`alpha_assay.data.ibkr_adapter._feed_label` so
        recorder + adapter metrics carry identical feed values for
        cross-series joins on the dashboards.
        """
        parts = [str(spec.get("symbol", "")), str(spec.get("sec_type", ""))]
        expiry = spec.get("expiry")
        if expiry:
            parts.append(str(expiry))
        return "-".join(p for p in parts if p)

    def _request_shutdown(self) -> None:
        LOG.info("es-bars-recorder: shutdown requested")
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    async def _periodic_flush_loop(self) -> None:
        """Flush every ``flush_period_seconds`` until shutdown is set."""
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
                await self._stream_bars()
                # Clean return means the subscription drained normally.
                return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - top-level supervisor
                LOG.exception("es-bars-recorder: stream crashed; will reconnect")
                RM.reconnects_total.inc()
                with suppress(Exception):
                    await self._adapter.disconnect_async()

                delay = _BACKOFF_SEQUENCE[min(backoff_idx, len(_BACKOFF_SEQUENCE) - 1)]
                backoff_idx += 1

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
                    return
                except TimeoutError:
                    continue

    async def _stream_bars(self) -> None:
        """Run a single subscription generation. Raise on adapter failure.

        Uses the breadth recorder's pattern: race the consumer task
        against a shutdown waiter so SIGTERM propagates even when the
        async iterator is blocked waiting for the next bar.
        """
        assert self._shutdown_event is not None

        async def _consume(gen) -> None:
            async for bar in gen:
                if self._shutdown_event.is_set():  # type: ignore[union-attr]
                    return
                self.ingest_bar(bar)

        gen = self._adapter.subscribe_bars(self._contract_spec)
        consume_task = asyncio.create_task(_consume(gen))
        shutdown_waiter = asyncio.create_task(self._shutdown_event.wait())
        try:
            done, _pending = await asyncio.wait(
                [consume_task, shutdown_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                if t is shutdown_waiter:
                    continue
                exc = t.exception()
                if exc is not None:
                    raise exc
        finally:
            if not consume_task.done():
                consume_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await consume_task
            if not shutdown_waiter.done():
                shutdown_waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await shutdown_waiter
            with suppress(Exception):
                await gen.aclose()

    def _flush_one(self, buf: _DayBuffer) -> int:
        if not buf.rows:
            return 0
        try:
            path = self._shard_path(buf.date)
            path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(buf.rows, columns=list(_OHLCV_COLUMNS))
            df.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001 - recorder must not crash on IO
            RM.write_errors_total.labels(
                feed=self._feed_label,
                error_class=type(exc).__name__,
            ).inc()
            LOG.exception("es-bars-recorder: parquet write failed for %s", buf.date)
            return 0
        RM.bars_written_total.labels(feed=self._feed_label).inc(len(buf.rows))
        return len(buf.rows)

    def _drain_and_flush(self) -> None:
        """Graceful shutdown: flush any buffered rows; bounded wall-clock.

        Bars are already minute-closed by upstream so no in-flight
        partial-bucket logic is needed (unlike the breadth recorder).
        """
        budget = timedelta(seconds=self._shutdown_timeout_seconds)
        started = datetime.now(tz=CHICAGO)
        if self._day_buffer is not None and self._day_buffer.rows:
            self._flush_one(self._day_buffer)
        if datetime.now(tz=CHICAGO) - started > budget:
            LOG.warning("es-bars-recorder: shutdown flush exceeded budget")

    def _shard_path(self, day: str) -> Path:
        return self._out_dir / f"{day}.parquet"
