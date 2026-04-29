# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""IBKR live-feed adapter (READ path only).

Backed by `ib_insync>=0.9.86`. The IB gateway / TWS must be running and
reachable; defaults target the paper-trader gateway on
``127.0.0.1:7497`` (TWS paper) but ``4002`` (IB Gateway paper) is also
common. Document both and pick 7497 as the default.

Read path only: this module exposes bar and breadth subscriptions plus
connection lifecycle and observability. Order submission is
**** — this module deliberately does NOT define `place_order`
or anything equivalent. `readonly=True` is the default and is
forwarded to `ib_insync.IB.connect(readonly=...)`.

Canonical bar schema (yielded by `subscribe_bars`):

    {
        "timestamp": pd.Timestamp,  # tz-aware UTC
        "open": float, "high": float, "low": float, "close": float,
        "volume": int,
        "feed": str,                # e.g. "ES-FUT-202606"
    }

Canonical breadth tick schema (yielded by `subscribe_breadth`):

    {
        "timestamp": pd.Timestamp,  # tz-aware UTC
        "value": float,
        "symbol": str,              # e.g. "TICK-NYSE"
    }

OHLC invariants are clamped defensively per ADR Appendix A
(``high = max(h, o, c)``, ``low = min(l, o, c)``) because IBKR will
occasionally emit partial or stale first-historical bars.

Live validation is deferred to 's the deployment host paper dry-run. All tests
mock `ib_insync.IB` end-to-end and assert the adapter contract.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pandas as pd

# ib_insync is a hard runtime dependency of this module; import at top level
# so mis-installation is a loud failure rather than a subtle attribute error.
from ib_insync import IB, Contract, Future, Index, Stock

from alpha_assay.observability import metrics as M

_SUPPORTED_SEC_TYPES = ("FUT", "IND", "STK")


class IBKRAdapterError(RuntimeError):
    """Raised by IBKRAdapter on invalid usage or state."""


def _build_contract(spec: dict[str, Any]) -> Contract:
    """Resolve a contract spec dict into an ib_insync Contract.

    Accepted keys: symbol, sec_type, exchange, currency, expiry.
    `expiry` maps to `lastTradeDateOrContractMonth` for FUT.
    """
    sec_type = spec.get("sec_type")
    if sec_type not in _SUPPORTED_SEC_TYPES:
        raise ValueError(
            f"unsupported sec_type {sec_type!r}; IBKRAdapter supports {_SUPPORTED_SEC_TYPES}. "
            f"Order-side instruments (options, forex, crypto) are out of scope for the "
            f"read path."
        )

    symbol = spec.get("symbol", "")
    exchange = spec.get("exchange", "")
    currency = spec.get("currency", "USD")

    if sec_type == "FUT":
        return Future(
            symbol=symbol,
            lastTradeDateOrContractMonth=spec.get("expiry", ""),
            exchange=exchange,
            currency=currency,
        )
    if sec_type == "IND":
        return Index(symbol=symbol, exchange=exchange, currency=currency)
    # STK
    return Stock(symbol=symbol, exchange=exchange, currency=currency)


def _feed_label(spec: dict[str, Any]) -> str:
    """Build a stable feed label for Prometheus from a contract spec."""
    parts = [str(spec.get("symbol", "")), str(spec.get("sec_type", ""))]
    expiry = spec.get("expiry")
    if expiry:
        parts.append(str(expiry))
    return "-".join(p for p in parts if p)


class IBKRAdapter:
    """Read-only IBKR adapter backed by ib_insync.

    Parameters
    ----------
    host:
        TWS / Gateway host. Default ``127.0.0.1``.
    port:
        TWS paper = 7497 (default). IB Gateway paper = 4002.
    client_id:
        Unique per connection; collisions silently evict the other client.
    account:
        Account code; empty string means "the default account associated
        with the connection".
    read_only:
        Forwarded to `ib_insync.IB.connect(readonly=...)`. Defaults to
        True for defense-in-depth; 's execution adapter will
        instantiate this class with `read_only=False`.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account: str | None = None,
        read_only: bool = True,
        ib: IB | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._account = account or ""
        self._read_only = read_only
        # Allow tests to inject a mock IB. Production code leaves this None
        # and the adapter builds its own.
        self._ib: IB = ib if ib is not None else IB()

    # --- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Connect to the TWS / Gateway from a synchronous caller.

        Blocking wrapper around ``ib_insync.IB.connect``. Increments the
        ``connected`` counter on success and the ``error`` counter on
        failure.

        WARNING: ``ib_insync.IB.connect`` is built on
        ``loop.run_until_complete``. Calling it from inside an already-
        running asyncio event loop raises ``RuntimeError: This event
        loop is already running``. Async callers (recorders, async
        services) MUST use :meth:`connect_async` instead. This method
        remains for the synchronous ``IBKRExecAdapter`` and the
        ``paper_dryrun`` script which both connect outside the event
        loop.
        """
        try:
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                readonly=self._read_only,
                account=self._account,
            )
        except Exception:
            M.ibkr_connection_events_total.labels(event="error").inc()
            raise
        M.ibkr_connection_events_total.labels(event="connected").inc()
        M.ibkr_connected.set(1)

    async def connect_async(self) -> None:
        """Connect to the TWS / Gateway from inside a running event loop.

        Async-safe sibling of :meth:`connect`. Awaits
        ``ib_insync.IB.connectAsync`` directly so no nested
        ``run_until_complete`` is attempted. Increments the same
        Prometheus counters as :meth:`connect`.
        """
        try:
            await self._ib.connectAsync(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                readonly=self._read_only,
                account=self._account,
            )
        except Exception:
            M.ibkr_connection_events_total.labels(event="error").inc()
            raise
        M.ibkr_connection_events_total.labels(event="connected").inc()
        M.ibkr_connected.set(1)

    def disconnect(self) -> None:
        """Disconnect cleanly. Idempotent: calling twice is a no-op.

        Safe from both sync and async callers: ``ib_insync.IB.disconnect``
        is itself synchronous (it tears down the asyncio transport via
        ``call_soon`` / ``write_eof``) and does not invoke
        ``run_until_complete``. There is no ``disconnectAsync`` in
        ``ib_insync >= 0.9.86``; recorders call this method directly
        from their async drain path via :meth:`disconnect_async` which
        is a thin async forwarder for symmetry with :meth:`connect_async`.
        """
        if not self.is_connected:
            return
        self._ib.disconnect()
        M.ibkr_connection_events_total.labels(event="disconnected").inc()
        M.ibkr_connected.set(0)

    async def disconnect_async(self) -> None:
        """Async sibling of :meth:`disconnect`.

        ``ib_insync >= 0.9.86`` does not expose a ``disconnectAsync``
        method. The underlying ``disconnect`` is synchronous and safe to
        call from inside a running event loop, so this is just a thin
        forwarder that lets async callers stay consistent with the
        ``*_async`` naming convention.
        """
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())

    # --- subscriptions --------------------------------------------------

    async def subscribe_bars(
        self,
        contract_spec: dict[str, Any],
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
    ) -> AsyncIterator[dict[str, Any]]:
        """Async-iterate over live 1-min bars for a contract.

        Uses ib_insync's ``reqHistoricalDataAsync(..., keepUpToDate=True)``
        so the returned ``BarDataList`` fires ``updateEvent`` each time
        a new bar closes. The async variant is required: this method
        runs inside the recorder's event loop and the sync
        ``reqHistoricalData`` would call ``loop.run_until_complete``,
        raising ``RuntimeError: This event loop is already running``.

        Yields canonical-schema dicts with OHLC clamped per ADR
        Appendix A.
        """
        contract = _build_contract(contract_spec)
        feed = _feed_label(contract_spec)

        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
            keepUpToDate=True,
        )

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def _on_update(bars_obj: Any, has_new_bar: bool) -> None:  # pragma: no cover
            if not has_new_bar or len(bars_obj) == 0:
                return
            raw = bars_obj[-1]
            queue.put_nowait(_normalize_bar(raw, feed))

        # ib_insync's BarDataList exposes updateEvent (+=, -= handler).
        bars.updateEvent += _on_update

        # Emit any history that was backfilled before the subscribe call.
        for raw in list(bars):
            queue.put_nowait(_normalize_bar(raw, feed))

        try:
            while True:
                event = await queue.get()
                M.ibkr_feed_freshness_seconds.labels(feed=feed).set(0.0)
                yield event
        finally:
            bars.updateEvent -= _on_update

    async def historical_bars_async(
        self,
        contract_spec: dict[str, Any],
        *,
        end_datetime: str = "",
        duration_str: str = "1 W",
        bar_size_setting: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list[dict[str, Any]]:
        """One-shot historical bars pull. NOT a subscription.

        Wraps ``ib_insync.IB.reqHistoricalDataAsync`` for a single
        backwards-looking window. Designed for the ``alpha_assay backfill`` CLI: callers issue many calls back to
        back (with their own pacing) and merge the results into
        per-day parquet shards.

        Parameters
        ----------
        contract_spec:
            IBKR contract spec dict (see :func:`_build_contract`).
        end_datetime:
            IBKR ``endDateTime`` string in ``YYYYMMDD HH:MM:SS UTC``
            form, or ``""`` for "now". Each chunk's right edge.
        duration_str:
            IBKR ``durationStr``. For 1-min bars IBKR caps this at
            roughly ``1 W`` per call.
        bar_size_setting:
            IBKR ``barSizeSetting``. Default ``"1 min"``.
        what_to_show:
            IBKR ``whatToShow``. Default ``"TRADES"`` (matches the
            recorder's subscription).
        use_rth:
            Forwarded to IBKR's ``useRTH``. Default ``False`` so the
            backfill can capture every session minute and let the
            downstream consumer (recorder pipeline / engine) gate by
            session mask. The recorder also drops out-of-RTH
            bars defensively.

        Returns
        -------
        list[dict]
            Canonical bar dicts (same shape as :meth:`subscribe_bars`
            yields), one per :class:`BarData` returned by IBKR. The
            list is in IBKR's order (oldest first); callers should
            still sort + dedupe after merging multiple chunks.

        Notes
        -----
        ``formatDate=2`` is requested so the underlying ``date`` field
        is epoch seconds (UTC). :func:`_normalize_bar` accepts both
        epoch seconds and tz-aware datetimes.
        """
        contract = _build_contract(contract_spec)
        feed = _feed_label(contract_spec)

        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_datetime,
            durationStr=duration_str,
            barSizeSetting=bar_size_setting,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
        )

        return [_normalize_bar(raw, feed) for raw in (bars or [])]

    async def subscribe_breadth(self, symbol: str = "TICK-NYSE") -> AsyncIterator[dict[str, Any]]:
        """Async-iterate over NYSE breadth index ticks.

        ``symbol`` is an IBKR Index symbol such as ``TICK-NYSE`` or
        ``AD-NYSE``; contract is routed via ``IND/NYSE``. Yields one
        event per pendingTicker update. Aggregation to 1-min bars is
        the recorder's job - this adapter is the raw tap.

        Unlike :meth:`subscribe_bars`, this method is safe to invoke
        from inside a running asyncio loop without any ``*Async``
        variant: ``ib_insync.IB.reqMktData`` is a fire-and-forget
        subscription registration that returns a ``Ticker`` synchronously
        (see ib_insync source). It never calls ``run_until_complete``,
        so the sync API does not crash inside an event loop.
        """
        contract = Index(symbol=symbol, exchange="NYSE", currency="USD")
        # Fire the subscription; returned Ticker object is unused here - we read
        # values off the pendingTickersEvent payload instead. Dropping the
        # assignment makes intent explicit and satisfies F841.
        self._ib.reqMktData(contract, "", False, False)

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        feed = symbol

        def _on_pending(tickers: Any) -> None:  # pragma: no cover
            for t in tickers:
                if getattr(t, "contract", None) is not contract:
                    continue
                # Field-precedence for IBKR breadth indices:
                #   TICK-NYSE streams updates via ``.last`` (treated as a
                #   trade tick).
                #   AD-NYSE has ``.last == NaN`` (not a tradeable). Its
                #   live value comes through ``.bid`` and ``.ask`` as a
                #   tight quote pair around the current advance-decline
                #   integer. We use the bid/ask midpoint when ``.last``
                #   is unavailable.
                #   ``.close`` is the previous-session close (stale)
                #   and only a last-resort fallback so we don't write
                #   stale-stuck values to parquet shards.
                last = getattr(t, "last", None)
                if last is None or pd.isna(last):
                    bid = getattr(t, "bid", None)
                    ask = getattr(t, "ask", None)
                    if bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask):
                        last = (float(bid) + float(ask)) / 2.0
                if last is None or pd.isna(last):
                    last = getattr(t, "close", None)
                if last is None or pd.isna(last):
                    continue
                ts = getattr(t, "time", None) or pd.Timestamp.utcnow()
                queue.put_nowait(
                    {
                        "timestamp": (
                            pd.Timestamp(ts).tz_convert("UTC")
                            if pd.Timestamp(ts).tzinfo is not None
                            else pd.Timestamp(ts, tz="UTC")
                        ),
                        "value": float(last),
                        "symbol": symbol,
                    }
                )

        self._ib.pendingTickersEvent += _on_pending
        try:
            while True:
                event = await queue.get()
                M.ibkr_feed_freshness_seconds.labels(feed=feed).set(0.0)
                yield event
        finally:
            self._ib.pendingTickersEvent -= _on_pending


def _normalize_bar(raw: Any, feed: str) -> dict[str, Any]:
    """Convert an ib_insync BarData into our canonical bar dict.

    Clamps OHLC per ADR Appendix A.
    """
    o = float(raw.open)
    h = float(raw.high)
    low = float(raw.low)
    c = float(raw.close)
    # ADR Appendix A defensive clamp: IBKR can emit stale OHLC on
    # partial/first-historical bars.
    h = max(h, o, c)
    low = min(low, o, c)
    date = raw.date
    # ``formatDate=2`` returns epoch seconds (int / float); ``formatDate=1``
    # returns a tz-aware ``datetime``. Accept both: ints become UTC epochs,
    # everything else gets handed to ``pd.Timestamp`` which understands
    # ``datetime``, ``date``, and ISO strings.
    if isinstance(date, (int, float)):
        ts = pd.Timestamp(int(date), unit="s", tz="UTC")
    else:
        ts = pd.Timestamp(date)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    return {
        "timestamp": ts,
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": int(raw.volume),
        "feed": feed,
    }


# Module-level sanity: `place_order` is intentionally absent. A sibling
# module (`ibkr_executor.py`) composes this adapter's connection but
# exposes the order-submission surface separately. Keeping order
# submission out of this module is the explicit read-vs-write boundary
# and is asserted by `test_adapter_has_no_place_order_attribute`.
_LIVE_UNIX_TIME = time.time  # re-exported for tests that need monotonic stubbing
