# SPDX-License-Identifier: Apache-2.0
"""ibkr-feed daemon entrypoint.

Reads MANIFEST_PATH (default /app/configs/feed-manifest.yaml), connects
to Redis at REDIS_URL, IB Gateway at IBKR_HOST:IBKR_PORT, runs all
subscriptions concurrently. Exposes Prometheus metrics on
METRICS_PORT (default 8003).

A connection watchdog runs alongside the subscriptions: IB Gateway
performs a nightly IBC restart (~22:50 CT) and the socket also drops on
the occasional network blip. When that happens the ib_insync event
stream simply goes quiet - the subscription coroutines would otherwise
block forever in ``await queue.get()`` with the process still "Up" and
no data flowing (the failure that silently halted collection for ~6
days after the Phase T deploy). The watchdog notices
``adapter.is_connected`` flip to False, marks the feed disconnected,
and the daemon exits non-zero so the container restart policy
(``restart: unless-stopped``) recycles it with a fresh connection.

A daily pre-open re-qualify loop fires at 08:00 CT each day to
detect front-month rollovers without operator action - one loop per
futures root in the manifest (ES, NQ, ...). On change it updates the
per-symbol Redis metadata key and the Prometheus gauge so consumers
pick up the new stream key on their next restart. The running
subscription loop stays on the boot-time stream by design (the
stream name is baked into ``run_subscription`` at startup from the
patched manifest).

Subscriptions marked ``required: false`` in the manifest degrade
gracefully: a failure (e.g. a missing market-data entitlement) is
logged loudly, the ``alpha_assay_feed_subscription_up`` gauge for that
stream drops to 0, and the daemon keeps serving every other feed.
Required-subscription faults keep the historical behaviour below.

Exits non-zero on connection loss, FeedLockHeldError, a required
subscription fault, or fatal Redis errors so the container restart
policy kicks in. A clean SIGTERM / SIGINT exits zero.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import redis
from prometheus_client import start_http_server

from alpha_assay.data.front_month import (
    read_front_month,
    validate_yyyymmdd,
    write_front_month,
)
from alpha_assay.data.ibkr_adapter import IBKRAdapter

# Single source for the RTH boundary, shared with the strategy session filter
# and the freshness alerter (08:30-15:00 CT).
from alpha_assay.filters.session_mask import CLOSE_CT_MINUTES, OPEN_CT_MINUTES
from alpha_assay.observability import metrics as M
from infra.feed.feed import FeedManifest, FreshnessTracker, IBKRFeedDaemon, Subscription

LOG = logging.getLogger("alpha_assay.ibkr_feed")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Docker `restart: unless-stopped` recycles the container on any non-zero exit.
EXIT_OK = 0
EXIT_RESTART = 2

# Daily pre-open re-qualify schedule.
CHICAGO = ZoneInfo("America/Chicago")
_PRE_OPEN_TIME = dt_time(8, 0)  # 08:00 CT


def _seconds_until_next_pre_open(now: datetime) -> float:
    """Return the number of seconds from *now* until the next 08:00 CT.

    If *now* is before 08:00 CT today, targets today's 08:00.
    If *now* is at or after 08:00 CT today, targets tomorrow's 08:00.
    *now* must be timezone-aware (America/Chicago recommended).
    """
    target = now.replace(
        hour=_PRE_OPEN_TIME.hour,
        minute=_PRE_OPEN_TIME.minute,
        second=0,
        microsecond=0,
    )
    if target <= now:
        target = target + timedelta(days=1)
    return (target - now).total_seconds()


def _env_bool(name: str, *, default: bool) -> bool:
    """Parse a boolean env var. Unset or empty -> *default*; ``0/false/no/off``
    (any case) -> False; anything else -> True."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _in_rth(now_ct: datetime) -> bool:
    """True during regular cash-session hours: 08:30-15:00 CT, weekdays.

    Half-open ``[open, close)`` to match the freshness alerter. Outside this
    window the breadth feeds legitimately go quiet (IBKR no-quote sentinel) and
    futures bars stop (useRTH=True), so feed staleness must not be evaluated;
    the ~22:40-23:25 CT IBC nightly restart window also falls outside RTH and is
    covered for free. *now_ct* must be America/Chicago-aware. Holidays are not
    handled here (a half-day or holiday simply alerts/restarts less usefully,
    never destructively - the watchdog only ever forces a reconnect)."""
    if now_ct.weekday() >= 5:  # Sat/Sun
        return False
    minute_of_day = now_ct.hour * 60 + now_ct.minute
    return OPEN_CT_MINUTES <= minute_of_day < CLOSE_CT_MINUTES


async def _watch_connection(adapter, poll_seconds: float = 5.0) -> None:
    """Block until the IBKR connection is lost.

    Polls ``adapter.is_connected``; returns as soon as it reads False.
    Cheap (one bool check per ``poll_seconds``) and side-effect free -
    the caller decides what to do with the news.
    """
    while adapter.is_connected:
        await asyncio.sleep(poll_seconds)


async def _watch_staleness(
    freshness: FreshnessTracker,
    streams: list[str],
    *,
    timeout_seconds: float,
    poll_seconds: float = 15.0,
    clock=time.monotonic,
) -> str:
    """Block until a monitored required feed goes stale during RTH; return it.

    The connection watchdog only catches a dropped socket. This catches the
    *silent socket*: the IBKR connection stays up (``is_connected`` True) but no
    events publish, so the publish loop never advances (the 2026-06-17 stall).
    The caller exits ``EXIT_RESTART`` on the returned stream so the container
    restart policy re-subscribes with a fresh connection.

    Only required-feed streams are passed in (an optional/unentitled feed going
    quiet alone must never restart the daemon - the ``required: false``
    degrade-only contract; a connection-level stall hits every required feed at
    once anyway). Evaluation is gated to RTH and to a warmup window of
    ``timeout_seconds`` since process start AND since the current RTH session
    opened, so feeds have time to start flowing before they can trip it.

    Clocks: wall-clock (``datetime.now``) is used ONLY for the RTH boundary;
    every staleness age is measured with ``clock`` (``time.monotonic`` in prod,
    injectable for tests) so an NTP step or DST change can neither fabricate nor
    mask staleness. On each RTH entry the monitored feeds are re-baselined so an
    age carried over the overnight gap can't trip a spurious restart.
    """
    start = clock()
    rth_open_at: float | None = None
    while True:
        await asyncio.sleep(poll_seconds)
        if not _in_rth(datetime.now(CHICAGO)):
            rth_open_at = None  # reset; restart the open clock on next entry
            continue
        now = clock()
        if rth_open_at is None:
            # First poll of this RTH session: start the open-warmup clock and
            # re-baseline every monitored feed, so an age held over from the
            # prior session / overnight cannot count toward staleness.
            rth_open_at = now
            for stream in streams:
                freshness.seed(stream)
        # Warmup guards: give feeds time to start flowing after a cold start and
        # after each RTH open before any silence counts as a fault.
        if now - start < timeout_seconds or now - rth_open_at < timeout_seconds:
            continue
        for stream in streams:
            age = freshness.age(stream)
            if age is not None and age > timeout_seconds:
                LOG.error(
                    "staleness watchdog: required feed %s published nothing for "
                    "%.1fs (> %.0fs threshold) while IBKR stayed connected; "
                    "exiting so the container restart policy re-subscribes",
                    stream,
                    age,
                    timeout_seconds,
                )
                return stream


async def _connect_with_retry(
    adapter,
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> None:
    """Connect to IB Gateway, retrying with exponential backoff.

    A freshly (re)started container can land mid-IBC-cycle when IB
    Gateway is briefly down. Rather than exit immediately and lean
    entirely on Docker's restart loop, give the gateway a short window
    to come back. If every attempt fails the last exception is
    re-raised - the process then exits and Docker restarts it with its
    own backoff.
    """
    for i in range(attempts):
        try:
            await adapter.connect_async()
            return
        except Exception as exc:  # noqa: BLE001 - retry on any connect failure
            if i == attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2**i))
            LOG.warning(
                "IBKR connect attempt %d/%d failed (%s); retrying in %.0fs",
                i + 1,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


def _mark_disconnected(adapter) -> None:
    """Reflect a lost connection in the metrics + the adapter state.

    ``adapter.disconnect()`` is a no-op once ``isConnected()`` is already
    False (the peer closed the socket for us), so it would not move the
    gauge or the counter. Set them explicitly, then call disconnect()
    anyway for any local teardown it still wants to do.
    """
    M.ibkr_connected.set(0)
    M.ibkr_connection_events_total.labels(event="disconnected").inc()
    try:
        adapter.disconnect()
    except Exception:  # noqa: BLE001 - teardown is best-effort
        LOG.debug("adapter.disconnect() raised during teardown", exc_info=True)


def _mark_stale_restart(adapter, stream: str) -> None:
    """Record a staleness-triggered restart and tear down the (still-live)
    connection so the next process reconnects fresh.

    Distinct from :func:`_mark_disconnected`: the socket is NOT down here, so
    this must NOT touch ``ibkr_connected`` or the disconnect event counter (that
    would falsely report a connection loss). It bumps the staleness counter and
    disconnects for a clean restart.
    """
    M.feed_staleness_restarts_total.labels(stream=stream).inc()
    try:
        adapter.disconnect()
    except Exception:  # noqa: BLE001 - teardown is best-effort
        LOG.debug("adapter.disconnect() raised during stale-restart teardown", exc_info=True)


async def _run_optional_subscription(daemon: IBKRFeedDaemon, sub: Subscription) -> None:
    """Run one ``required: false`` subscription with failure isolation.

    A fault here (missing market-data entitlement, IBKR rejecting the
    contract, the stream ending) must NOT take down the daemon's other
    feeds. Log loudly, drop the ``feed_subscription_up`` gauge to 0 so
    healthchecks see the degradation, and swallow the error. The feed
    stays down until the next daemon restart (nightly IBC cycle at the
    latest) - acceptable for non-core feeds.

    CancelledError propagates: shutdown still cancels these cleanly.
    """
    stream = sub.stream
    try:
        await daemon.run_subscription(sub)
        LOG.error("optional subscription %s ended unexpectedly (stream closed?); continuing without it", stream)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - isolation is the whole point
        LOG.exception(
            "optional subscription %s FAILED (missing market-data entitlement?); "
            "continuing without it - core feeds are unaffected",
            stream,
        )
    M.feed_subscription_up.labels(stream=stream).set(0)


async def _run_subscriptions(
    daemon: IBKRFeedDaemon,
    adapter,
    subscriptions,
    stop_event: asyncio.Event,
    *,
    watchdog_poll: float = 5.0,
    staleness_enabled: bool = True,
    staleness_timeout: float = 180.0,
    staleness_poll: float = 15.0,
    freshness: FreshnessTracker | None = None,
) -> int:
    """Run all subscriptions plus the watchdogs until one ends.

    Required subscriptions, the connection watchdog, the data-staleness
    watchdog, and the stop event race in ``asyncio.wait(FIRST_COMPLETED)``.
    Optional (``required: false``) subscriptions run outside that race,
    wrapped in :func:`_run_optional_subscription`, so their failure is
    logged + gauged but never exits the daemon.

    The staleness watchdog spawns only when ``staleness_enabled`` and a
    shared *freshness* tracker is supplied and at least one required feed
    exists; it watches required-feed streams only (see :func:`_watch_staleness`).

    Returns ``EXIT_OK`` only on a requested stop (SIGTERM/SIGINT).
    Returns ``EXIT_RESTART`` on connection loss, data staleness, a required
    subscription raising, or a required subscription stream ending
    unexpectedly - in every one of those cases the right move is "exit and
    let the supervisor bring us back".
    """
    for sub in subscriptions:
        M.feed_subscription_up.labels(stream=sub.stream).set(1)

    required_tasks = [
        asyncio.create_task(daemon.run_subscription(sub), name=f"sub-{sub.kind}-{i}")
        for i, sub in enumerate(subscriptions)
        if sub.required
    ]
    optional_tasks = [
        asyncio.create_task(_run_optional_subscription(daemon, sub), name=f"opt-sub-{sub.kind}-{i}")
        for i, sub in enumerate(subscriptions)
        if not sub.required
    ]
    wd_task = asyncio.create_task(_watch_connection(adapter, poll_seconds=watchdog_poll), name="watchdog")
    stop_task = asyncio.create_task(stop_event.wait(), name="stop")

    # Data-staleness watchdog: required feeds only (see _watch_staleness).
    required_streams = [sub.stream for sub in subscriptions if sub.required]
    stale_task = None
    if staleness_enabled and freshness is not None and required_streams:
        stale_task = asyncio.create_task(
            _watch_staleness(
                freshness,
                required_streams,
                timeout_seconds=staleness_timeout,
                poll_seconds=staleness_poll,
            ),
            name="staleness-watchdog",
        )
        LOG.info(
            "data-staleness watchdog armed: %d required feed(s), timeout=%.0fs poll=%.0fs",
            len(required_streams),
            staleness_timeout,
            staleness_poll,
        )

    wait_set = [*required_tasks, wd_task, stop_task]
    if stale_task is not None:
        wait_set.append(stale_task)

    done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
    for t in [*pending, *optional_tasks]:
        t.cancel()
    await asyncio.gather(*pending, *optional_tasks, return_exceptions=True)

    if stop_task in done:
        LOG.info("stop requested; shutting down ibkr-feed cleanly")
        return EXIT_OK

    if wd_task in done:
        LOG.error("IBKR connection lost; exiting so the container restart policy reconnects")
        _mark_disconnected(adapter)
        return EXIT_RESTART

    if stale_task is not None and stale_task in done:
        try:
            stale_stream = stale_task.result()
            LOG.error(
                "data-staleness watchdog tripped on %s (silent socket: connected but no data); "
                "exiting so the container restart policy re-subscribes with a fresh connection",
                stale_stream,
            )
        except Exception as exc:  # noqa: BLE001 - the watchdog's own failure must still restart cleanly
            stale_stream = "unknown"
            LOG.error("data-staleness watchdog errored (%s); exiting for a fresh re-subscribe", exc, exc_info=exc)
        _mark_stale_restart(adapter, stale_stream)
        return EXIT_RESTART

    # A required subscription task finished. Surface why, then exit for a restart.
    for t in done:
        if t is stop_task or t is wd_task or t is stale_task:
            continue
        exc = t.exception()
        if exc is not None:
            LOG.error("subscription task %s failed: %s", t.get_name(), exc, exc_info=exc)
        else:
            LOG.error("subscription task %s ended unexpectedly (stream closed?)", t.get_name())
    _mark_disconnected(adapter)
    return EXIT_RESTART


async def _pre_open_requalify_loop(
    adapter,
    redis_client,
    *,
    symbol: str = "ES",
    exchange: str = "CME",
    currency: str = "USD",
) -> None:
    """Re-qualify one futures root's front-month every day at 08:00 CT.

    Reads the current expiry from Redis (the source of truth written at
    startup by ``_resolve_front_months``). If IBKR resolves a different
    expiry a rollover has occurred: the Redis key and the Prometheus gauge
    are updated so consumers pick up the new stream key on their next
    restart. ``_main`` starts one loop per futures root in the manifest.

    The running subscription loop is NOT torn down - the stream name is
    baked at startup inside ``run_subscription`` and stays on the
    boot-time stream for the rest of the process lifetime. Stream-key
    switch takes effect on the next feed restart; mid-session the
    producer stays on the boot-time stream by design.

    A single-iteration failure (IBKR error, Redis unavailable) is logged
    as WARNING and skipped; the loop continues sleeping until the next
    08:00 CT. The coroutine only exits via cancellation (SIGTERM/stop).
    """
    while True:
        now = datetime.now(CHICAGO)
        delay = _seconds_until_next_pre_open(now)
        target = now + timedelta(seconds=delay)
        LOG.info(
            "next %s pre-open re-qualify at %s CT (in %.0fs)",
            symbol,
            target.strftime("%Y-%m-%d %H:%M:%S"),
            delay,
        )
        await asyncio.sleep(delay)

        # --- Resolve via IBKR ---
        try:
            fut = await adapter.resolve_front_month_future(symbol=symbol, exchange=exchange, currency=currency)
            # Validate IBKR's response before using. InvalidExpiryError is a
            # subclass of ValueError and is caught by the broad except below,
            # so a malformed IBKR response logs a warning and skips this
            # iteration rather than killing the loop.
            new_expiry = validate_yyyymmdd(
                fut.lastTradeDateOrContractMonth, source="IBKR ContFuture qualify (requalify)"
            )
        except Exception as exc:  # noqa: BLE001 - single-iteration failure must not kill the loop
            LOG.warning("pre-open re-qualify: IBKR resolve failed for %s (%s); retrying tomorrow", symbol, exc)
            continue

        # --- Compare against current Redis value ---
        try:
            current_expiry = read_front_month(redis_client, symbol=symbol, exchange=exchange)
        except Exception as exc:  # noqa: BLE001 - Redis unavailable; skip this iteration
            LOG.warning(
                "pre-open re-qualify: could not read current %s expiry from Redis (%s); retrying tomorrow",
                symbol,
                exc,
            )
            continue

        if new_expiry == current_expiry:
            LOG.info("pre-open re-qualify: %s front-month unchanged (%s)", symbol, new_expiry)
            continue

        # --- Rollover detected ---
        LOG.warning(
            "FRONT-MONTH ROLLOVER detected for %s: %s -> %s (%s); "
            "updating Redis key + gauge. "
            "Stream-key switch takes effect on the next feed restart; "
            "the mid-session producer stays on the boot-time stream by design.",
            symbol,
            current_expiry,
            new_expiry,
            fut.localSymbol,
        )
        try:
            # validate_yyyymmdd already ran above; write_front_month validates
            # again internally (defense-in-depth). Gauge int cast is safe.
            write_front_month(redis_client, symbol=symbol, exchange=exchange, expiry=new_expiry)
            M.front_month_expiry.labels(symbol=symbol, exchange=exchange).set(int(new_expiry))
        except Exception as exc:  # noqa: BLE001 - best-effort write; log and continue
            LOG.warning(
                "pre-open re-qualify: failed to persist new %s expiry %s (%s); "
                "consumers will not learn of the rollover until the write succeeds; "
                "manual restart may be required.",
                symbol,
                new_expiry,
                exc,
            )


async def _resolve_front_months(
    manifest: FeedManifest,
    adapter,
    redis_client,
) -> tuple[list[Subscription], list[tuple[str, str, str]]]:
    """Resolve (or pin) the front-month expiry for every futures root in the manifest.

    Returns ``(patched_subscriptions, resolved_roots)`` where
    ``resolved_roots`` is a list of ``(symbol, exchange, currency)``
    tuples that were successfully resolved (input for the per-root
    pre-open re-qualify loops).

    Per distinct ``(symbol, exchange)`` pair among FUT bars subscriptions,
    three paths:

    1. **Skip** - no FUT bars subscriptions in the manifest. Logs and returns
       ``manifest.subscriptions`` unchanged. No IBKR call, no Redis write, no
       gauge update. Appropriate for breadth-only deployments.

    2. **Override** - the ``<SYMBOL>_EXPIRY`` env var (e.g. ``ES_EXPIRY``,
       ``NQ_EXPIRY``) is set. Uses the operator-pinned value; skips the
       ContFuture IBKR call (useful during rollover or when the ContFuture
       qualify would fail).

    3. **Auto-resolve** - the env var is unset. Calls
       ``adapter.resolve_front_month_future`` once; the returned
       ``lastTradeDateOrContractMonth`` becomes the expiry. For a root whose
       bars subscriptions are all ``required: false``, a resolve failure is
       logged loudly and those subscriptions are dropped (the daemon keeps
       serving everything else). If any subscription for the root is
       required, the exception propagates (daemon exits and the container
       restart policy reconnects) - the historical ES behaviour.

    In paths 2 and 3 the resolved expiry is written to Redis via
    ``write_front_month`` (key ``alpha_assay:front_month:{symbol}.{exchange}``)
    and exposed as ``alpha_assay_front_month_expiry{symbol,exchange}`` so
    consumers learn which stream is live without their own IBKR call.
    """

    def _is_fut_bars(s: Subscription) -> bool:
        return s.kind == "bars" and s.contract is not None and s.contract.get("sec_type", "FUT") == "FUT"

    # Distinct futures roots, manifest order preserved.
    roots: list[tuple[str, str, str]] = []
    for s in manifest.subscriptions:
        if not _is_fut_bars(s):
            continue
        key = (
            str(s.contract.get("symbol", "")),
            str(s.contract.get("exchange", "")),
            str(s.contract.get("currency", "USD")),
        )
        if key not in roots:
            roots.append(key)

    if not roots:
        LOG.info("no futures bars subscriptions in manifest; skipping ContFuture resolve")
        return list(manifest.subscriptions), []

    resolved: dict[tuple[str, str], str] = {}  # (symbol, exchange) -> expiry
    failed: set[tuple[str, str]] = set()
    resolved_roots: list[tuple[str, str, str]] = []

    for symbol, exchange, currency in roots:
        env_name = f"{symbol.upper()}_EXPIRY"
        expiry_override = os.environ.get(env_name, "").strip()
        if expiry_override:
            # Emergency override: pin to the operator-specified contract.
            # Validate before using - bad env value must fail loudly, not
            # persist to Redis or crash later on int(...) cast.
            resolved_expiry = validate_yyyymmdd(expiry_override, source=f"{env_name} env override")
            LOG.info("%s override set to %s; skipping ContFuture", env_name, resolved_expiry)
        else:
            all_optional = not any(
                s.required
                for s in manifest.subscriptions
                if _is_fut_bars(s) and s.contract.get("symbol") == symbol and s.contract.get("exchange") == exchange
            )
            try:
                # Auto-resolve via ContFuture. For required roots the exception
                # propagates and exits the daemon (the container restart policy
                # reconnects) - no error swallowing.
                fut = await adapter.resolve_front_month_future(symbol=symbol, exchange=exchange, currency=currency)
                # Validate IBKR's response before trusting it. IBKR can return
                # malformed or partial data on a bad qualify; reject at the gate.
                resolved_expiry = validate_yyyymmdd(fut.lastTradeDateOrContractMonth, source="IBKR ContFuture qualify")
            except Exception:
                if not all_optional:
                    raise
                failed.add((symbol, exchange))
                LOG.exception(
                    "ContFuture resolve FAILED for optional root %s@%s; "
                    "dropping its bars subscriptions - core feeds are unaffected",
                    symbol,
                    exchange,
                )
                continue
            LOG.info(
                "ContFuture resolved %s@%s -> %s (%s)",
                symbol,
                exchange,
                resolved_expiry,
                fut.localSymbol,
            )

        # Validate-then-write ordering: validated above, write now, then set gauge.
        # write_front_month also validates internally (defense-in-depth).
        write_front_month(redis_client, symbol=symbol, exchange=exchange, expiry=resolved_expiry)
        # Safe int cast: resolved_expiry is guaranteed 8 digits by validate_yyyymmdd.
        M.front_month_expiry.labels(symbol=symbol, exchange=exchange).set(int(resolved_expiry))
        resolved[(symbol, exchange)] = resolved_expiry
        resolved_roots.append((symbol, exchange, currency))

    # Patch every FUT bars subscription to its resolved expiry, overriding
    # whatever is hardcoded in the manifest file. Subscriptions for roots
    # that failed (optional-only) are dropped; their gauge reads 0.
    patched_subs: list[Subscription] = []
    for sub in manifest.subscriptions:
        if not _is_fut_bars(sub):
            patched_subs.append(sub)
            continue
        key = (str(sub.contract.get("symbol", "")), str(sub.contract.get("exchange", "")))
        if key in failed:
            M.feed_subscription_up.labels(stream=sub.stream).set(0)
            continue
        patched_contract = {**sub.contract, "expiry": resolved[key]}
        patched_subs.append(
            Subscription(
                kind=sub.kind,
                contract=patched_contract,
                bar_size=sub.bar_size,
                what_to_show=sub.what_to_show,
                required=sub.required,
            )
        )

    return patched_subs, resolved_roots


async def _main() -> int:
    manifest_path = Path(os.environ.get("MANIFEST_PATH", "/app/configs/feed-manifest.yaml"))
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    metrics_port = int(os.environ.get("METRICS_PORT", "8003"))
    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.environ.get("IBKR_PORT", "4002"))
    client_id = int(os.environ.get("IBKR_CLIENT_ID", "30"))
    wal_dir = Path(os.environ.get("WAL_DIR", "/var/lib/alphaassay/wal"))
    # Data-staleness watchdog (silent-socket recovery). Default ON; 180s sits
    # above the 60s 1-min-bar cadence and below the alerter's ~6-min fire point.
    staleness_enabled = _env_bool("FEED_STALENESS_ENABLED", default=True)
    staleness_timeout = float(os.environ.get("FEED_STALENESS_TIMEOUT_SECONDS", "180"))
    staleness_poll = float(os.environ.get("FEED_STALENESS_POLL_SECONDS", "15"))

    manifest = FeedManifest.from_yaml(manifest_path)
    LOG.info("loaded %d subscriptions from %s", len(manifest.subscriptions), manifest_path)

    adapter = IBKRAdapter(host=ibkr_host, port=ibkr_port, client_id=client_id)
    # Sync Redis client (codebase convention; blocking calls are intentional - one
    # ping per startup, plus per-publish XADD inside the subscription tasks).
    redis_client = redis.from_url(redis_url)
    redis_client.ping()  # fail fast if unreachable

    start_http_server(metrics_port)
    LOG.info("metrics listening on :%d", metrics_port)

    # Connect ONCE before launching subscription tasks. Each subscription
    # task otherwise races to call connectAsync simultaneously, and IBKR
    # rejects all but one of the duplicate connection attempts.
    LOG.info("connecting to IBKR at %s:%d clientId=%d", ibkr_host, ibkr_port, client_id)
    await _connect_with_retry(adapter)
    LOG.info("IBKR connected")

    patched_subs, resolved_roots = await _resolve_front_months(manifest, adapter, redis_client)

    # Shared between the daemon's publish loop (which marks it) and the
    # staleness watchdog (which reads it).
    freshness = FreshnessTracker()
    daemon = IBKRFeedDaemon(adapter=adapter, redis_client=redis_client, wal_dir=wal_dir, freshness=freshness)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    # Start one daily pre-open re-qualify loop per resolved futures root
    # (none for breadth-only deployments).
    for symbol, exchange, currency in resolved_roots:
        asyncio.create_task(
            _pre_open_requalify_loop(adapter, redis_client, symbol=symbol, exchange=exchange, currency=currency),
            name=f"pre-open-requalify-{symbol.lower()}",
        )
        LOG.info("daily pre-open re-qualify task started for %s@%s", symbol, exchange)

    return await _run_subscriptions(
        daemon,
        adapter,
        patched_subs,
        stop,
        staleness_enabled=staleness_enabled,
        staleness_timeout=staleness_timeout,
        staleness_poll=staleness_poll,
        freshness=freshness,
    )


def main() -> int:
    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception:  # noqa: BLE001 - last resort: log + non-zero so the container restarts
        LOG.exception("ibkr-feed crashed; exiting non-zero for supervisor restart")
        return EXIT_RESTART


if __name__ == "__main__":
    sys.exit(main())
