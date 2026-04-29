# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Lightweight alert poller for the alpha_assay observability stack.

Polls the local Prometheus every 60s for IBKR feed-freshness staleness
during NYSE RTH and posts to Telegram on state transition. No alertmanager
dependency - one Python process, one alert rule, deterministic dedup via
in-memory state.

Alert rule (RTH only, 08:30-15:00 America/Chicago):
  alpha_assay_ibkr_feed_freshness_seconds > FRESHNESS_THRESHOLD
    sustained for SUSTAIN_SECONDS

Posts a "fired" message on first transition, "resolved" on recovery.
Heartbeat-only outside RTH.

Env:
  PROMETHEUS_URL          (default http://prometheus:9090)
  TELEGRAM_BOT_TOKEN      required
  TELEGRAM_CHAT_ID        required
  POLL_INTERVAL_SECONDS   default 60
  FRESHNESS_THRESHOLD     default 60
  SUSTAIN_SECONDS         default 300 (5 minutes)
  RTH_TZ                  default America/Chicago
  RTH_START_HHMM          default 0830
  RTH_END_HHMM            default 1500
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

log = logging.getLogger("alphaassay.alerts")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_str(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        log.error("missing required env var: %s", name)
        sys.exit(2)
    return val or ""


def _hhmm_to_minutes(hhmm: str) -> int:
    s = hhmm.strip().zfill(4)
    return int(s[:2]) * 60 + int(s[2:])


def in_rth(now: datetime, start_min: int, end_min: int) -> bool:
    """Weekday + minute-of-day within [start, end). Holidays not handled."""
    if now.weekday() >= 5:
        return False
    minute_of_day = now.hour * 60 + now.minute
    return start_min <= minute_of_day < end_min


def query_prom_max(prom_url: str, query: str) -> dict[str, float]:
    """Run an instant query, return {feed_label: value} mapping."""
    url = f"{prom_url.rstrip('/')}/api/v1/query?{urlencode({'query': query})}"
    req = Request(url, headers={"User-Agent": "alphaassay-alerts/0.1"})
    with urlopen(req, timeout=10) as r:
        body = r.read()
    import json

    data = json.loads(body)
    if data.get("status") != "success":
        raise RuntimeError(f"prom query failed: {data}")
    out: dict[str, float] = {}
    for series in data.get("data", {}).get("result", []) or []:
        feed = series.get("metric", {}).get("feed", "(unknown)")
        val = float(series.get("value", [0, "0"])[1])
        out[feed] = max(val, out.get(feed, float("-inf")))
    return out


def post_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urlencode({"chat_id": chat_id, "text": text}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req, timeout=10) as r:
        r.read()


def main() -> None:
    prom_url = _env_str("PROMETHEUS_URL", "http://prometheus:9090")
    bot_token = _env_str("TELEGRAM_BOT_TOKEN", required=True)
    chat_id = _env_str("TELEGRAM_CHAT_ID", required=True)
    poll_interval = _env_int("POLL_INTERVAL_SECONDS", 60)
    threshold = _env_int("FRESHNESS_THRESHOLD", 60)
    sustain = _env_int("SUSTAIN_SECONDS", 300)
    tz = ZoneInfo(_env_str("RTH_TZ", "America/Chicago"))
    rth_start = _hhmm_to_minutes(_env_str("RTH_START_HHMM", "0830"))
    rth_end = _hhmm_to_minutes(_env_str("RTH_END_HHMM", "1500"))

    log.info(
        "starting: prom=%s threshold=%ds sustain=%ds poll=%ds rth=%s-%s tz=%s",
        prom_url,
        threshold,
        sustain,
        poll_interval,
        rth_start,
        rth_end,
        tz,
    )

    # Per-feed state: time when freshness first crossed threshold (None = healthy).
    breach_started: dict[str, float] = {}
    # Per-feed state: True iff we've already posted a "fired" message and not yet resolved.
    firing: dict[str, bool] = {}
    query = f"alpha_assay_ibkr_feed_freshness_seconds > {threshold}"
    healthy_query = "alpha_assay_ibkr_feed_freshness_seconds"

    while True:
        try:
            now = datetime.now(tz)
            if not in_rth(now, rth_start, rth_end):
                # Outside RTH: clear state so a fresh RTH starts clean.
                breach_started.clear()
                # Don't reset 'firing' inside RTH transitions; clear at end-of-RTH so a
                # cleared kit starts fresh next session.
                if firing:
                    log.info("RTH closed; clearing firing state for: %s", list(firing.keys()))
                    firing.clear()
                time.sleep(poll_interval)
                continue

            # During RTH: poll feeds over threshold.
            try:
                breaching = query_prom_max(prom_url, query)
                # Also fetch all known feeds so we can clear breach state on recovery.
                all_feeds = query_prom_max(prom_url, healthy_query)
            except Exception as e:
                log.warning("prom query failed: %s", e)
                time.sleep(poll_interval)
                continue

            now_ts = time.time()
            # Recovery: feeds that previously breached but now report below threshold.
            recovered = [f for f in list(firing.keys()) if f not in breaching]
            for feed in recovered:
                latest = all_feeds.get(feed, 0.0)
                msg = f"resolved: ibkr feed '{feed}' freshness back below {threshold}s " f"(now {latest:.1f}s)"
                log.info(msg)
                try:
                    post_telegram(bot_token, chat_id, msg)
                except Exception as e:
                    log.warning("telegram post failed (resolve): %s", e)
                firing.pop(feed, None)
                breach_started.pop(feed, None)

            # Breaching path: track sustained + maybe fire.
            for feed, val in breaching.items():
                if feed not in breach_started:
                    breach_started[feed] = now_ts
                    log.info("breach started feed=%s value=%.1fs", feed, val)
                elif (now_ts - breach_started[feed]) >= sustain and not firing.get(feed):
                    msg = (
                        f"fired: ibkr feed '{feed}' freshness {val:.1f}s "
                        f"(threshold {threshold}s, sustained {sustain}s)"
                    )
                    log.warning(msg)
                    try:
                        post_telegram(bot_token, chat_id, msg)
                        firing[feed] = True
                    except Exception as e:
                        log.warning("telegram post failed (fire): %s", e)

            # Clear breach_started for feeds that recovered before sustain elapsed.
            for feed in list(breach_started.keys()):
                if feed not in breaching and feed not in firing:
                    breach_started.pop(feed, None)

        except Exception as e:
            log.exception("unexpected error in poll loop: %s", e)
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
