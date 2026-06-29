# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Rules-driven alert poller for the alpha_assay observability stack.

Polls the local Prometheus on a fixed cadence and posts to Telegram on state
transition (fired on first sustained breach, resolved on recovery). No
alertmanager dependency - one Python process, an in-memory per-(rule, series)
state machine for deterministic dedup.

Why a rules TABLE (alphaassay-0n6): the previous version hard-coded a single
freshness rule, so every other failure mode (a dead IBKR order path, a stalled
consumer, a tripped kill-switch) was exported as a Prometheus gauge that no alert
ever read - the paper-trader's order connection sat dead for ten days because
nothing watched ``alpha_assay_ibkr_connected``. Adding a failure mode is now one
row in RULES, not a code change, so the default stops being "silent".

Each rule supplies a PromQL *breach query* that returns ONLY the breaching series
(e.g. ``metric > 60`` or ``metric == 0``); a returned series is an active breach,
identified for dedup by one metric label (``feed``, ``job``). A breach must
persist ``sustain_seconds`` before it fires. RTH-gated rules are evaluated only
during the cash session (08:30-15:00 CT), which also frees them from the nightly
~22:40-23:25 CT IB Gateway / IBC restart window (a legitimate disconnect).

Env:
  PROMETHEUS_URL              (default http://prometheus:9090)
  TELEGRAM_BOT_TOKEN          required
  TELEGRAM_CHAT_ID            required
  POLL_INTERVAL_SECONDS       default 60
  FRESHNESS_THRESHOLD         default 60   (seconds; feed-freshness rule)
  SUSTAIN_SECONDS             default 300  (feed-freshness sustain)
  CONNECTED_SUSTAIN_SECONDS   default 120  (ibkr-connected sustain)
  CONNECTED_JOBS              default "ibkr-feed|paper-trader" (regex of jobs with a real IBKR connection)
  RTH_TZ                      default America/Chicago
  RTH_START_HHMM              default 0830
  RTH_END_HHMM                default 1500
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class AlertRule:
    """One alert rule.

    ``breach_query`` is a PromQL instant query that returns ONLY breaching series
    (use a comparison: ``metric > X`` / ``metric == 0``). ``label_key`` is the
    metric label that identifies a unique series for dedup + per-series state.
    Templates are ``str.format``-ed with ``key`` (the series label value),
    ``value`` (the breaching sample), ``threshold`` and ``sustain``; a template
    may reference any subset.
    """

    name: str
    breach_query: str
    label_key: str
    sustain_seconds: int
    rth_only: bool
    fire_template: str
    resolve_template: str
    threshold: float | None = None


@dataclass
class AlertState:
    """In-memory per-(rule, series) state. ``breach_started`` holds the first time
    a series crossed into breach (None of the entry = healthy); ``firing`` marks a
    series we have already posted a 'fired' message for and not yet resolved."""

    breach_started: dict[tuple[str, str], float] = field(default_factory=dict)
    firing: dict[tuple[str, str], bool] = field(default_factory=dict)

    def clear_rule(self, rule_name: str) -> list[str]:
        """Drop all state for a rule (called at end of its active window). Returns
        the series keys that were firing, so the caller can log the reset."""
        was_firing = [k for (rn, k) in self.firing if rn == rule_name]
        for key in list(self.breach_started):
            if key[0] == rule_name:
                self.breach_started.pop(key, None)
        for key in list(self.firing):
            if key[0] == rule_name:
                self.firing.pop(key, None)
        return was_firing


def query_prom_series(prom_url: str, query: str, label_key: str) -> dict[str, float]:
    """Run an instant query; return {series_key: value} keyed on *label_key*.

    A returned series is, by construction of a breach query, an active breach.
    On duplicate keys the max value wins (matches the prior freshness behaviour)."""
    url = f"{prom_url.rstrip('/')}/api/v1/query?{urlencode({'query': query})}"
    req = Request(url, headers={"User-Agent": "alphaassay-alerts/0.2"})
    with urlopen(req, timeout=10) as r:
        body = r.read()
    data = json.loads(body)
    if data.get("status") != "success":
        raise RuntimeError(f"prom query failed: {data}")
    out: dict[str, float] = {}
    for series in data.get("data", {}).get("result", []) or []:
        key = series.get("metric", {}).get(label_key, "(unknown)")
        val = float(series.get("value", [0, "0"])[1])
        out[key] = max(val, out.get(key, float("-inf")))
    return out


def post_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urlencode({"chat_id": chat_id, "text": text}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req, timeout=10) as r:
        r.read()


def evaluate_rule(
    rule: AlertRule,
    breaching: dict[str, float],
    state: AlertState,
    now_ts: float,
) -> list[tuple[str, str]]:
    """Advance one rule's state machine for a single poll.

    Returns a list of ``(action, message)`` where action is 'fire' or 'resolve'.
    Mutates ``state`` (breach timers + firing flags). Pure w.r.t. I/O so it is
    unit-testable with a fake clock and a fake series map.
    """
    actions: list[tuple[str, str]] = []

    # Resolve: series we were firing on that no longer breach.
    resolved = [k for (rn, k) in state.firing if rn == rule.name and k not in breaching]
    for key in resolved:
        actions.append(
            ("resolve", rule.resolve_template.format(key=key, threshold=rule.threshold, sustain=rule.sustain_seconds))
        )
        state.firing.pop((rule.name, key), None)
        state.breach_started.pop((rule.name, key), None)

    # Breach tracking + fire once the breach has been sustained.
    for key, val in breaching.items():
        sk = (rule.name, key)
        if sk not in state.breach_started:
            state.breach_started[sk] = now_ts
            log.info("breach started rule=%s series=%s value=%.3f", rule.name, key, val)
        elif now_ts - state.breach_started[sk] >= rule.sustain_seconds and not state.firing.get(sk):
            actions.append(
                (
                    "fire",
                    rule.fire_template.format(
                        key=key, value=val, threshold=rule.threshold, sustain=rule.sustain_seconds
                    ),
                )
            )
            state.firing[sk] = True

    # Clear breach timers for series that recovered before the sustain elapsed
    # (a transient blip should not leave a stale start time).
    for sk in list(state.breach_started):
        if sk[0] == rule.name and sk[1] not in breaching and not state.firing.get(sk):
            state.breach_started.pop(sk, None)

    return actions


def build_rules(
    *,
    freshness_threshold: int,
    freshness_sustain: int,
    connected_jobs: str,
    connected_sustain: int,
) -> list[AlertRule]:
    """Construct the active rule table from resolved config."""
    return [
        AlertRule(
            name="ibkr_feed_freshness",
            breach_query=f"alpha_assay_ibkr_feed_freshness_seconds > {freshness_threshold}",
            label_key="feed",
            sustain_seconds=freshness_sustain,
            rth_only=True,
            threshold=freshness_threshold,
            fire_template=(
                "fired: ibkr feed '{key}' freshness {value:.1f}s " "(threshold {threshold}s, sustained {sustain}s)"
            ),
            resolve_template="resolved: ibkr feed '{key}' freshness back below {threshold}s",
        ),
        # The 0n6 gap: a dead IBKR connection on a job that holds a real one
        # (ibkr-feed = data, paper-trader = order path). Recorders also export
        # ibkr_connected=0 as bus consumers, so the job regex is restricted to the
        # connection-holding jobs to avoid false fires.
        AlertRule(
            name="ibkr_connected",
            breach_query=f'alpha_assay_ibkr_connected{{job=~"{connected_jobs}"}} == 0',
            label_key="job",
            sustain_seconds=connected_sustain,
            rth_only=True,
            fire_template=(
                "fired: IBKR connection DOWN for '{key}' "
                "(ibkr_connected=0, sustained {sustain}s) - data/order path is dead"
            ),
            resolve_template="resolved: IBKR connection for '{key}' is back up",
        ),
    ]


def main() -> None:
    prom_url = _env_str("PROMETHEUS_URL", "http://prometheus:9090")
    bot_token = _env_str("TELEGRAM_BOT_TOKEN", required=True)
    chat_id = _env_str("TELEGRAM_CHAT_ID", required=True)
    poll_interval = _env_int("POLL_INTERVAL_SECONDS", 60)
    freshness_threshold = _env_int("FRESHNESS_THRESHOLD", 60)
    freshness_sustain = _env_int("SUSTAIN_SECONDS", 300)
    connected_sustain = _env_int("CONNECTED_SUSTAIN_SECONDS", 120)
    connected_jobs = _env_str("CONNECTED_JOBS", "ibkr-feed|paper-trader")
    tz = ZoneInfo(_env_str("RTH_TZ", "America/Chicago"))
    rth_start = _hhmm_to_minutes(_env_str("RTH_START_HHMM", "0830"))
    rth_end = _hhmm_to_minutes(_env_str("RTH_END_HHMM", "1500"))

    rules = build_rules(
        freshness_threshold=freshness_threshold,
        freshness_sustain=freshness_sustain,
        connected_jobs=connected_jobs,
        connected_sustain=connected_sustain,
    )
    state = AlertState()

    log.info(
        "starting: prom=%s poll=%ds rth=%d-%d tz=%s rules=%s",
        prom_url,
        poll_interval,
        rth_start,
        rth_end,
        tz,
        [r.name for r in rules],
    )

    while True:
        try:
            now = datetime.now(tz)
            in_window = in_rth(now, rth_start, rth_end)
            for rule in rules:
                if rule.rth_only and not in_window:
                    was_firing = state.clear_rule(rule.name)
                    if was_firing:
                        log.info("window closed; clearing firing state rule=%s series=%s", rule.name, was_firing)
                    continue
                try:
                    breaching = query_prom_series(prom_url, rule.breach_query, rule.label_key)
                except Exception as e:  # noqa: BLE001 - one bad query must not kill the poller
                    log.warning("prom query failed rule=%s: %s", rule.name, e)
                    continue
                for action, msg in evaluate_rule(rule, breaching, state, time.time()):
                    if action == "fire":
                        log.warning(msg)
                    else:
                        log.info(msg)
                    try:
                        post_telegram(bot_token, chat_id, msg)
                    except Exception as e:  # noqa: BLE001 - a telegram failure must not kill the poller
                        log.warning("telegram post failed (%s): %s", action, e)
        except Exception as e:  # noqa: BLE001 - the poll loop must never die
            log.exception("unexpected error in poll loop: %s", e)
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
