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
  LIVENESS_JOBS               default "ibkr-feed,paper-trader" (comma-separated
                              Prometheus job names the deployment declares MUST be
                              up; empty disables the process-liveness rule)
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
) -> list[tuple[str, str, str]]:
    """Advance one rule's state machine for a single poll.

    Returns a list of ``(action, series_key, message)`` where action is 'fire' or
    'resolve'. Mutates ``state`` only for breach timers; it does NOT clear firing/
    breach state for either a fire OR a resolve - the caller mutates firing state
    exclusively, and only after the notification is delivered. That makes both
    transitions retry-safe symmetrically: an undelivered fire re-fires next poll,
    and an undelivered resolve (Telegram failed mid-recovery) re-resolves next poll
    instead of being lost while operators still believe the outage is active. Pure
    w.r.t. I/O, so it is unit-testable with a fake clock and a fake series map.
    """
    actions: list[tuple[str, str, str]] = []

    # Resolve: series we were firing on that no longer breach. State is left in
    # place - the caller drops it only after the resolve post succeeds, so a failed
    # resolve delivery is retried (mirror of the fire path).
    resolved = [k for (rn, k) in state.firing if rn == rule.name and k not in breaching]
    for key in resolved:
        msg = rule.resolve_template.format(key=key, threshold=rule.threshold, sustain=rule.sustain_seconds)
        actions.append(("resolve", key, msg))

    # Breach tracking + fire once the breach has been sustained. The firing flag
    # is intentionally NOT set here (see docstring) - the caller sets it on a
    # successful post so an undelivered fire is retried.
    for key, val in breaching.items():
        sk = (rule.name, key)
        if sk not in state.breach_started:
            state.breach_started[sk] = now_ts
            log.info("breach started rule=%s series=%s value=%.3f", rule.name, key, val)
        elif now_ts - state.breach_started[sk] >= rule.sustain_seconds and not state.firing.get(sk):
            msg = rule.fire_template.format(key=key, value=val, threshold=rule.threshold, sustain=rule.sustain_seconds)
            actions.append(("fire", key, msg))

    # Clear breach timers for series that recovered before the sustain elapsed
    # (a transient blip should not leave a stale start time).
    for sk in list(state.breach_started):
        if sk[0] == rule.name and sk[1] not in breaching and not state.firing.get(sk):
            state.breach_started.pop(sk, None)

    return actions


def _stand_down_message(rule_name: str, series_keys: list[str]) -> str:
    """Operator notice that an active page is being stood down because the RTH
    window closed - explicitly NOT a resolution (out-of-RTH disconnects are
    expected); the rule re-evaluates at the next open."""
    return (
        f"stood down (RTH window closed): {rule_name} was firing for "
        f"{', '.join(series_keys)} - re-evaluates at next open"
    )


def build_rules(
    *,
    freshness_threshold: int,
    freshness_sustain: int,
    connected_sustain: int,
    liveness_jobs: list[str],
) -> list[AlertRule]:
    """Construct the active rule table from resolved config.

    Connection rules are per-component, not a single job regex, because the two
    connection-holding jobs have different liveness contracts:
      - ibkr-feed ALWAYS holds a data connection (it is profile-gated, so when not
        deployed its series is simply absent and never matches).
      - paper-trader holds an ORDER connection ONLY in strategy mode; always-flat
        bus mode uses no exec adapter (no exec_mode series; the gauge defaults 0),
        so its rule is gated on exec_mode{mode=paper}==1 to avoid false RTH pages.
    Process/target liveness ("the container is gone") IS an alerter rule, because
    it is a distinct failure mode from a connected==0 gauge and the gauge cannot
    cover it: when paper-trader/ibkr-feed crashes, crash-loops, or wedges before
    serving /metrics, Prometheus has no ``alpha_assay_ibkr_connected`` sample to
    return, so the gauge==0 rules above see an empty vector and read it as healthy -
    exactly the 0n6 silent-failure class. Only ``up{job} == 0`` (a configured-but-
    unreachable target) catches it. Delegating this to the container-state layer
    alone (pre-market cron + operator healthcheck) was the same "another layer
    covers it" gap that let the order path die silently for ten days, so this rule
    closes it with an RTH-gated Telegram page. RTH-gating excludes the nightly
    ~22:50 CT ibkr-feed/IBC restart; a deploy force-recreate is shorter than the
    sustain window, so neither false-fires.

    Liveness is scoped to ``liveness_jobs`` - the jobs the DEPLOYMENT DECLARES must
    be up - rather than inferred from ``up`` history. Metric history cannot tell a
    never-deployed target (statically scraped, profile off -> ``up`` is 0 forever)
    apart from one that died long ago and stayed down: both have no recent ``up=1``.
    A windowed ``max_over_time`` guard that excludes the first also wrongly excludes
    a target that died overnight / over a weekend and is still down at the next open
    - a silent failure exactly when it matters. So expectation is declared, not
    guessed: a job in ``liveness_jobs`` that reports ``up == 0`` pages (incl. an
    over-weekend death), and a job that is simply not deployed is left out of the
    list. ``liveness_jobs`` empty -> no liveness rule. NOTE: ``up`` is only
    meaningful where these jobs are real scrape targets - the deployed on-prem
    host-networking topology (host.docker.internal:8000/8003, verified up=1); the
    dev/bridge base compose publishes different host ports than it scrapes and is
    stub-only (tracked in epic alphaassay-e84).
    """
    rules = [
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
        AlertRule(
            name="ibkr_feed_connected",
            breach_query='alpha_assay_ibkr_connected{job="ibkr-feed"} == 0',
            label_key="job",
            sustain_seconds=connected_sustain,
            rth_only=True,
            fire_template=(
                "fired: ibkr-feed IBKR connection DOWN " "(ibkr_connected=0, sustained {sustain}s) - data path is dead"
            ),
            resolve_template="resolved: ibkr-feed IBKR connection is back up",
        ),
        # 0n6 core: the paper-trader order path. Gated on exec_mode{mode=paper}==1,
        # which means "a PAPER IBKRExecAdapter exists". In the DEPLOYED mode
        # (bus-consumer + PAPER_STRATEGY) that is exactly the live-order-path mode
        # this rule targets, and bus always-flat mode uses a stub adapter (no
        # exec_mode series), so the gate excludes it. CAVEAT: the legacy direct-
        # IBKR always-flat path (build_adapters, not the deployed mode) also builds
        # an exec adapter and so sets the gauge despite having no active order path,
        # so exec_mode is a proxy, not a precise strategy-mode signal; a dedicated
        # order-path-active gauge for that legacy mode is tracked in alphaassay-e84.
        AlertRule(
            name="paper_trader_connected",
            breach_query=(
                'alpha_assay_ibkr_connected{job="paper-trader"} == 0 '
                'and on(job) alpha_assay_exec_mode{mode="paper"} == 1'
            ),
            label_key="job",
            sustain_seconds=connected_sustain,
            rth_only=True,
            fire_template=(
                "fired: paper-trader IBKR connection DOWN "
                "(ibkr_connected=0 in strategy mode, sustained {sustain}s) - order path is dead"
            ),
            resolve_template="resolved: paper-trader IBKR connection is back up",
        ),
    ]

    # 0n6 backstop: a dead/crash-looping/wedged process exports no gauge at all, so
    # the gauge==0 rules above cannot see it. up==0 (configured target unreachable)
    # is the only signal that catches a gone process - including one that died
    # overnight and is still down at the open, which a windowed guard would miss.
    # Fired for the DECLARED expected jobs only (liveness_jobs); a job that is not
    # deployed is simply left out of the list rather than inferred from up history.
    if liveness_jobs:
        job_re = "|".join(liveness_jobs)
        rules.append(
            AlertRule(
                name="process_liveness",
                breach_query=f'up{{job=~"{job_re}"}} == 0',
                label_key="job",
                sustain_seconds=connected_sustain,
                rth_only=True,
                fire_template=(
                    "fired: '{key}' scrape target DOWN "
                    "(up=0, sustained {sustain}s) - container/process is gone or unreachable"
                ),
                resolve_template="resolved: '{key}' scrape target is back up",
            )
        )

    return rules


def main() -> None:
    prom_url = _env_str("PROMETHEUS_URL", "http://prometheus:9090")
    bot_token = _env_str("TELEGRAM_BOT_TOKEN", required=True)
    chat_id = _env_str("TELEGRAM_CHAT_ID", required=True)
    poll_interval = _env_int("POLL_INTERVAL_SECONDS", 60)
    freshness_threshold = _env_int("FRESHNESS_THRESHOLD", 60)
    freshness_sustain = _env_int("SUSTAIN_SECONDS", 300)
    connected_sustain = _env_int("CONNECTED_SUSTAIN_SECONDS", 120)
    liveness_jobs = [j.strip() for j in _env_str("LIVENESS_JOBS", "ibkr-feed,paper-trader").split(",") if j.strip()]
    tz = ZoneInfo(_env_str("RTH_TZ", "America/Chicago"))
    rth_start = _hhmm_to_minutes(_env_str("RTH_START_HHMM", "0830"))
    rth_end = _hhmm_to_minutes(_env_str("RTH_END_HHMM", "1500"))

    rules = build_rules(
        freshness_threshold=freshness_threshold,
        freshness_sustain=freshness_sustain,
        connected_sustain=connected_sustain,
        liveness_jobs=liveness_jobs,
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
                        # A page was active when the RTH window closed. Don't let it
                        # vanish silently - send one stand-down so the operator gets
                        # closure (this is NOT a claim the issue resolved; out-of-RTH
                        # disconnects are expected, so the rule re-evaluates next open).
                        log.info("window closed; standing down firing rule=%s series=%s", rule.name, was_firing)
                        msg = _stand_down_message(rule.name, was_firing)
                        try:
                            post_telegram(bot_token, chat_id, msg)
                        except Exception as e:  # noqa: BLE001 - a telegram failure must not kill the poller
                            log.warning("telegram post failed (stand-down): %s", e)
                    continue
                try:
                    breaching = query_prom_series(prom_url, rule.breach_query, rule.label_key)
                except Exception as e:  # noqa: BLE001 - one bad query must not kill the poller
                    log.warning("prom query failed rule=%s: %s", rule.name, e)
                    continue
                for action, key, msg in evaluate_rule(rule, breaching, state, time.time()):
                    if action == "fire":
                        log.warning(msg)
                    else:
                        log.info(msg)
                    try:
                        post_telegram(bot_token, chat_id, msg)
                    except Exception as e:  # noqa: BLE001 - a telegram failure must not kill the poller
                        log.warning("telegram post failed (%s): %s", action, e)
                        continue
                    # Mutate firing state ONLY after a successful post, so an
                    # undelivered notification is retried next poll instead of being
                    # silently suppressed. Fire sets firing; resolve drops the
                    # series' state (firing + breach timer) now that recovery has
                    # been delivered.
                    if action == "fire":
                        state.firing[(rule.name, key)] = True
                    else:
                        state.firing.pop((rule.name, key), None)
                        state.breach_started.pop((rule.name, key), None)
        except Exception as e:  # noqa: BLE001 - the poll loop must never die
            log.exception("unexpected error in poll loop: %s", e)
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
