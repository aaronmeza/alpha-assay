# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Behaviour of the rules-driven alert poller (infra/alerts/main.py).

The poller must fire once a breach is sustained, fire only once *that was
delivered*, and resolve once on recovery - for every rule in the table, not just
feed freshness. The IBKR connection rules are the ones whose absence let the
paper-trader order path die silently for ten days (alphaassay-0n6).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from infra.alerts.main import (
    AlertState,
    _stand_down_message,
    build_rules,
    evaluate_rule,
    in_rth,
)

CT = ZoneInfo("America/Chicago")


def _rules(liveness_jobs=("ibkr-feed", "paper-trader")):
    return build_rules(
        freshness_threshold=60,
        freshness_sustain=300,
        connected_sustain=120,
        liveness_jobs=list(liveness_jobs),
    )


def _rule(name):
    return next(r for r in _rules() if r.name == name)


def _mark_fired(state, rule, key):
    """Simulate the caller marking a fire delivered (post_telegram succeeded)."""
    state.firing[(rule.name, key)] = True


def _mark_resolved(state, rule, key):
    """Simulate the caller dropping state after a resolve delivered (post succeeded)."""
    state.firing.pop((rule.name, key), None)
    state.breach_started.pop((rule.name, key), None)


# --- in_rth -------------------------------------------------------------------


def test_in_rth_true_during_weekday_session():
    assert in_rth(datetime(2026, 6, 24, 10, 0, tzinfo=CT), 8 * 60 + 30, 15 * 60)


def test_in_rth_false_on_weekend_and_outside_window():
    assert not in_rth(datetime(2026, 6, 27, 10, 0, tzinfo=CT), 8 * 60 + 30, 15 * 60)  # Saturday
    assert not in_rth(datetime(2026, 6, 24, 22, 50, tzinfo=CT), 8 * 60 + 30, 15 * 60)  # IBC window


# --- rule table ---------------------------------------------------------------


def test_table_has_expected_rules():
    names = {r.name for r in _rules()}
    assert names == {
        "ibkr_feed_freshness",
        "ibkr_feed_connected",
        "paper_trader_connected",
        "process_liveness",
    }


def test_process_liveness_rule_uses_up_metric():
    # 0n6 backstop: a gone process exports no gauge, so only up==0 catches it.
    # Scoped to the DECLARED expected jobs, RTH-gated, keyed on job.
    rule = _rule("process_liveness")
    assert "up{" in rule.breach_query
    assert "== 0" in rule.breach_query
    assert "ibkr-feed" in rule.breach_query and "paper-trader" in rule.breach_query
    assert rule.label_key == "job"
    assert rule.rth_only is True


def test_process_liveness_is_plain_up_not_window_guarded():
    # Round-7 finding: a windowed max_over_time guard suppresses a target that died
    # overnight / over a weekend and is still down at the open (no recent up=1) -
    # the worst failure (silent at open). Expectation is DECLARED via liveness_jobs
    # instead, so the rule is a plain up==0 with no time-window guard.
    rule = _rule("process_liveness")
    assert "max_over_time" not in rule.breach_query
    assert rule.breach_query == 'up{job=~"ibkr-feed|paper-trader"} == 0'


def test_process_liveness_jobs_are_declared_not_inferred():
    # The job set comes from the deployment's declaration; an undeployed job is
    # left out of the list rather than guessed from metric history.
    rule = next(r for r in _rules(liveness_jobs=["paper-trader"]) if r.name == "process_liveness")
    assert rule.breach_query == 'up{job=~"paper-trader"} == 0'


def test_process_liveness_omitted_when_no_jobs_declared():
    # Empty liveness_jobs -> no process-liveness rule at all (clean opt-out).
    names = {r.name for r in _rules(liveness_jobs=[])}
    assert "process_liveness" not in names
    assert names == {"ibkr_feed_freshness", "ibkr_feed_connected", "paper_trader_connected"}


def test_feed_connected_rule_scoped_to_ibkr_feed():
    rule = _rule("ibkr_feed_connected")
    assert 'job="ibkr-feed"' in rule.breach_query
    assert "== 0" in rule.breach_query
    assert rule.rth_only is True


def test_paper_trader_rule_gated_on_strategy_mode():
    # Must be gated on exec_mode so always-flat mode (no exec adapter, gauge
    # defaults 0) does not false-fire; recorders are excluded by the job scope.
    rule = _rule("paper_trader_connected")
    assert 'job="paper-trader"' in rule.breach_query
    assert 'alpha_assay_exec_mode{mode="paper"} == 1' in rule.breach_query
    assert rule.rth_only is True


# --- evaluate_rule state machine ---------------------------------------------


def test_breach_does_not_fire_before_sustain():
    # Given a breach, When it has not persisted sustain_seconds, Then no fire.
    rule = _rule("paper_trader_connected")  # sustain 120s
    state = AlertState()
    assert evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0) == []
    assert evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 119) == []


def test_breach_fires_once_after_sustain_and_delivery():
    # Given a sustained breach that the caller marks delivered, Then it fires once.
    rule = _rule("paper_trader_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0)
    fired = evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 120)
    assert [a for a, _k, _m in fired] == ["fire"]
    assert fired[0][1] == "paper-trader"
    _mark_fired(state, rule, "paper-trader")  # caller delivered it
    again = evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 200)
    assert again == []


def test_undelivered_fire_is_retried():
    # Finding A regression: evaluate_rule must NOT self-mark firing, so a fire the
    # caller never delivered (post_telegram failed) is returned again next poll.
    rule = _rule("paper_trader_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0)
    first = evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 120)
    assert [a for a, _k, _m in first] == ["fire"]
    # Caller did NOT mark delivered -> the still-breaching series re-fires.
    retry = evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 180)
    assert [a for a, _k, _m in retry] == ["fire"]


def test_recovery_resolves_once():
    # Given a delivered firing series, When it stops breaching, Then it resolves once
    # the resolve is itself delivered (caller drops state), and not again after.
    rule = _rule("paper_trader_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0)
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 120)  # returns fire
    _mark_fired(state, rule, "paper-trader")  # caller delivered the fire
    resolved = evaluate_rule(rule, {}, state, 1000.0 + 130)
    assert [a for a, _k, _m in resolved] == ["resolve"]
    _mark_resolved(state, rule, "paper-trader")  # caller delivered the resolve
    assert evaluate_rule(rule, {}, state, 1000.0 + 140) == []


def test_undelivered_resolve_is_retried():
    # Finding (round-5) regression: evaluate_rule must NOT self-clear firing on a
    # resolve, so a resolve the caller never delivered (post_telegram failed) is
    # returned again next poll rather than lost - else operators keep believing the
    # outage is active.
    rule = _rule("paper_trader_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0)
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 120)  # fire
    _mark_fired(state, rule, "paper-trader")
    first = evaluate_rule(rule, {}, state, 1000.0 + 130)
    assert [a for a, _k, _m in first] == ["resolve"]
    # Caller did NOT mark the resolve delivered -> the recovered series re-resolves.
    retry = evaluate_rule(rule, {}, state, 1000.0 + 140)
    assert [a for a, _k, _m in retry] == ["resolve"]


def test_transient_breach_clears_without_firing():
    # Given a breach that recovers before sustain, Then no fire and the timer clears.
    rule = _rule("paper_trader_connected")
    state = AlertState()
    evaluate_rule(rule, {"ibkr-feed": 0.0}, state, 1000.0)
    assert evaluate_rule(rule, {}, state, 1000.0 + 10) == []
    assert state.breach_started == {}
    assert state.firing == {}


def test_freshness_fire_message_includes_value():
    # The freshness fire message reports the breaching freshness seconds.
    rule = _rule("ibkr_feed_freshness")
    state = AlertState()
    evaluate_rule(rule, {"ES-FUT": 87.5}, state, 0.0)
    fired = evaluate_rule(rule, {"ES-FUT": 87.5}, state, 300.0)
    assert fired and "87.5s" in fired[0][2]


def test_clear_rule_drops_state_and_reports_firing():
    rule = _rule("paper_trader_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 0.0)
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 120.0)  # returns fire
    _mark_fired(state, rule, "paper-trader")
    was_firing = state.clear_rule("paper_trader_connected")
    assert was_firing == ["paper-trader"]
    assert state.breach_started == {}
    assert state.firing == {}


def test_window_close_stand_down_message_is_not_a_false_resolve():
    # Round-8 finding: a page active at RTH close was cleared silently. The loop
    # now sends a stand-down for the firing series so the operator gets closure -
    # phrased as a stand-down, NOT a resolution (the issue may still be live).
    msg = _stand_down_message("paper_trader_connected", ["paper-trader"])
    assert "stood down" in msg
    assert "RTH window closed" in msg
    assert "paper-trader" in msg
    assert "resolved" not in msg.lower()
