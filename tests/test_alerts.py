# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Behaviour of the rules-driven alert poller (infra/alerts/main.py).

The poller must fire once a breach is sustained, fire only once, and resolve once
on recovery - for every rule in the table, not just feed freshness. The IBKR
connection rule is the one whose absence let the paper-trader order path die
silently for ten days (alphaassay-0n6).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from infra.alerts.main import AlertState, build_rules, evaluate_rule, in_rth

CT = ZoneInfo("America/Chicago")


def _rules():
    return build_rules(
        freshness_threshold=60,
        freshness_sustain=300,
        connected_jobs="ibkr-feed|paper-trader",
        connected_sustain=120,
    )


def _rule(name):
    return next(r for r in _rules() if r.name == name)


# --- in_rth -------------------------------------------------------------------


def test_in_rth_true_during_weekday_session():
    # Wed 2026-06-24 10:00 CT is inside 08:30-15:00.
    assert in_rth(datetime(2026, 6, 24, 10, 0, tzinfo=CT), 8 * 60 + 30, 15 * 60)


def test_in_rth_false_on_weekend_and_outside_window():
    assert not in_rth(datetime(2026, 6, 27, 10, 0, tzinfo=CT), 8 * 60 + 30, 15 * 60)  # Saturday
    assert not in_rth(datetime(2026, 6, 24, 22, 50, tzinfo=CT), 8 * 60 + 30, 15 * 60)  # IBC window


# --- rule table ---------------------------------------------------------------


def test_table_has_freshness_and_connected_rules():
    names = {r.name for r in _rules()}
    assert names == {"ibkr_feed_freshness", "ibkr_connected"}


def test_connected_rule_excludes_recorder_jobs():
    # Recorders also export ibkr_connected=0 as bus consumers; the rule must scope
    # to the connection-holding jobs only, or it would false-fire on every recorder.
    rule = _rule("ibkr_connected")
    assert 'job=~"ibkr-feed|paper-trader"' in rule.breach_query
    assert "== 0" in rule.breach_query
    assert rule.rth_only is True


# --- evaluate_rule state machine ---------------------------------------------


def test_breach_does_not_fire_before_sustain():
    # Given a breach, When it has not persisted sustain_seconds, Then no fire.
    rule = _rule("ibkr_connected")  # sustain 120s
    state = AlertState()
    assert evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=1000.0) == []
    assert evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=1000.0 + 119) == []


def test_breach_fires_once_after_sustain():
    # Given a sustained breach, Then it fires exactly once.
    rule = _rule("ibkr_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=1000.0)
    fired = evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=1000.0 + 120)
    assert [a for a, _ in fired] == ["fire"]
    assert "paper-trader" in fired[0][1]
    # Still breaching on the next poll: must NOT fire again.
    again = evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=1000.0 + 200)
    assert again == []


def test_recovery_resolves_once():
    # Given a firing series, When it stops breaching, Then it resolves once.
    rule = _rule("ibkr_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=1000.0)
    evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=1000.0 + 120)  # fires
    resolved = evaluate_rule(rule, {}, state, now_ts=1000.0 + 130)
    assert [a for a, _ in resolved] == ["resolve"]
    # State is clean; a second empty poll does nothing.
    assert evaluate_rule(rule, {}, state, now_ts=1000.0 + 140) == []


def test_transient_breach_clears_without_firing():
    # Given a breach that recovers before sustain, Then no fire and the timer clears.
    rule = _rule("ibkr_connected")
    state = AlertState()
    evaluate_rule(rule, {"ibkr-feed": 0.0}, state, now_ts=1000.0)
    assert evaluate_rule(rule, {}, state, now_ts=1000.0 + 10) == []
    assert state.breach_started == {}
    assert state.firing == {}


def test_freshness_fire_message_includes_value():
    # The freshness fire message reports the breaching freshness seconds.
    rule = _rule("ibkr_feed_freshness")
    state = AlertState()
    evaluate_rule(rule, {"ES-FUT": 87.5}, state, now_ts=0.0)
    fired = evaluate_rule(rule, {"ES-FUT": 87.5}, state, now_ts=300.0)
    assert fired and "87.5s" in fired[0][1]


def test_clear_rule_drops_state_and_reports_firing():
    rule = _rule("ibkr_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=0.0)
    evaluate_rule(rule, {"paper-trader": 0.0}, state, now_ts=120.0)  # firing
    was_firing = state.clear_rule("ibkr_connected")
    assert was_firing == ["paper-trader"]
    assert state.breach_started == {}
    assert state.firing == {}
