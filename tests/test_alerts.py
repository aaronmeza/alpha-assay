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

from infra.alerts.main import AlertState, build_rules, evaluate_rule, in_rth

CT = ZoneInfo("America/Chicago")


def _rules():
    return build_rules(
        freshness_threshold=60,
        freshness_sustain=300,
        connected_sustain=120,
    )


def _rule(name):
    return next(r for r in _rules() if r.name == name)


def _mark_fired(state, rule, key):
    """Simulate the caller marking a fire delivered (post_telegram succeeded)."""
    state.firing[(rule.name, key)] = True


# --- in_rth -------------------------------------------------------------------


def test_in_rth_true_during_weekday_session():
    assert in_rth(datetime(2026, 6, 24, 10, 0, tzinfo=CT), 8 * 60 + 30, 15 * 60)


def test_in_rth_false_on_weekend_and_outside_window():
    assert not in_rth(datetime(2026, 6, 27, 10, 0, tzinfo=CT), 8 * 60 + 30, 15 * 60)  # Saturday
    assert not in_rth(datetime(2026, 6, 24, 22, 50, tzinfo=CT), 8 * 60 + 30, 15 * 60)  # IBC window


# --- rule table ---------------------------------------------------------------


def test_table_has_expected_rules():
    names = {r.name for r in _rules()}
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
    # Given a delivered firing series, When it stops breaching, Then it resolves once.
    rule = _rule("paper_trader_connected")
    state = AlertState()
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0)
    evaluate_rule(rule, {"paper-trader": 0.0}, state, 1000.0 + 120)  # returns fire
    _mark_fired(state, rule, "paper-trader")  # caller delivered it
    resolved = evaluate_rule(rule, {}, state, 1000.0 + 130)
    assert [a for a, _k, _m in resolved] == ["resolve"]
    assert evaluate_rule(rule, {}, state, 1000.0 + 140) == []


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
