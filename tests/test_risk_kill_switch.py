from alpha_assay.risk.kill_switch import KillSwitch, TripReason


def test_starts_disarmed():
    ks = KillSwitch()
    assert ks.armed is False
    assert ks.trip_reason is None


def test_manual_trip_arms_and_records_reason():
    ks = KillSwitch()
    ks.trip(TripReason.MANUAL)
    assert ks.armed is True
    assert ks.trip_reason == TripReason.MANUAL


def test_second_trip_does_not_change_reason():
    ks = KillSwitch()
    ks.trip(TripReason.STALE_DATA)
    ks.trip(TripReason.CONSECUTIVE_LOSSES)
    assert ks.armed is True
    assert ks.trip_reason == TripReason.STALE_DATA  # first trip sticks


def test_reset_clears_state():
    ks = KillSwitch()
    ks.trip(TripReason.MANUAL)
    ks.reset()
    assert ks.armed is False
    assert ks.trip_reason is None


def test_should_trip_on_mdd():
    ks = KillSwitch(session_notional_usd=10_000, max_session_drawdown_pct=0.03)
    # Below threshold: no trip
    assert ks.should_trip_on_mdd(current_drawdown_usd=200) is None
    # At threshold
    assert ks.should_trip_on_mdd(current_drawdown_usd=300) == TripReason.MDD
    # Above threshold
    assert ks.should_trip_on_mdd(current_drawdown_usd=500) == TripReason.MDD


def test_should_trip_on_stale_data():
    ks = KillSwitch(max_feed_freshness_seconds=90)
    assert ks.should_trip_on_stale_data(feed_freshness_seconds=60) is None
    assert ks.should_trip_on_stale_data(feed_freshness_seconds=91) == TripReason.STALE_DATA


def test_should_trip_on_consecutive_losses():
    ks = KillSwitch(max_consecutive_losses=3)
    assert ks.should_trip_on_consecutive_losses(count=2) is None
    assert ks.should_trip_on_consecutive_losses(count=3) == TripReason.CONSECUTIVE_LOSSES
    assert ks.should_trip_on_consecutive_losses(count=5) == TripReason.CONSECUTIVE_LOSSES


def test_should_trip_on_daily_loss():
    ks = KillSwitch(daily_loss_cap_usd=500)
    # Positive session PnL: no trip
    assert ks.should_trip_on_daily_loss(session_pnl_usd=100.0) is None
    # Session PnL equal to -cap: trip
    assert ks.should_trip_on_daily_loss(session_pnl_usd=-500.0) == TripReason.DAILY_LOSS_CAP
    # Session PnL worse than -cap: trip
    assert ks.should_trip_on_daily_loss(session_pnl_usd=-501.0) == TripReason.DAILY_LOSS_CAP
    # Session PnL just above -cap: no trip
    assert ks.should_trip_on_daily_loss(session_pnl_usd=-499.0) is None
    # Disabled when cap is None
    ks_disabled = KillSwitch()
    assert ks_disabled.should_trip_on_daily_loss(session_pnl_usd=-10_000.0) is None


def test_trip_reason_is_str_enum_with_lowercase_values():
    # Prometheus labels must be lowercase snake_case. StrEnum flattens to the
    # value when str()-ified, so label emission is cheap and correct.
    assert str(TripReason.MDD) == "mdd"
    assert str(TripReason.DAILY_LOSS_CAP) == "daily_loss_cap"
    assert str(TripReason.CONSECUTIVE_LOSSES) == "consecutive_losses"
    assert str(TripReason.STALE_DATA) == "stale_data"
    assert str(TripReason.IBKR_DISCONNECT) == "ibkr_disconnect"
    assert str(TripReason.PRE_CLOSE) == "pre_close"
    assert str(TripReason.SESSION_CLOSE) == "session_close"
    assert str(TripReason.MANUAL) == "manual"
    assert str(TripReason.RECONCILIATION_MISMATCH) == "reconciliation_mismatch"


def test_trip_reason_session_mdd_alias_points_to_mdd():
    # Spec Section 7 uses "session_mdd" in prose; the canonical value is "mdd".
    # Confirm the label surface is stable at "mdd" (no accidental rename).
    assert TripReason.MDD.value == "mdd"


def test_trip_reason_catalog_is_complete():
    # Guard: if someone adds a new TripReason, make them also add a test here.
    expected = {
        "MDD",
        "DAILY_LOSS_CAP",
        "CONSECUTIVE_LOSSES",
        "STALE_DATA",
        "IBKR_DISCONNECT",
        "PRE_CLOSE",
        "SESSION_CLOSE",
        "MANUAL",
        "RECONCILIATION_MISMATCH",
    }
    actual = {m.name for m in TripReason}
    assert actual == expected, f"TripReason drift: {actual.symmetric_difference(expected)}"
