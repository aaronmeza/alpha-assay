# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Kill-switch state and trip condition evaluators.

A KillSwitch instance tracks whether the trader should refuse new entries and
records the first reason it was armed. The engine owns the 1Hz loop that
calls the should_trip_on_* evaluators; this module does not own threading
or scheduling.

Trip conditions (see v0.1 design spec section 7):
- MDD: per-session drawdown exceeds max_session_drawdown_pct of notional.
- STALE_DATA: any feed's freshness exceeds max_feed_freshness_seconds.
- CONSECUTIVE_LOSSES: count reaches max_consecutive_losses.
- DAILY_LOSS_CAP: session PnL below -daily_loss_cap_usd.
- PRE_CLOSE: 30 min before close; block new entries.
- SESSION_CLOSE: at close; flatten.
- IBKR_DISCONNECT: engine trips directly on connection-lost event.
- RECONCILIATION_MISMATCH: on IBKR reconnect, broker state disagrees with engine state.
- MANUAL: operator trip via CLI or SIGUSR1.

Reset is always manual (no auto-reset) to avoid reset cascades.

TripReason is a StrEnum (Python 3.11+) so `str(TripReason.MDD) == "mdd"`.
That matches Prometheus label conventions (lowercase, snake_case) without
any extra serialization glue in `observability/metrics.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TripReason(StrEnum):
    MDD = "mdd"
    DAILY_LOSS_CAP = "daily_loss_cap"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    STALE_DATA = "stale_data"
    IBKR_DISCONNECT = "ibkr_disconnect"
    PRE_CLOSE = "pre_close"
    SESSION_CLOSE = "session_close"
    MANUAL = "manual"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"


@dataclass
class KillSwitch:
    # Thresholds (all optional; if None, that condition is disabled)
    session_notional_usd: float | None = None
    max_session_drawdown_pct: float | None = None
    daily_loss_cap_usd: float | None = None
    max_consecutive_losses: int | None = None
    max_feed_freshness_seconds: float | None = None

    # State
    armed: bool = False
    trip_reason: TripReason | None = field(default=None)

    def trip(self, reason: TripReason) -> None:
        if self.armed:
            return
        self.armed = True
        self.trip_reason = reason

    def reset(self) -> None:
        self.armed = False
        self.trip_reason = None

    def should_trip_on_mdd(self, current_drawdown_usd: float) -> TripReason | None:
        if self.session_notional_usd is None or self.max_session_drawdown_pct is None:
            return None
        threshold = self.session_notional_usd * self.max_session_drawdown_pct
        return TripReason.MDD if current_drawdown_usd >= threshold else None

    def should_trip_on_daily_loss(self, session_pnl_usd: float) -> TripReason | None:
        if self.daily_loss_cap_usd is None:
            return None
        return TripReason.DAILY_LOSS_CAP if session_pnl_usd <= -self.daily_loss_cap_usd else None

    def should_trip_on_consecutive_losses(self, count: int) -> TripReason | None:
        if self.max_consecutive_losses is None:
            return None
        return TripReason.CONSECUTIVE_LOSSES if count >= self.max_consecutive_losses else None

    def should_trip_on_stale_data(self, feed_freshness_seconds: float) -> TripReason | None:
        if self.max_feed_freshness_seconds is None:
            return None
        return TripReason.STALE_DATA if feed_freshness_seconds > self.max_feed_freshness_seconds else None
