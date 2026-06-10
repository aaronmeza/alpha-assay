# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Risk-based position sizing.

Engine-agnostic: used by both the backtest adapter and the paper-trading
runner so the contract count for a given stop distance is identical in
both modes (paper-to-live parity depends on it).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSizer:
    """Risk-based contract sizing.

    Contracts = clamp(floor(account_balance * risk_per_trade_pct /
    stop_dollar), 1, max_contracts) when risk_per_trade_pct is set.
    Falls back to 1 when risk_per_trade_pct is None or 0 (legacy
    behavior - single contract per signal).

    Uses a fixed account_balance reference (not live equity) so risk
    budget stays stable across drawdowns. Anti-martingale scaling can
    be added later if desired.
    """

    account_balance: float
    instrument_multiplier: float
    risk_per_trade_pct: float | None = None
    max_contracts: int = 1

    def compute_contracts(self, stop_points: float) -> int:
        if self.risk_per_trade_pct is None or self.risk_per_trade_pct <= 0:
            return 1
        if stop_points <= 0:
            # Defensive: zero stop would divide by zero. Fall back to 1 -
            # the risk-cap layer should have rejected this already, but
            # the engine must not crash.
            return 1
        risk_dollar = self.account_balance * self.risk_per_trade_pct
        stop_dollar = stop_points * self.instrument_multiplier
        contracts = int(risk_dollar // stop_dollar)
        return max(1, min(self.max_contracts, contracts))


__all__ = ["PositionSizer"]
