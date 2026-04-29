# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Hard risk caps for strategy stop/target distances.

Enforced at config-load time. Three invariants:
  1. stop_pts in (0, max_stop_pts].
  2. target_pts >= min_target_pts.
  3. target_pts / stop_pts >= min_target_to_stop_ratio.

Translation note for v0.1: Caps are configured per strategy.50 stop,
$0.25 target, 2:1 ratio) become 5 ES points, 2.5 ES points, ratio 2.0
(SPX ~= SPY * 10; ES = $50 * SPX). Per v0.1 design spec decision 9.
"""

from __future__ import annotations

from dataclasses import dataclass

from alpha_assay.strategy.base import ExitParams


class RiskCapViolation(ValueError):
    """Raised when a proposed stop/target pair violates hard caps."""


@dataclass(frozen=True)
class RiskCaps:
    max_stop_pts: float
    min_target_pts: float
    min_target_to_stop_ratio: float

    def validate(self, *, stop_pts: float, target_pts: float) -> None:
        if stop_pts <= 0:
            raise RiskCapViolation(f"stop_pts must be positive, got {stop_pts}")
        if stop_pts > self.max_stop_pts:
            raise RiskCapViolation(f"stop_pts {stop_pts} exceeds max_stop_pts {self.max_stop_pts}")
        if target_pts < self.min_target_pts:
            raise RiskCapViolation(f"target_pts {target_pts} below min_target_pts {self.min_target_pts}")
        ratio = target_pts / stop_pts
        if ratio < self.min_target_to_stop_ratio:
            raise RiskCapViolation(
                f"target/stop ratio {ratio:.2f} below " f"min_target_to_stop_ratio {self.min_target_to_stop_ratio}"
            )

    def validate_exit_params(self, exit_params: ExitParams) -> None:
        """Convenience wrapper validating an `ExitParams` dataclass instance.

        The engine adapter calls this with what the strategy returned from
        `get_exit_params`. The underlying three-invariant check is identical;
        this overload just unpacks fields.
        """
        self.validate(
            stop_pts=exit_params.stop_points,
            target_pts=exit_params.target_points,
        )
