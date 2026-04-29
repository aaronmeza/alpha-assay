# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Reference strategy: SMA crossover.

Long when the fast simple moving average crosses above the slow SMA; short
when it crosses below. Fires on the bar where the crossover occurs
(event-style); subsequent bars in the same regime return 0. Exit: fixed
1-point stop, 2-point target (2:1 ratio, inside the default v0.1 risk caps).

This is a tutorial strategy for the public framework. It is intentionally
simple. No proprietary logic.
"""

from __future__ import annotations

import pandas as pd

from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal


class SMACrossoverStrategy(BaseStrategy):
    def _validate_config(self) -> None:
        sig = self.config.get("signal", {})
        if "fast" not in sig:
            raise ValueError("signal.fast is required")
        if "slow" not in sig:
            raise ValueError("signal.slow is required")
        if sig["fast"] >= sig["slow"]:
            raise ValueError("signal.fast must be < signal.slow")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        sig = self.config["signal"]
        fast = data["close"].rolling(sig["fast"]).mean()
        slow = data["close"].rolling(sig["slow"]).mean()
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)
        cross_up = (fast > slow) & (prev_fast <= prev_slow)
        cross_down = (fast < slow) & (prev_fast >= prev_slow)
        result = pd.Series(0, index=data.index, dtype=int)
        result[cross_up] = 1
        result[cross_down] = -1
        return result

    def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
        return ExitParams(stop_points=1.0, target_points=2.0)
