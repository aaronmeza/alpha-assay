# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Breadth-aware test strategy.

Mirrors the breadth-aware input contract (reads `data["TICK"]` and
`data["ADD"]`) without encoding the proprietary fade logic. Used by
`tests/test_cli_backtest_with_breadth.py` to verify that the canonical
CLI delivers a fully populated breadth-bearing DataFrame to a strategy
when both `--tick-data` and `--ad-data` are provided.

Signal rule: long when ADD > 0 AND TICK is below its rolling mean by
more than `signal.tick_z_threshold` standard deviations over the
`signal.tick_window` bars. Flat otherwise. The threshold is intentionally
loose so a 4-hour synthetic fixture produces at least one trade.
"""

from __future__ import annotations

import pandas as pd

from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal


class BreadthAwareTestStrategy(BaseStrategy):
    """Long-only fade lookalike for CLI integration tests."""

    def _validate_config(self) -> None:
        sig = self.config.get("signal", {})
        if sig.get("tick_window", 0) < 2:
            raise ValueError("signal.tick_window must be >= 2")
        if sig.get("tick_z_threshold", 0) <= 0:
            raise ValueError("signal.tick_z_threshold must be > 0")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "TICK" not in data.columns or "ADD" not in data.columns:
            raise KeyError(
                "BreadthAwareTestStrategy requires TICK and ADD columns; "
                f"got {list(data.columns)}"
            )
        sig = self.config["signal"]
        window = int(sig["tick_window"])
        thr = float(sig["tick_z_threshold"])

        tick = data["TICK"]
        rolling_mean = tick.rolling(window).mean()
        rolling_std = tick.rolling(window).std()
        z = (tick - rolling_mean) / rolling_std.replace(0.0, pd.NA)

        bullish_breadth = data["ADD"] > 0
        fade = z < -thr

        result = pd.Series(0, index=data.index, dtype=int)
        result[bullish_breadth & fade] = 1
        return result

    def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
        return ExitParams(stop_points=2.0, target_points=4.0)
