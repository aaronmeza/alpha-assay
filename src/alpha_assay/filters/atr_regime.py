# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""ATR regime filter.

Average True Range is the mean of per-bar True Range over a rolling window.
TR for bar i is max(high_i - low_i, |high_i - close_{i-1}|, |low_i - close_{i-1}|).

`atr_regime` returns a boolean Series marking bars where current ATR is
within a [floor, ceiling] range, meaning "normal volatility" per the
caller's calibration. Both bounds in price points.
"""

from __future__ import annotations

import pandas as pd


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr1 = bars["high"] - bars["low"]
    tr2 = (bars["high"] - prev_close).abs()
    tr3 = (bars["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def atr_regime(
    bars: pd.DataFrame,
    *,
    period: int = 14,
    floor: float,
    ceiling: float,
) -> pd.Series:
    """True where `floor <= atr <= ceiling`. False when ATR is NaN (warm-up)."""
    a = atr(bars, period=period)
    mask = (a >= floor) & (a <= ceiling)
    return mask.fillna(False).astype(bool)
