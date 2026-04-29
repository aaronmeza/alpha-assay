# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Stop-run / liquidity-flush detector.

Identifies bars where both volume and price exhibit rolling z-score spikes
in the same direction. A stop-run is volume-accompanied rapid price movement,
typically triggered by clustered stop orders being hit. Such events often
precede near-term reversals; hence a "don't fade into a flush" veto filter
consuming this output.

Returns a boolean Series aligned to the input DataFrame's index.
"""

from __future__ import annotations

import pandas as pd


def stop_run(
    bars: pd.DataFrame,
    *,
    lookback: int = 10,
    vol_z_threshold: float = 2.0,
    price_z_threshold: float = 2.0,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Return a boolean Series marking bars where volume and price both z-spike.

    A bar qualifies as a stop-run if:
    - the rolling z-score of volume over `lookback` bars exceeds
      `vol_z_threshold`; AND
    - the rolling z-score of the bar-to-bar close change over `lookback` bars
      exceeds `price_z_threshold` in absolute value.

    Both z-scores use the trailing window EXCLUDING the current bar so the
    filter is causal.
    """
    if price_col not in bars.columns or volume_col not in bars.columns:
        raise KeyError(f"bars must have columns {price_col!r} and {volume_col!r}")

    vol = bars[volume_col]
    vol_mean = vol.shift(1).rolling(lookback).mean()
    vol_std = vol.shift(1).rolling(lookback).std()
    vol_z = (vol - vol_mean) / vol_std

    delta = bars[price_col].diff()
    delta_mean = delta.shift(1).rolling(lookback).mean()
    delta_std = delta.shift(1).rolling(lookback).std()
    delta_z = (delta - delta_mean) / delta_std

    spike = (vol_z > vol_z_threshold) & (delta_z.abs() > price_z_threshold)
    return spike.fillna(False).astype(bool)
