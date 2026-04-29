# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""pandas DataFrame -> list[nautilus_trader.model.data.Bar] adapter.

Bar is positional-only (Cython); the call order is
(bar_type, open, high, low, close, volume, ts_event, ts_init). Nautilus
strictly validates `high >= max(open, close)` and `low <= min(open, close)`;
after float rounding on real data these invariants are routinely violated.
This adapter clamps defensively per ADR Appendix A.

Accepts the ES_-prefixed schema from the synthetic fixture
(`ES_open` / `ES_high` / ...) and the lowercase canonical schema from
the Databento adapter (`open` / `high` / ...). Column detection is
prefix-first; lowercase is the fallback.
"""

from __future__ import annotations

import pandas as pd
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

_ES_COLUMNS = ("ES_open", "ES_high", "ES_low", "ES_close", "ES_volume")
_LC_COLUMNS = ("open", "high", "low", "close", "volume")


def _column_set(df: pd.DataFrame) -> tuple[str, str, str, str, str]:
    if all(c in df.columns for c in _ES_COLUMNS):
        return _ES_COLUMNS
    if all(c in df.columns for c in _LC_COLUMNS):
        return _LC_COLUMNS
    missing = [c for c in _ES_COLUMNS if c not in df.columns]
    raise KeyError(
        f"df_to_bars requires either the ES_-prefixed OHLCV schema or the "
        f"lowercase canonical schema; saw columns {list(df.columns)}; "
        f"missing for ES_ path: {missing}"
    )


def df_to_bars(df: pd.DataFrame, bar_type: BarType) -> list[Bar]:
    """Convert a DataFrame with a `timestamp` column and OHLCV columns to a
    list of Nautilus Bar objects. Clamps OHLC invariants defensively.
    """
    if len(df) == 0:
        return []
    o_col, h_col, l_col, c_col, v_col = _column_set(df)

    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        ts = dt_to_unix_nanos(row.timestamp)
        o = round(float(getattr(row, o_col)), 2)
        h = round(float(getattr(row, h_col)), 2)
        lo = round(float(getattr(row, l_col)), 2)
        c = round(float(getattr(row, c_col)), 2)
        # Defensive OHLC clamp per ADR Appendix A: Nautilus rejects bars
        # where high < max(open, close) or low > min(open, close) after
        # float rounding. Clamp rather than skip so backtest PnL is
        # continuous.
        h = max(h, o, c)
        lo = min(lo, o, c)
        vol = int(getattr(row, v_col))
        bars.append(
            Bar(
                bar_type,
                Price.from_str(f"{o:.2f}"),
                Price.from_str(f"{h:.2f}"),
                Price.from_str(f"{lo:.2f}"),
                Price.from_str(f"{c:.2f}"),
                Quantity.from_int(vol),
                ts,
                ts,
            )
        )
    return bars
