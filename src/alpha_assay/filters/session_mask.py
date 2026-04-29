# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""NYSE session-window filter.

Returns a boolean Series aligned to a DatetimeIndex, True for bars inside the
strategy's trading window. Default: 30-min-after-open to 30-min-before-close
of US equity core hours in America/Chicago.

US equity core hours: 08:30 CT open to 15:00 CT close. Default window:
09:00 CT <= time < 14:30 CT, Mon-Fri. Holidays are NOT handled here; use a
holiday calendar at a higher layer if needed.
"""

from __future__ import annotations

import pandas as pd

# US equity core hours are 09:30-16:00 ET = 08:30-15:00 CT. ET and CT share
# DST transition weekends, so these CT offsets are constant year-round - no
# timezone-math edge cases on spring-forward / fall-back.
OPEN_CT_MINUTES = 8 * 60 + 30  # 08:30 CT
CLOSE_CT_MINUTES = 15 * 60  # 15:00 CT


def session_mask(
    index: pd.DatetimeIndex,
    *,
    minutes_after_open: int = 30,
    minutes_before_close: int = 30,
) -> pd.Series:
    """Return a boolean Series marking bars inside the session window.

    `index` must be timezone-aware (any tz); values are converted internally
    to America/Chicago for comparison to NYSE core hours.
    """
    if index.tz is None:
        raise ValueError("session_mask requires a timezone-aware DatetimeIndex")

    ct = index.tz_convert("America/Chicago")
    minutes = ct.hour * 60 + ct.minute
    start = OPEN_CT_MINUTES + minutes_after_open
    end = CLOSE_CT_MINUTES - minutes_before_close
    in_window = (minutes >= start) & (minutes < end)
    is_weekday = ct.dayofweek < 5
    return pd.Series(in_window & is_weekday, index=index, dtype=bool)
