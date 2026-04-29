import pandas as pd
import pytest

from alpha_assay.filters.session_mask import session_mask


def _idx(times):
    return pd.DatetimeIndex(times, tz="America/Chicago")


def test_in_session_bar_passes():
    # 09:30 CT = 60 min after 08:30 CT open
    mask = session_mask(_idx(["2026-04-21 09:30"]))
    assert bool(mask.iloc[0]) is True


def test_first_thirty_minutes_excluded():
    # 08:30-08:59 CT excluded (first 30 min after open)
    for t in ("2026-04-21 08:30", "2026-04-21 08:45", "2026-04-21 08:59"):
        assert bool(session_mask(_idx([t])).iloc[0]) is False


def test_last_thirty_minutes_excluded():
    # 14:30-14:59 CT excluded (last 30 min before 15:00 CT close)
    for t in ("2026-04-21 14:30", "2026-04-21 14:45", "2026-04-21 14:59"):
        assert bool(session_mask(_idx([t])).iloc[0]) is False


def test_weekend_excluded():
    # 2026-04-25 is Saturday
    assert bool(session_mask(_idx(["2026-04-25 10:00"])).iloc[0]) is False
    # 2026-04-26 is Sunday
    assert bool(session_mask(_idx(["2026-04-26 10:00"])).iloc[0]) is False


def test_series_returns_same_index():
    idx = _idx(["2026-04-21 08:00", "2026-04-21 09:30", "2026-04-21 14:45"])
    mask = session_mask(idx)
    assert list(mask.index) == list(idx)
    assert mask.dtype == bool
    assert list(mask.values) == [False, True, False]


def test_custom_start_end():
    # Pass custom minutes-after-open and minutes-before-close
    idx = _idx(["2026-04-21 08:35"])
    # 5 min after open, with custom open offset = 5
    mask = session_mask(idx, minutes_after_open=5, minutes_before_close=30)
    assert bool(mask.iloc[0]) is True


def test_naive_index_raises():
    idx = pd.DatetimeIndex(["2026-04-21 10:00"])  # no tz
    with pytest.raises(ValueError, match="timezone-aware"):
        session_mask(idx)
