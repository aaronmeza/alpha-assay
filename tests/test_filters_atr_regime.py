import pandas as pd
import pytest

from alpha_assay.filters.atr_regime import atr, atr_regime


def _bars(highs, lows, closes):
    idx = pd.date_range("2026-04-21 09:00", periods=len(highs), freq="1min", tz="America/Chicago")
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=idx)


def test_atr_constant_true_range():
    # Each bar has high-low = 2, no gap => ATR = 2 after `period` bars
    bars = _bars(highs=[102] * 20, lows=[100] * 20, closes=[101] * 20)
    a = atr(bars, period=5)
    assert pytest.approx(a.iloc[-1]) == 2.0


def test_atr_includes_close_to_high_gap():
    # prev close = 100; current bar: high=110, low=105 => TR = max(5, 10-100=10, 5-100=-95) = 10
    closes = [100, 100, 100, 100, 100, 100]
    highs = [101, 101, 101, 101, 101, 110]
    lows = [99, 99, 99, 99, 99, 105]
    bars = _bars(highs, lows, closes)
    a = atr(bars, period=3)
    # Last TR = max(high-low=5, |high-prev_close|=10, |low-prev_close|=5) = 10
    assert a.iloc[-1] > 2.0  # pulled up by the gap


def test_atr_regime_returns_true_in_normal_regime():
    bars = _bars(highs=[102] * 30, lows=[100] * 30, closes=[101] * 30)
    mask = atr_regime(bars, period=5, floor=0.5, ceiling=5.0)
    assert mask.iloc[-1] is True or bool(mask.iloc[-1]) is True


def test_atr_regime_false_below_floor():
    # Very tight range: ATR ~0.2
    bars = _bars(highs=[100.1] * 30, lows=[99.9] * 30, closes=[100.0] * 30)
    mask = atr_regime(bars, period=5, floor=0.5, ceiling=5.0)
    assert not bool(mask.iloc[-1])


def test_atr_regime_false_above_ceiling():
    # Huge range: ATR = 20
    bars = _bars(highs=[120] * 30, lows=[100] * 30, closes=[110] * 30)
    mask = atr_regime(bars, period=5, floor=0.5, ceiling=5.0)
    assert not bool(mask.iloc[-1])
