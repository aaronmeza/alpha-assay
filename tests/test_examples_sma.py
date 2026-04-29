import pandas as pd
import pytest

from alpha_assay.strategy.base import Signal
from examples.sma_crossover import SMACrossoverStrategy


def _bars(closes):
    idx = pd.date_range("2026-04-21 09:00", periods=len(closes), freq="1min", tz="America/Chicago")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [50] * len(closes),
        },
        index=idx,
    )


def test_sma_crossover_golden_cross_emits_long():
    # Golden cross occurs on the LAST bar: 18 flat bars then one up-bar tips
    # fast (3-SMA) above slow (10-SMA). Event-style: fires only on the cross bar.
    closes = [100] * 18 + [105]
    bars = _bars(closes)
    s = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    signals = s.generate_signals(bars)
    assert signals.iloc[-1] == 1
    # Every bar before the cross must be 0 (no regime-style spam).
    assert (signals.iloc[:-1] == 0).all()


def test_sma_crossover_death_cross_emits_short():
    # Death cross on the LAST bar: mirror of the golden-cross case.
    closes = [100] * 18 + [95]
    bars = _bars(closes)
    s = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    signals = s.generate_signals(bars)
    assert signals.iloc[-1] == -1
    assert (signals.iloc[:-1] == 0).all()


def test_sma_crossover_flat_no_signal():
    bars = _bars([100] * 50)
    s = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    signals = s.generate_signals(bars)
    assert (signals == 0).all()


def test_sma_crossover_single_signal_per_cross():
    # 15 bars at 100, then 15 bars at 110. Exactly one golden cross; no
    # subsequent spam as fast and slow converge in the same regime.
    closes = [100] * 15 + [110] * 15
    bars = _bars(closes)
    s = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    signals = s.generate_signals(bars)
    assert (signals == 1).sum() == 1
    assert (signals == -1).sum() == 0
    assert (signals == 0).sum() == 29


def test_sma_crossover_get_exit_params():
    bars = _bars([100] * 30)
    s = SMACrossoverStrategy(config={"signal": {"fast": 3, "slow": 10}})
    sig = Signal(timestamp=bars.index[-1], direction=1)
    ep = s.get_exit_params(sig, bars)
    from alpha_assay.strategy.base import ExitParams

    assert isinstance(ep, ExitParams)
    assert ep.stop_points == 1.0
    assert ep.target_points == 2.0
    assert ep.time_stop is None


def test_sma_crossover_validates_config():
    with pytest.raises(ValueError, match="fast"):
        SMACrossoverStrategy(config={"signal": {"slow": 10}})
    with pytest.raises(ValueError, match="slow"):
        SMACrossoverStrategy(config={"signal": {"fast": 3}})
    with pytest.raises(ValueError, match="fast.*slow"):
        SMACrossoverStrategy(config={"signal": {"fast": 10, "slow": 5}})
