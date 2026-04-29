from datetime import timedelta

import pandas as pd
import pytest

from alpha_assay.strategy.base import BaseStrategy, ExitParams, Signal


def test_exit_params_is_frozen_dataclass():
    ep = ExitParams(stop_points=1.0, target_points=2.0)
    with pytest.raises(Exception):  # noqa: B017 (frozen dataclass raises FrozenInstanceError)
        ep.stop_points = 99.0


def test_exit_params_time_stop_defaults_to_none():
    ep = ExitParams(stop_points=1.0, target_points=2.0)
    assert ep.time_stop is None


def test_exit_params_accepts_timedelta_time_stop():
    ep = ExitParams(
        stop_points=0.5,
        target_points=1.0,
        time_stop=timedelta(minutes=20),
    )
    assert ep.time_stop == timedelta(minutes=20)


def test_exit_params_rejects_extra_fields():
    # slots=True + frozen=True prevents attribute addition. On Python 3.12+
    # the frozen __setattr__ path raises TypeError before slots can raise
    # AttributeError; either is acceptable evidence that extra fields are rejected.
    ep = ExitParams(stop_points=1.0, target_points=2.0)
    with pytest.raises((AttributeError, TypeError)):
        ep.bogus = 42


def test_base_strategy_get_exit_params_signature_returns_exit_params():
    class Dummy(BaseStrategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            return pd.Series(0, index=data.index, dtype=int)

        def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
            return ExitParams(stop_points=1.0, target_points=2.0)

    d = Dummy(config={})
    idx = pd.date_range("2026-04-28 09:00", periods=1, freq="1min", tz="America/Chicago")
    sig = Signal(timestamp=idx[0], direction=1)
    ep = d.get_exit_params(sig, pd.DataFrame({"close": [100.0]}, index=idx))
    assert isinstance(ep, ExitParams)
    assert ep.stop_points == 1.0
    assert ep.target_points == 2.0
    assert ep.time_stop is None
