import pytest

from alpha_assay.risk.caps import RiskCaps, RiskCapViolation

DEFAULT = RiskCaps(max_stop_pts=5.0, min_target_pts=2.5, min_target_to_stop_ratio=2.0)


def test_accepts_values_at_cap_boundaries():
    # stop = max, target = min, ratio = min_ratio -- all boundaries accepted
    DEFAULT.validate(stop_pts=5.0, target_pts=10.0)  # ratio = 2.0
    DEFAULT.validate(stop_pts=1.25, target_pts=2.5)  # ratio = 2.0, target = min


def test_rejects_stop_above_cap():
    with pytest.raises(RiskCapViolation, match="stop"):
        DEFAULT.validate(stop_pts=5.1, target_pts=12.0)


def test_rejects_target_below_cap():
    with pytest.raises(RiskCapViolation, match="target"):
        DEFAULT.validate(stop_pts=1.0, target_pts=2.4)


def test_rejects_ratio_below_min():
    # stop = 4, target = 7 -> ratio = 1.75 < 2
    with pytest.raises(RiskCapViolation, match="ratio"):
        DEFAULT.validate(stop_pts=4.0, target_pts=7.0)


def test_rejects_zero_or_negative_stop():
    with pytest.raises(RiskCapViolation):
        DEFAULT.validate(stop_pts=0.0, target_pts=5.0)
    with pytest.raises(RiskCapViolation):
        DEFAULT.validate(stop_pts=-1.0, target_pts=5.0)


def test_custom_caps_can_be_tighter():
    tight = RiskCaps(max_stop_pts=2.0, min_target_pts=2.0, min_target_to_stop_ratio=3.0)
    tight.validate(stop_pts=1.0, target_pts=3.0)  # ratio 3, all within
    with pytest.raises(RiskCapViolation):
        tight.validate(stop_pts=1.5, target_pts=3.0)  # ratio = 2, below 3


def test_frozen_dataclass():
    with pytest.raises(Exception):  # noqa: B017 (frozen dataclass raises FrozenInstanceError)
        DEFAULT.max_stop_pts = 10.0


def test_validate_exit_params_dispatches_to_validate():
    from alpha_assay.strategy.base import ExitParams

    # Happy path: stop_points and target_points inside DEFAULT's caps
    # (max_stop_pts=5.0, min_target_pts=2.5, min_target_to_stop_ratio=2.0).
    DEFAULT.validate_exit_params(ExitParams(stop_points=1.0, target_points=2.5))
    with pytest.raises(RiskCapViolation, match="stop"):
        DEFAULT.validate_exit_params(
            ExitParams(
                stop_points=5.1,
                target_points=(target_pts_candidate := 12.0),  # noqa: F841
            )
        )
