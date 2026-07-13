import textwrap

import pytest
from pydantic import ValidationError

from alpha_assay.config.loader import AlphaAssayConfig, load_config

VALID = textwrap.dedent("""\
    strategy:
      class: mypkg.my_strategy:MyStrategy
      params:
        tick_window: 10
    risk_caps:
      max_stop_pts: 5.0
      min_target_pts: 2.5
      min_target_to_stop_ratio: 2.0
    session:
      minutes_after_open: 30
      minutes_before_close: 30
    execution:
      mode: paper
      instrument: MES
""")


def test_loads_valid_config(tmp_path):
    p = tmp_path / "valid.yaml"
    p.write_text(VALID)
    cfg = load_config(p)
    assert isinstance(cfg, AlphaAssayConfig)
    assert cfg.risk_caps.max_stop_pts == 5.0
    assert cfg.execution.mode == "paper"


def test_rejects_missing_risk_caps(tmp_path):
    invalid = "risk_caps:\n" "  max_stop_pts: 5.0\n" "  min_target_pts: 2.5\n" "  min_target_to_stop_ratio: 2.0\n"
    assert invalid in VALID  # sanity: the block we strip is actually present
    bad = VALID.replace(invalid, "")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValidationError):
        load_config(p)


def test_rejects_invalid_mode(tmp_path):
    invalid = VALID.replace("mode: paper", "mode: bogus")
    p = tmp_path / "bad_mode.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match="mode"):
        load_config(p)


def test_rejects_negative_max_stop(tmp_path):
    invalid = VALID.replace("max_stop_pts: 5.0", "max_stop_pts: -1.0")
    p = tmp_path / "neg.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError):
        load_config(p)


def test_config_path_strategy_class(tmp_path):
    p = tmp_path / "valid.yaml"
    p.write_text(VALID)
    cfg = load_config(p)
    # Format must be "module:Class"
    assert ":" in cfg.strategy.class_


def test_rejects_numeric_module(tmp_path):
    invalid = VALID.replace(
        "class: mypkg.my_strategy:MyStrategy",
        "class: 1foo:Bar",
    )
    p = tmp_path / "numeric.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match="strategy.class"):
        load_config(p)


def test_rejects_spaces_in_class_path(tmp_path):
    invalid = VALID.replace(
        "class: mypkg.my_strategy:MyStrategy",
        "class: foo.bar:My Class",
    )
    p = tmp_path / "spaces.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match="strategy.class"):
        load_config(p)


# --- Load-time exit-params vs risk-caps cross-check ----------------------
#
# A strategy that declares STATIC exit distances (stop/target points) does
# so under `strategy.params.risk` - the shape static-exit strategies use.
# Those distances must satisfy the same hard
# caps the engine enforces per-signal (alpha_assay.risk.caps.RiskCaps),
# otherwise EVERY signal is silently filtered at runtime and the strategy
# can never trade. We catch that contradiction at load time. The caps in
# this fixture are deliberately compatible with the exits.

VALID_WITH_EXITS = textwrap.dedent("""\
    strategy:
      class: mypkg.my_strategy:MyStrategy
      params:
        signal:
          tick_window: 10
        risk:
          stop_points: 0.5
          target_points: 1.0
    risk_caps:
      max_stop_pts: 5.0
      min_target_pts: 1.0
      min_target_to_stop_ratio: 2.0
    session:
      minutes_after_open: 30
      minutes_before_close: 30
    execution:
      mode: paper
      instrument: ESM6
""")


def test_accepts_static_exit_params_within_caps(tmp_path):
    p = tmp_path / "exits_ok.yaml"
    p.write_text(VALID_WITH_EXITS)
    cfg = load_config(p)
    assert cfg.strategy.params["risk"]["target_points"] == 1.0


def test_rejects_static_target_below_min_target_pts(tmp_path):
    # The unsatisfiable case: exit target 1.0 vs cap min_target 2.5.
    # 1.0 >= 2.5 is unsatisfiable, so the engine filters 100% of signals.
    # Must fail loudly at load, not silently at runtime.
    invalid = VALID_WITH_EXITS.replace("min_target_pts: 1.0", "min_target_pts: 2.5")
    p = tmp_path / "target_below_min.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match="target"):
        load_config(p)


def test_rejects_static_stop_above_max_stop_pts(tmp_path):
    invalid = VALID_WITH_EXITS.replace("stop_points: 0.5", "stop_points: 6.0")
    p = tmp_path / "stop_above_max.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match="stop"):
        load_config(p)


def test_rejects_static_target_to_stop_ratio_below_min(tmp_path):
    # stop 1.0, target 1.5 -> ratio 1.5 < 2.0. target 1.5 >= min_target 1.0
    # and stop 1.0 <= max_stop 5.0, so the ratio is the only failing invariant.
    invalid = VALID_WITH_EXITS.replace("stop_points: 0.5", "stop_points: 1.0").replace(
        "target_points: 1.0", "target_points: 1.5"
    )
    p = tmp_path / "ratio_below.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match="ratio"):
        load_config(p)


def test_skips_exit_cap_check_when_no_static_risk_block(tmp_path):
    # Dynamic-exit strategies declare no static risk block; the cross-check
    # must not fire. VALID has params without a `risk` key and min_target 2.5,
    # and must still load.
    p = tmp_path / "no_risk_block.yaml"
    p.write_text(VALID)
    cfg = load_config(p)
    assert "risk" not in cfg.strategy.params
