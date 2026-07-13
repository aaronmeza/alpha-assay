import textwrap
import warnings

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


def test_rejects_quoted_static_stop_above_max_stop_pts(tmp_path):
    invalid = VALID_WITH_EXITS.replace("stop_points: 0.5", 'stop_points: "50.0"')
    p = tmp_path / "quoted_stop_above_max.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match="stop"):
        load_config(p)


def test_accepts_quoted_static_exit_params_within_caps(tmp_path):
    # A quoted numeric IS a distance, so it is coerced and cap-checked rather
    # than skipped. The coerced floats are written back, so a strategy reading
    # params.risk gets the same number the caps approved - never a string that
    # would blow up later in the per-signal check or the bracket arithmetic.
    valid = VALID_WITH_EXITS.replace("stop_points: 0.5", 'stop_points: "0.5"').replace(
        "target_points: 1.0", 'target_points: "1.0"'
    )
    p = tmp_path / "quoted_exits_ok.yaml"
    p.write_text(valid)
    cfg = load_config(p)
    risk = cfg.strategy.params["risk"]
    assert risk["stop_points"] == 0.5
    assert risk["target_points"] == 1.0
    assert isinstance(risk["stop_points"], float)
    assert isinstance(risk["target_points"], float)


def test_rejects_non_numeric_static_stop(tmp_path):
    invalid = VALID_WITH_EXITS.replace("stop_points: 0.5", 'stop_points: "abc"')
    p = tmp_path / "non_numeric_stop.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match=r"strategy\.params\.risk\.stop_points"):
        load_config(p)


def test_rejects_bool_static_stop_even_when_caps_would_pass(tmp_path):
    invalid = VALID_WITH_EXITS.replace("stop_points: 0.5", "stop_points: true").replace(
        "target_points: 1.0", "target_points: 5.0"
    )
    p = tmp_path / "bool_stop.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match=r"strategy\.params\.risk\.stop_points"):
        load_config(p)


def test_rejects_nan_static_stop(tmp_path):
    invalid = VALID_WITH_EXITS.replace("stop_points: 0.5", "stop_points: .nan")
    p = tmp_path / "nan_stop.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match=r"strategy\.params\.risk\.stop_points"):
        load_config(p)


def test_rejects_nan_static_target(tmp_path):
    invalid = VALID_WITH_EXITS.replace("target_points: 1.0", "target_points: .nan")
    p = tmp_path / "nan_target.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match=r"strategy\.params\.risk\.target_points"):
        load_config(p)


def test_rejects_inf_static_target(tmp_path):
    invalid = VALID_WITH_EXITS.replace("target_points: 1.0", "target_points: .inf")
    p = tmp_path / "inf_target.yaml"
    p.write_text(invalid)
    with pytest.raises(ValidationError, match=r"strategy\.params\.risk\.target_points"):
        load_config(p)


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


def test_skips_exit_cap_check_when_risk_block_has_only_unrelated_keys(tmp_path):
    # A `risk` block naming NEITHER reserved key is ordinary opaque plugin
    # config. It must load SILENTLY - warning here would nag every third-party
    # strategy that happens to have a `risk` block.
    valid = VALID_WITH_EXITS.replace(
        "    risk:\n      stop_points: 0.5\n      target_points: 1.0\n",
        "    risk:\n      position_size_multiplier: 2\n",
    )
    p = tmp_path / "unrelated_risk_keys.yaml"
    p.write_text(valid)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        cfg = load_config(p)
    assert cfg.strategy.params["risk"]["position_size_multiplier"] == 2


def test_warns_but_loads_when_static_exit_block_is_partial(tmp_path):
    # Only a COMPLETE pair is a static-exit declaration: the ratio invariant
    # needs both, and treating a lone key as reserved would reject a
    # third-party strategy keeping unrelated config under `risk`, breaking the
    # promise that strategy.params is opaque. So this loads - but it WARNS,
    # because a typo (`target_pts`) would otherwise look like a validated
    # config. The caps are not bypassed: the engine validates whatever
    # get_exit_params() returns on every signal and drops the trade on
    # violation, so the over-cap stop here can only cause a zero-trade run,
    # never a live out-of-cap order.
    partial = VALID_WITH_EXITS.replace("      target_points: 1.0\n", "").replace(
        "stop_points: 0.5", "stop_points: 50.0"
    )
    p = tmp_path / "partial_static_exit.yaml"
    p.write_text(partial)
    with pytest.warns(UserWarning, match="target_points"):
        cfg = load_config(p)
    risk = cfg.strategy.params["risk"]
    assert risk["stop_points"] == 50.0
    assert "target_points" not in risk


def test_skips_exit_cap_check_when_static_exit_values_are_null(tmp_path):
    # An explicit YAML null means "not declared", same as an absent key, so a
    # dynamic-exit strategy can carry the keys without inviting a cap check.
    nulled = VALID_WITH_EXITS.replace("stop_points: 0.5", "stop_points: null").replace(
        "target_points: 1.0", "target_points: null"
    )
    p = tmp_path / "null_static_exit.yaml"
    p.write_text(nulled)
    cfg = load_config(p)
    assert cfg.strategy.params["risk"]["stop_points"] is None
