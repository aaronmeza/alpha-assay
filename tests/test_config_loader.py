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
