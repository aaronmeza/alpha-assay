# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Config schema and YAML loader.

All config errors surface here at load time with field-specific messages.
Uses pydantic v2.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CLASS_PATH_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*$")

# The reserved keys under `strategy.params.risk` that declare a static exit
# pair. Everything else under `strategy.params` stays opaque to the framework.
_STATIC_EXIT_KEYS = ("stop_points", "target_points")


def _finite_exit_distance(field: str, value: Any) -> float:
    """Coerce a declared static exit distance to a real, finite float.

    Anything that is not a usable price distance is rejected here rather than
    left for the caps to compare against, because the caps cannot catch it:
    a bool subclasses int (so `float(True)` would silently become a 1.0-point
    distance), and every comparison against NaN evaluates False (so a NaN
    would satisfy all three cap invariants and reach bracket construction as
    a NaN price). A quoted YAML numeric IS a distance and is coerced, not
    rejected.
    """
    if isinstance(value, bool):
        raise ValueError(f"strategy.params.risk.{field} must be a finite number (got {value!r})")
    try:
        distance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strategy.params.risk.{field} must be a finite number (got {value!r})") from exc
    if not math.isfinite(distance):
        raise ValueError(f"strategy.params.risk.{field} must be a finite number (got {value!r})")
    return distance


class StrategySection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_: str = Field(alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_")
    @classmethod
    def validate_class_path(cls, v: str) -> str:
        if not _CLASS_PATH_RE.match(v):
            raise ValueError(f"strategy.class must match 'package.module:ClassName' (got: {v!r})")
        return v


class RiskCapsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_stop_pts: float = Field(gt=0)
    min_target_pts: float = Field(gt=0)
    min_target_to_stop_ratio: float = Field(gt=0)


class SessionSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minutes_after_open: int = Field(ge=0)
    minutes_before_close: int = Field(ge=0)


class ExecutionSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["backtest", "paper", "live"]
    instrument: str
    # Risk-based position sizing. Engine computes contracts from
    # account_balance * risk_per_trade_pct / stop_dollar, then clamps
    # to [1, max_contracts]. If risk_per_trade_pct is None or 0, falls
    # back to a single contract per signal (legacy behavior).
    risk_per_trade_pct: float | None = Field(default=None, ge=0, le=1)
    max_contracts: int = Field(default=1, ge=1)


class AlphaAssayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: StrategySection
    risk_caps: RiskCapsSection
    session: SessionSection
    execution: ExecutionSection

    @model_validator(mode="after")
    def _static_exits_within_risk_caps(self) -> AlphaAssayConfig:
        """Enforce the hard risk caps against complete static exit distances.

        A strategy with fixed stop/target points declares them under
        ``strategy.params.risk`` (the shape static-exit strategies use).
        Those distances must satisfy the same three
        invariants ``RiskCaps`` enforces per-signal in the engine; otherwise
        an unsatisfiable pair (e.g. ``target_points: 1.0`` against
        ``min_target_pts: 2.5``) silently filters 100% of signals at runtime
        and the strategy can never trade. Catch the contradiction here, where
        it is a loud config error, not a silent zero-trade run.

        Strategies with dynamic (e.g. ATR-based) exits declare no static risk
        block, and a ``risk`` block that does not declare BOTH distances is
        left alone - ``strategy.params`` stays opaque to the framework apart
        from this one complete, reserved key path, so a third-party strategy
        keeping unrelated config under ``risk`` still loads.

        A config that passes this check has usable distances: the coerced
        floats are written back, so what the strategy later reads out of
        ``params.risk`` is the same finite number the caps approved.
        """
        # A `risk` block that is not a dict, or that does not declare BOTH
        # distances as non-null, is not a static-exit declaration. Leave it
        # opaque rather than guess: `strategy.params` belongs to the strategy,
        # and a third-party plugin may legitimately keep unrelated config here.
        risk = self.strategy.params.get("risk")
        if not isinstance(risk, dict):
            return self
        declared = [key for key in _STATIC_EXIT_KEYS if risk.get(key) is not None]
        if len(declared) < len(_STATIC_EXIT_KEYS):
            # Half a pair cannot be cap-checked (the ratio invariant needs
            # both), so the load-time check is skipped - but say so rather than
            # skip in silence, because a typo (`target_pts` for
            # `target_points`) otherwise looks exactly like a validated config.
            # This warns instead of raising: the caps are not bypassed, since
            # the engine validates whatever get_exit_params() returns on EVERY
            # signal and drops the trade on violation, so the worst case is a
            # zero-trade run, never an out-of-cap order. Raising here would
            # instead reject a third-party strategy that uses one of these key
            # names for something else entirely.
            if declared:
                missing = [key for key in _STATIC_EXIT_KEYS if key not in declared]
                warnings.warn(
                    f"strategy.params.risk declares {declared[0]} without {missing[0]}; "
                    "the load-time risk-cap check needs both and was skipped. Declare both "
                    "to have static exit distances validated against risk_caps at load time.",
                    UserWarning,
                    stacklevel=2,
                )
            return self

        # Coerce each declared distance to a real, finite float before the caps
        # see it. A quoted YAML numeric ("0.5") is a distance and must be
        # checked; a bool is not (it subclasses int, so float(True) would pass
        # as a 1.0-point distance); NaN and inf are not (every comparison
        # against NaN is False, so they would satisfy all three cap invariants
        # and reach bracket construction as a NaN price).
        stop_pts = _finite_exit_distance("stop_points", risk["stop_points"])
        target_pts = _finite_exit_distance("target_points", risk["target_points"])

        # Local import keeps the caps module out of the loader's import graph
        # at module-load time (avoids any cycle); the invariant lives in one
        # place - RiskCaps - and is reused verbatim.
        from alpha_assay.risk.caps import RiskCaps

        RiskCaps(
            max_stop_pts=self.risk_caps.max_stop_pts,
            min_target_pts=self.risk_caps.min_target_pts,
            min_target_to_stop_ratio=self.risk_caps.min_target_to_stop_ratio,
        ).validate(stop_pts=stop_pts, target_pts=target_pts)

        # Hand the strategy the value the caps actually approved. Without this
        # the loader would certify a config it has not normalized: a strategy
        # that passes params.risk straight into ExitParams would carry a string
        # into the per-signal cap check and the bracket arithmetic, where it
        # fails long after the config was pronounced valid.
        risk["stop_points"] = stop_pts
        risk["target_points"] = target_pts
        return self


def load_config(path: str | Path) -> AlphaAssayConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AlphaAssayConfig.model_validate(raw)
