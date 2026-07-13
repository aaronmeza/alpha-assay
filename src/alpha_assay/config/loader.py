# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Config schema and YAML loader.

All config errors surface here at load time with field-specific messages.
Uses pydantic v2.
"""

from __future__ import annotations

import math
import re
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
        block, and a ``risk`` block holding unrelated keys is left alone -
        ``strategy.params`` stays opaque to the framework apart from this one
        reserved key path.
        """
        # Leave non-dict or unrelated strategy risk parameters opaque so
        # third-party strategies can keep plugin-specific configuration here.
        risk = self.strategy.params.get("risk")
        if not isinstance(risk, dict):
            return self

        # A key counts as declared only when present and non-null, so both an
        # absent key and an explicit YAML null mean "no static exit here" and
        # leave dynamic-exit strategies untouched.
        declared = {key: risk[key] for key in _STATIC_EXIT_KEYS if risk.get(key) is not None}
        if not declared:
            return self

        # Half a pair is a typo (`target_pts` for `target_points`), not a
        # contract: it cannot be cap-checked, and today it surfaces as a
        # KeyError mid-session instead of a config error before the run.
        if len(declared) == 1:
            missing = next(key for key in _STATIC_EXIT_KEYS if key not in declared)
            present = next(iter(declared))
            raise ValueError(
                f"strategy.params.risk.{missing} must be declared alongside strategy.params.risk.{present}"
            )

        # Coerce each declared distance to a real, finite float before the caps
        # see it. A quoted YAML numeric ("0.5") is a distance and must be
        # checked; a bool is not (it subclasses int, so float(True) would pass
        # as a 1.0-point distance); NaN and inf are not (every comparison
        # against NaN is False, so they would satisfy all three cap invariants
        # and reach bracket construction as a NaN price).
        stop_pts = _finite_exit_distance("stop_points", declared["stop_points"])
        target_pts = _finite_exit_distance("target_points", declared["target_points"])

        # Local import keeps the caps module out of the loader's import graph
        # at module-load time (avoids any cycle); the invariant lives in one
        # place - RiskCaps - and is reused verbatim.
        from alpha_assay.risk.caps import RiskCaps

        RiskCaps(
            max_stop_pts=self.risk_caps.max_stop_pts,
            min_target_pts=self.risk_caps.min_target_pts,
            min_target_to_stop_ratio=self.risk_caps.min_target_to_stop_ratio,
        ).validate(stop_pts=stop_pts, target_pts=target_pts)
        return self


def load_config(path: str | Path) -> AlphaAssayConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AlphaAssayConfig.model_validate(raw)
