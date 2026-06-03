# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Config schema and YAML loader.

All config errors surface here at load time with field-specific messages.
Uses pydantic v2.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CLASS_PATH_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*$")


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
        """Enforce the hard risk caps against any STATIC exit distances at
        load time.

        A strategy with fixed stop/target points declares them under
        ``strategy.params.risk`` (the shape static-exit strategies use).
        Those distances must satisfy the same three
        invariants ``RiskCaps`` enforces per-signal in the engine; otherwise
        an unsatisfiable pair (e.g. ``target_points: 1.0`` against
        ``min_target_pts: 2.5``) silently filters 100% of signals at runtime
        and the strategy can never trade. Catch the contradiction here, where
        it is a loud config error, not a silent zero-trade run.

        Strategies with dynamic (e.g. ATR-based) exits declare no static risk
        block, or non-numeric values; those are skipped - there is nothing to
        check until runtime.
        """
        risk = self.strategy.params.get("risk")
        if not isinstance(risk, dict):
            return self
        stop = risk.get("stop_points")
        target = risk.get("target_points")
        if not isinstance(stop, (int, float)) or not isinstance(target, (int, float)):
            return self
        # Local import keeps the caps module out of the loader's import graph
        # at module-load time (avoids any cycle); the invariant lives in one
        # place - RiskCaps - and is reused verbatim.
        from alpha_assay.risk.caps import RiskCaps

        RiskCaps(
            max_stop_pts=self.risk_caps.max_stop_pts,
            min_target_pts=self.risk_caps.min_target_pts,
            min_target_to_stop_ratio=self.risk_caps.min_target_to_stop_ratio,
        ).validate(stop_pts=float(stop), target_pts=float(target))
        return self


def load_config(path: str | Path) -> AlphaAssayConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AlphaAssayConfig.model_validate(raw)
