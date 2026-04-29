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
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class AlphaAssayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: StrategySection
    risk_caps: RiskCapsSection
    session: SessionSection
    execution: ExecutionSection


def load_config(path: str | Path) -> AlphaAssayConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AlphaAssayConfig.model_validate(raw)
