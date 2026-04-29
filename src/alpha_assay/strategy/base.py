# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Public strategy contract.

Subclass `BaseStrategy` to plug your signal logic into the alpha_assay engine.
The framework owns data, execution, risk enforcement, and reporting; the
strategy owns only signal generation and per-signal exit parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Signal:
    """A trading signal emitted by a strategy at a single bar.

    `direction` follows the convention: +1 = long, -1 = short, 0 = flat.
    """

    timestamp: pd.Timestamp
    direction: int


@dataclass(frozen=True, slots=True)
class ExitParams:
    """Per-signal exit distances (and optional time-stop) the engine applies
    when constructing the bracket at fill time.

    Distances are in price points relative to the fill price; the engine
    handles direction (long vs short) when placing the bracket. `time_stop`,
    when set, bounds how long a position may remain open before the engine
    flattens at market. `None` means "no time-stop; stop/target only".
    """

    stop_points: float
    target_points: float
    time_stop: timedelta | None = None


class BaseStrategy(ABC):
    """Base class for all alpha_assay strategies.

    Lifecycle:
      1. `__init__(config)` stores config and calls `_validate_config()`.
      2. The engine calls `generate_signals(data)` once per backtest run, or
         per bar in live mode, to obtain a `{-1, 0, +1}` series.
      3. For each non-zero entry, the engine calls `get_exit_params(signal, data)`
         to determine stop, target, and optional time-stop at fill time.

    Subclasses MUST NOT place orders, mutate broker state, or read from the
    broker directly. All execution flows through the engine's `exec/` layer so
    that paper-to-live is a config change, not a code change.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Override to enforce required config keys. Default: no-op.

        Raise `ValueError` with a clear message when a required key is missing
        or a value is out of range. The engine catches and surfaces this before
        any data is loaded.
        """
        return None

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a series of `{-1, 0, +1}` aligned to `data.index`.

        `data` is a DataFrame indexed by timestamp (timezone-aware,
        America/Chicago) with at minimum OHLCV columns for the primary symbol.
        Additional columns (TICK, ADD, etc.) are present when the run config
        requests them.

        Implementations MUST NOT use any value from a future bar. Lookahead bias
        is a release blocker; the engine asserts strict causality in tests.
        """
        ...

    @abstractmethod
    def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> ExitParams:
        """Return `ExitParams(stop_points, target_points, time_stop)` for a
        non-zero signal.

        `data` is sliced up to and including `signal.timestamp` (no future
        bars). Distances are in price points relative to the fill price.
        `time_stop=None` disables the time-stop for this signal.
        """
        ...
