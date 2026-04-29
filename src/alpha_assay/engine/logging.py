# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Canonical LoggingConfig for the NautilusTrader adapter.

Nautilus' Rust logger is a process-lifetime singleton (see ADR Appendix A).
Multiple BacktestEngine instances in the same process trigger SIGABRT
inside `nautilus_pyo3.init_logging` unless `bypass_logging=True` is set.
The test suite necessarily creates multiple engines per process; every
engine build in this repo therefore uses this LOGGING_CONFIG.

`reset_for_test()` exists as a future hook in case a later Nautilus
version adds a proper teardown; today it is a no-op documenting intent.
"""

from __future__ import annotations

from nautilus_trader.common.config import LoggingConfig

LOGGING_CONFIG: LoggingConfig = LoggingConfig(bypass_logging=True)


def reset_for_test() -> None:
    """No-op hook reserved for future Nautilus versions that expose a
    Rust-logger teardown. Safe to call from pytest fixtures.
    """
    return None
