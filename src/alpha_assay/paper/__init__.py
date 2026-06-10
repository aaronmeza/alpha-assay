# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Pluggable paper-trading runner.

`alpha_assay.paper` turns the always-flat paper dry-run into a generic
strategy host: an env-selected `BaseStrategy` subclass receives the
canonical joined minute frame (built incrementally from bus messages)
and its signals are executed as paper bracket orders through the exec
layer. See `alpha_assay.paper.runner` for the entrypoint surface.
"""

from alpha_assay.paper.frame import MinuteCloseAggregator, SessionFrameBuilder
from alpha_assay.paper.runner import PaperStrategyRunner, load_paper_strategy

__all__ = [
    "MinuteCloseAggregator",
    "PaperStrategyRunner",
    "SessionFrameBuilder",
    "load_paper_strategy",
]
