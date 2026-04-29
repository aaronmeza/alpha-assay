# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""alpha_assay CLI entry point."""

from __future__ import annotations

import click

from alpha_assay.cli.backfill import backfill
from alpha_assay.cli.backtest import backtest
from alpha_assay.cli.live_check import live_check
from alpha_assay.cli.report import report


@click.group()
def main() -> None:
    """alpha_assay: open-source scalper framework for ES/MES futures."""


main.add_command(backfill)
main.add_command(backtest)
main.add_command(live_check)
main.add_command(report)

__all__ = ["main"]
