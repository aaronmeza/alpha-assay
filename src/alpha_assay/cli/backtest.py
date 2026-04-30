# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""`alpha_assay backtest` command.

Loads a YAML config via config/loader, loads parquet via the Databento
adapter (canonical schema), instantiates the strategy by dotted-path
reference, runs the engine, and writes trades.csv + session_metrics.json
to --out.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import click
import pandas as pd

from alpha_assay.config.loader import load_config
from alpha_assay.data.databento_adapter import load_parquet
from alpha_assay.data.joined_loader import load_es_with_breadth
from alpha_assay.engine.nautilus_runner import NautilusBacktestRunner
from alpha_assay.risk.caps import RiskCaps


def _load_strategy_class(path: str):
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise click.BadParameter(f"--strategy must be 'module:Class', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@click.command()
@click.option("--strategy", required=True, help="Strategy reference 'module:Class'.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML config path.",
)
@click.option(
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet data path (Databento canonical schema).",
)
@click.option(
    "--tick-data",
    "tick_data_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional parquet of per-minute TICK-NYSE breadth bars.",
)
@click.option(
    "--ad-data",
    "ad_data_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional parquet of per-minute AD-NYSE breadth bars.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory (created if absent).",
)
def backtest(
    strategy: str,
    config_path: str,
    data_path: str,
    tick_data_path: str | None,
    ad_data_path: str | None,
    out_dir: str,
) -> None:
    """Run a backtest and write trades.csv + session_metrics.json."""
    # Breadth flags come as a pair: breadth-aware
    # strategies read both `data["TICK"]` and `data["ADD"]`. Failing
    # loudly here beats silently dropping one column and NaN-ing the
    # z-score path inside the strategy.
    if (tick_data_path is None) ^ (ad_data_path is None):
        raise click.UsageError(
            "--tick-data and --ad-data must be provided together " "(breadth-aware strategies need both feeds)."
        )

    cfg = load_config(config_path)
    strategy_cls = _load_strategy_class(strategy)
    strategy_instance = strategy_cls(config=cfg.strategy.params)

    risk_caps = RiskCaps(
        max_stop_pts=cfg.risk_caps.max_stop_pts,
        min_target_pts=cfg.risk_caps.min_target_pts,
        min_target_to_stop_ratio=cfg.risk_caps.min_target_to_stop_ratio,
    )

    if tick_data_path is not None and ad_data_path is not None:
        df = load_es_with_breadth(data_path, tick_path=tick_data_path, ad_path=ad_data_path).reset_index()
    else:
        df = load_parquet(data_path).reset_index()
    # Engine expects a `timestamp` column (not index) + ES_-prefixed OHLCV
    # or lowercase; bar_adapter auto-detects lowercase here.
    runner = NautilusBacktestRunner(
        strategy=strategy_instance,
        data=df,
        instrument_symbol=cfg.execution.instrument,
        starting_balance_usd=100_000.0,
        risk_caps=risk_caps,
        risk_per_trade_pct=cfg.execution.risk_per_trade_pct,
        max_contracts=cfg.execution.max_contracts,
    )
    result = runner.run()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(result.trades)
    trades_df.to_csv(out / "trades.csv", index=False)
    (out / "session_metrics.json").write_text(json.dumps(result.session_metrics, default=str, indent=2))
    click.echo(f"Wrote {len(result.trades)} trades and session_metrics.json to {out}")
