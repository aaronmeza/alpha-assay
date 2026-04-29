# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Regression tests for `alpha_assay report` total/annualized return.

The `total_return_pct` field in the enriched session_metrics.json must
reconcile with the dollar-balance change reflected in
`final_balance_usd`. Quantstats' compound-return number was previously
leaking into this field and producing wrong-sign / wrong-magnitude
values on real backtest runs (see PR fix/report-return-pct).

These tests pin down:

- Direct reconciliation: `total_return_pct == (final - start) / start *
  100` within float tolerance.
- The headline negative-return case from the SMA crossover backtest on
  30 days of real ES bars: final $99,903.75 on $100,000 start =>
  -0.09625% (NOT +1.099%).
- The annualized return matches the elapsed-time formula and is
  negative when the total return is negative.
- The CLI `--starting-balance` flag wins when no session_metrics
  override is present, and a `starting_balance_usd` field embedded in
  session_metrics.json takes precedence over the flag.
- A `compound_return_pct` field is preserved separately so the
  quantstats compound figure is still available without confusing the
  dollar-basis reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from alpha_assay.cli import main as cli_main
from alpha_assay.cli.report import (
    _annualized_from_total_and_days,
    _elapsed_days_from_per_trade,
    _reconciled_total_return_pct,
)

pytest.importorskip("quantstats")


def _write_mock_run_dir(
    tmp_path: Path,
    *,
    final_balance_usd: float,
    starting_balance_usd: float | None = None,
    trades: list[dict] | None = None,
) -> Path:
    """Write a mock backtest output dir with trades.csv + session_metrics.json.

    The trades list mirrors the Nautilus runner's order emission shape:
    every trade is two rows (market entry + bracket exit). The default
    builds 30 trades whose USD PnL on a multiplier of 50 sums to
    ``final_balance_usd - 100_000`` so the report's per-trade frame and
    the runner's reported final balance agree on the dollar-basis change.
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    if trades is None:
        # Default: 30 trades, one per calendar day, evenly spaced. We
        # tune the per-trade points so the cumulative USD PnL equals
        # final_balance_usd - 100_000 with multiplier 50.
        target_pnl_usd = final_balance_usd - 100_000.0
        n = 30
        per_trade_pnl_usd = target_pnl_usd / n
        per_trade_points = per_trade_pnl_usd / 50.0
        rows: list[dict] = []
        base_ts = pd.Timestamp("2026-03-01 14:30:00", tz="UTC")
        for i in range(n):
            entry_ts = base_ts + pd.Timedelta(days=i)
            exit_ts = entry_ts + pd.Timedelta(minutes=30)
            entry_price = 5000.0
            exit_price = entry_price + per_trade_points
            rows.append(
                {
                    "timestamp": entry_ts.isoformat(),
                    "side": "buy",
                    "price": entry_price,
                    "quantity": 1.0,
                    "order_type": "market",
                }
            )
            rows.append(
                {
                    "timestamp": exit_ts.isoformat(),
                    "side": "sell",
                    "price": exit_price,
                    "quantity": 1.0,
                    "order_type": "limit",
                }
            )
        trades = rows

    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)

    metrics: dict = {
        "run_status": "completed",
        "submitted_signals": len([r for r in trades if r["order_type"] == "market"]),
        "orders_submitted": len(trades),
        "signals_filtered_risk_cap": 0,
        "final_balance_usd": final_balance_usd,
    }
    if starting_balance_usd is not None:
        metrics["starting_balance_usd"] = starting_balance_usd
    (out_dir / "session_metrics.json").write_text(json.dumps(metrics))
    return out_dir


def _run_report(
    tmp_path: Path,
    in_dir: Path,
    *,
    starting_balance: float | None = None,
    multiplier: float = 50.0,
) -> dict:
    """Invoke the report CLI and return the enriched session_metrics.json."""
    out = tmp_path / "report_out"
    runner = CliRunner()
    args = [
        "report",
        "--in",
        str(in_dir),
        "--out",
        str(out),
        "--format",
        "md",
        "--instrument-multiplier",
        str(multiplier),
    ]
    if starting_balance is not None:
        args += ["--starting-balance", str(starting_balance)]
    result = runner.invoke(cli_main, args)
    assert result.exit_code == 0, result.output
    return json.loads((out / "session_metrics.json").read_text())


def test_reconciled_total_return_pct_matches_dollar_basis():
    # 99903.75 on 100000 => -0.09625% (the real-data SMA backtest case).
    pct = _reconciled_total_return_pct(100_000.0, 99_903.75)
    assert pct is not None
    assert pct == pytest.approx(-0.09625, abs=1e-6)


def test_reconciled_total_return_pct_zero_start_returns_none():
    assert _reconciled_total_return_pct(0.0, 100.0) is None


def test_reconciled_total_return_pct_positive_case():
    pct = _reconciled_total_return_pct(100_000.0, 105_000.0)
    assert pct == pytest.approx(5.0, abs=1e-9)


def test_annualized_from_total_and_days_negative_preserved():
    # 30-day -0.09625% return should annualize to a small negative number.
    ann = _annualized_from_total_and_days(-0.09625, 30.0)
    assert ann is not None
    assert ann < 0
    # Compounded: (1 - 0.0009625)^(365/30) - 1 ~= -1.16%
    assert ann == pytest.approx(-1.1607, abs=5e-3)


def test_annualized_from_total_and_days_short_window_returns_none():
    assert _annualized_from_total_and_days(1.0, 0.5) is None
    assert _annualized_from_total_and_days(None, 30.0) is None
    assert _annualized_from_total_and_days(1.0, None) is None


def test_annualized_from_total_and_days_total_loss_floors_at_minus_100():
    ann = _annualized_from_total_and_days(-100.0, 30.0)
    assert ann == pytest.approx(-100.0)


def test_elapsed_days_from_per_trade_simple_window():
    df = pd.DataFrame(
        {
            "entry_ts": [
                pd.Timestamp("2026-03-01 14:30:00", tz="UTC"),
                pd.Timestamp("2026-03-15 14:30:00", tz="UTC"),
            ],
            "exit_ts": [
                pd.Timestamp("2026-03-01 15:00:00", tz="UTC"),
                pd.Timestamp("2026-03-31 15:00:00", tz="UTC"),
            ],
        }
    )
    days = _elapsed_days_from_per_trade(df)
    assert days is not None
    assert days == pytest.approx(30 + 30 / (24 * 60), rel=1e-6)


def test_elapsed_days_empty_returns_none():
    assert _elapsed_days_from_per_trade(pd.DataFrame()) is None


def test_cli_total_return_pct_matches_final_balance(tmp_path):
    # The headline real-data case: final $99,903.75 on $100k start.
    final = 99_903.75
    in_dir = _write_mock_run_dir(tmp_path, final_balance_usd=final)
    enriched = _run_report(tmp_path, in_dir)

    expected_pct = (final - 100_000.0) / 100_000.0 * 100.0
    assert enriched["total_return_pct"] == pytest.approx(expected_pct, abs=1e-9)
    # Direction: must be negative on a losing run.
    assert enriched["total_return_pct"] < 0
    # Reconciliation: same number derivable from the JSON itself.
    final_json = float(enriched["final_balance_usd"])
    start_json = float(enriched["starting_balance_usd"])
    assert abs(enriched["total_return_pct"] - (final_json - start_json) / start_json * 100.0) < 1e-3


def test_cli_total_return_pct_positive_case(tmp_path):
    in_dir = _write_mock_run_dir(tmp_path, final_balance_usd=105_000.0)
    enriched = _run_report(tmp_path, in_dir)
    assert enriched["total_return_pct"] == pytest.approx(5.0, abs=1e-9)


def test_cli_annualized_return_pct_matches_elapsed_time_formula(tmp_path):
    # 30 trades evenly spaced one day apart => 29-30 day span. The
    # annualized return should match the formula applied to that span.
    final = 99_903.75
    in_dir = _write_mock_run_dir(tmp_path, final_balance_usd=final)
    enriched = _run_report(tmp_path, in_dir)

    pt = pd.read_csv(tmp_path / "report_out" / "per_trade_metrics.csv")
    pt["entry_ts"] = pd.to_datetime(pt["entry_ts"], utc=True)
    pt["exit_ts"] = pd.to_datetime(pt["exit_ts"], utc=True)
    days = (pt["exit_ts"].max() - pt["entry_ts"].min()).total_seconds() / 86400.0

    expected = ((1 + enriched["total_return_pct"] / 100.0) ** (365 / days) - 1) * 100.0
    assert enriched["annualized_return_pct"] is not None
    assert enriched["annualized_return_pct"] == pytest.approx(expected, abs=1e-6)
    # And on a losing run, the annualized number must be negative.
    assert enriched["annualized_return_pct"] < 0


def test_cli_compound_return_pct_emitted_separately(tmp_path):
    # The quantstats compound figure is preserved under its own key
    # so callers that want it can still find it - but it is no longer
    # the value reported as `total_return_pct`.
    in_dir = _write_mock_run_dir(tmp_path, final_balance_usd=99_903.75)
    enriched = _run_report(tmp_path, in_dir)
    assert "compound_return_pct" in enriched


def test_cli_starting_balance_flag_used_when_metrics_omits_it(tmp_path):
    # Flag set, no starting_balance_usd in session_metrics.json: flag wins.
    in_dir = _write_mock_run_dir(tmp_path, final_balance_usd=49_500.0)
    enriched = _run_report(tmp_path, in_dir, starting_balance=50_000.0)
    assert enriched["starting_balance_usd"] == pytest.approx(50_000.0)
    assert enriched["total_return_pct"] == pytest.approx(-1.0, abs=1e-6)


def test_cli_starting_balance_in_metrics_overrides_flag(tmp_path):
    # session_metrics.json carries its own starting_balance_usd, which
    # must win over the flag (a future runner is the authoritative source).
    in_dir = _write_mock_run_dir(
        tmp_path,
        final_balance_usd=24_750.0,
        starting_balance_usd=25_000.0,
    )
    # Pass a misleading flag value to confirm it's ignored.
    enriched = _run_report(tmp_path, in_dir, starting_balance=999_999.0)
    assert enriched["starting_balance_usd"] == pytest.approx(25_000.0)
    assert enriched["total_return_pct"] == pytest.approx(-1.0, abs=1e-6)


def test_cli_reconciliation_holds_under_any_final_balance(tmp_path):
    # Pin the contract: total_return_pct is always (final - start) / start
    # within 1e-3, regardless of how trades are distributed.
    for final in (95_000.0, 99_000.0, 99_903.75, 100_000.0, 100_500.0, 110_000.0):
        sub_tmp = tmp_path / f"run_{int(final)}"
        sub_tmp.mkdir()
        in_dir = _write_mock_run_dir(sub_tmp, final_balance_usd=final)
        enriched = _run_report(sub_tmp, in_dir)
        expected = (final - 100_000.0) / 100_000.0 * 100.0
        assert (
            abs(enriched["total_return_pct"] - expected) < 1e-3
        ), f"final={final}: got {enriched['total_return_pct']}, expected {expected}"
