# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""PositionSizer unit tests.

Sizing math is the load-bearing piece of risk-per-trade scaling. The
engine integration is covered by tests/test_engine_*.py and the CLI
plumbing by tests/test_cli_backtest.py; here we just exercise the
formula and edge cases in isolation.
"""

from __future__ import annotations

import pytest

from alpha_assay.engine.nautilus_runner import PositionSizer


def test_default_falls_back_to_one_contract():
    sizer = PositionSizer(account_balance=100_000, instrument_multiplier=50.0)
    assert sizer.compute_contracts(stop_points=0.5) == 1


def test_zero_pct_falls_back_to_one_contract():
    sizer = PositionSizer(
        account_balance=100_000,
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.0,
        max_contracts=20,
    )
    assert sizer.compute_contracts(stop_points=0.5) == 1


def test_basic_risk_based_sizing_es():
    # $100K * 0.5% = $500 risk budget. 0.5pt stop * $50/pt = $25/contract.
    # $500 // $25 = 20 contracts.
    sizer = PositionSizer(
        account_balance=100_000,
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.005,
        max_contracts=100,
    )
    assert sizer.compute_contracts(stop_points=0.5) == 20


def test_basic_risk_based_sizing_mes():
    # MES is 1/10 of ES: $5/pt. $500 / ($5 * 0.5) = 200 contracts.
    sizer = PositionSizer(
        account_balance=100_000,
        instrument_multiplier=5.0,
        risk_per_trade_pct=0.005,
        max_contracts=500,
    )
    assert sizer.compute_contracts(stop_points=0.5) == 200


def test_max_contracts_cap_applied():
    sizer = PositionSizer(
        account_balance=100_000,
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.005,
        max_contracts=5,
    )
    # Would compute 20 but capped at 5.
    assert sizer.compute_contracts(stop_points=0.5) == 5


def test_floor_one_contract_minimum():
    # Tiny risk budget that would round down to 0; floor enforces 1.
    sizer = PositionSizer(
        account_balance=100,  # $100 account
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.001,  # 0.1% = $0.10 risk budget
        max_contracts=20,
    )
    # $0.10 / $25 = 0 contracts naturally; sizer returns 1.
    assert sizer.compute_contracts(stop_points=0.5) == 1


def test_zero_stop_points_safe():
    """Zero stop would div-by-zero; fall back to 1 contract."""
    sizer = PositionSizer(
        account_balance=100_000,
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.005,
        max_contracts=20,
    )
    assert sizer.compute_contracts(stop_points=0.0) == 1


def test_negative_stop_safe():
    sizer = PositionSizer(
        account_balance=100_000,
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.005,
        max_contracts=20,
    )
    assert sizer.compute_contracts(stop_points=-1.0) == 1


def test_wider_stop_smaller_position():
    """Doubling the stop halves the contract count (risk budget invariant)."""
    sizer = PositionSizer(
        account_balance=100_000,
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.005,
        max_contracts=100,
    )
    n_at_05 = sizer.compute_contracts(stop_points=0.5)  # 20
    n_at_10 = sizer.compute_contracts(stop_points=1.0)  # 10
    n_at_25 = sizer.compute_contracts(stop_points=2.5)  # 4
    assert (n_at_05, n_at_10, n_at_25) == (20, 10, 4)


def test_large_account_clamped_by_max_contracts():
    """$10M account at 1% risk would be huge; max_contracts must keep it sane."""
    sizer = PositionSizer(
        account_balance=10_000_000,
        instrument_multiplier=50.0,
        risk_per_trade_pct=0.01,
        max_contracts=50,
    )
    # Without cap: $100K / $25 = 4000 contracts. With cap=50, clamped.
    assert sizer.compute_contracts(stop_points=0.5) == 50


@pytest.mark.parametrize(
    "balance,pct,stop,mult,cap,expected",
    [
        (100_000, 0.0050, 0.50, 50, 100, 20),  # baseline
        (100_000, 0.0010, 0.50, 50, 100, 4),  # quarter risk -> quarter size
        (50_000, 0.0050, 0.50, 50, 100, 10),  # half balance -> half size
        (100_000, 0.0050, 1.00, 50, 100, 10),  # double stop -> half size
        (100_000, 0.0050, 0.25, 50, 100, 40),  # half stop -> double size
    ],
)
def test_sizing_table(balance, pct, stop, mult, cap, expected):
    sizer = PositionSizer(
        account_balance=balance,
        instrument_multiplier=mult,
        risk_per_trade_pct=pct,
        max_contracts=cap,
    )
    assert sizer.compute_contracts(stop_points=stop) == expected
