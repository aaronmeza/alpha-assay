# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Unit tests for `alpha_assay.cli.report.pair_trades`.

The pairing logic is the most fragile piece of the report pipeline:
get this wrong and every downstream metric is wrong. These tests cover
every order-shape we expect from the Nautilus runner emission:
long round-trips, short round-trips, target/stop classification, the
defensive flip case, and degenerate edge cases (empty input,
unmatched entries).
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_assay.cli.report import pair_trades


def _orders(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_pair_trades_long_target_hit():
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "side": "sell",
            "price": 102.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 1
    t = paired[0]
    assert t.side == "long"
    assert t.entry_price == 100.0
    assert t.exit_price == 102.0
    assert t.exit_reason == "target"


def test_pair_trades_long_stop_hit():
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:31:00+00:00",
            "side": "sell",
            "price": 99.0,
            "quantity": 1.0,
            "order_type": "stop_market",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 1
    assert paired[0].side == "long"
    assert paired[0].exit_reason == "stop"


def test_pair_trades_short_target_hit():
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "sell",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "side": "buy",
            "price": 98.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 1
    assert paired[0].side == "short"
    assert paired[0].exit_reason == "target"


def test_pair_trades_short_stop_hit():
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "sell",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:31:00+00:00",
            "side": "buy",
            "price": 101.0,
            "quantity": 1.0,
            "order_type": "stop_market",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 1
    assert paired[0].side == "short"
    assert paired[0].exit_reason == "stop"


def test_pair_trades_multiple_round_trips_in_sequence():
    # Mirrors the actual SMA crossover demo: 3 round trips back-to-back.
    rows = [
        # Trade 1: long target
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "side": "sell",
            "price": 102.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
        # Trade 2: short stop
        {
            "timestamp": "2026-04-27 14:34:00+00:00",
            "side": "sell",
            "price": 99.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:35:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "stop_market",
        },
        # Trade 3: long stop
        {
            "timestamp": "2026-04-27 14:38:00+00:00",
            "side": "buy",
            "price": 98.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:39:00+00:00",
            "side": "sell",
            "price": 97.0,
            "quantity": 1.0,
            "order_type": "stop_market",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 3
    assert [t.side for t in paired] == ["long", "short", "long"]
    assert [t.exit_reason for t in paired] == ["target", "stop", "stop"]


def test_pair_trades_signal_flip_closes_open_position():
    # New market entry arrives before the previous bracket exits.
    # Defensive: close prior at the new market fill price, mark "flip".
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        # No bracket exit. Then a flip:
        {
            "timestamp": "2026-04-27 14:35:00+00:00",
            "side": "sell",
            "price": 101.5,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:38:00+00:00",
            "side": "buy",
            "price": 99.0,
            "quantity": 1.0,
            "order_type": "stop_market",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 2
    # First trade is the closed-out long, exited at the flip price.
    assert paired[0].side == "long"
    assert paired[0].exit_price == 101.5
    assert paired[0].exit_reason == "flip"
    # Second trade is the new short, exited at the stop.
    assert paired[1].side == "short"
    assert paired[1].exit_reason == "stop"


def test_pair_trades_unmatched_entry_dropped():
    # An entry with no following exit: nothing to pair => empty result.
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert paired == []


def test_pair_trades_empty_input():
    paired = pair_trades(pd.DataFrame())
    assert paired == []


def test_pair_trades_orphan_exit_skipped():
    # A bracket exit with no preceding entry must be silently skipped.
    rows = [
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "side": "sell",
            "price": 102.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
        {
            "timestamp": "2026-04-27 14:31:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:32:00+00:00",
            "side": "sell",
            "price": 103.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 1
    assert paired[0].entry_price == 100.0


def test_pair_trades_orders_get_sorted_by_timestamp():
    # Out-of-order orders must be sorted before pairing.
    rows = [
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "side": "sell",
            "price": 102.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 1
    assert paired[0].entry_ts < paired[0].exit_ts


def test_pair_trades_missing_columns_raises():
    df = pd.DataFrame([{"timestamp": "2026-04-27", "side": "buy"}])
    with pytest.raises(ValueError, match="missing required columns"):
        pair_trades(df)


def test_pair_trades_unknown_order_type_does_not_break_pairing():
    # An unknown order_type (e.g. a partial-fill cancel) must not
    # close an open entry. The next valid bracket exit should still
    # match the original entry.
    rows = [
        {
            "timestamp": "2026-04-27 14:29:00+00:00",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "order_type": "market",
        },
        {
            "timestamp": "2026-04-27 14:29:30+00:00",
            "side": "sell",
            "price": 0.0,
            "quantity": 0.0,
            "order_type": "cancel",
        },
        {
            "timestamp": "2026-04-27 14:30:00+00:00",
            "side": "sell",
            "price": 102.0,
            "quantity": 1.0,
            "order_type": "limit",
        },
    ]
    paired = pair_trades(_orders(rows))
    assert len(paired) == 1
    assert paired[0].entry_price == 100.0
    assert paired[0].exit_price == 102.0
