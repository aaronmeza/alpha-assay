# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Tests for the IBKR exec adapter .

Spec Section 10 Risk 3: flipping from PAPER to LIVE must require ALL
THREE locks independently. The 2^3=8 case matrix below is parameterized
explicitly so each lock is verified to reject on its own, with no
shared fixture coupling the three signals.

All tests mock ib_insync end-to-end; no network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from alpha_assay.cli import main as cli_main
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from alpha_assay.exec.ibkr import (
    ExecMode,
    IBKRExecAdapter,
    LiveModeLocks,
    build_exec_adapter,
    check_locks,
)
from alpha_assay.observability import metrics as M

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _gauge_value(gauge, **labels):
    if labels:
        return gauge.labels(**labels)._value.get()
    return gauge._value.get()


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


def _make_read_adapter() -> IBKRAdapter:
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = False
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = list(range(1000, 2000))
    return IBKRAdapter(ib=ib, read_only=False)


def _write_checklist(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("signed-by: test\n")


# ---------------------------------------------------------------------
# The 8-case lock matrix. Exactly one combination yields LIVE.
# Each case is independent -- no shared fixture sets two locks at once.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_set,cli_flag,checklist_signed,expected_mode,case_name",
    [
        (False, False, False, ExecMode.PAPER, "reject_000_all_missing"),
        (True, False, False, ExecMode.PAPER, "reject_100_env_only"),
        (False, True, False, ExecMode.PAPER, "reject_010_cli_only"),
        (False, False, True, ExecMode.PAPER, "reject_001_checklist_only"),
        (True, True, False, ExecMode.PAPER, "reject_110_env_and_cli"),
        (True, False, True, ExecMode.PAPER, "reject_101_env_and_checklist"),
        (False, True, True, ExecMode.PAPER, "reject_011_cli_and_checklist"),
        (True, True, True, ExecMode.LIVE, "accept_111_all_engaged"),
    ],
)
def test_lock_matrix_8_cases(
    env_set,
    cli_flag,
    checklist_signed,
    expected_mode,
    case_name,
    tmp_path,
):
    """The exhaustive 2^3=8 case matrix. Each case constructs its own
    inputs independently -- no fixture shares state across combinations.
    """
    checklist_path = tmp_path / f"signed_{case_name}"
    if checklist_signed:
        _write_checklist(checklist_path)
    env = {"ALPHA_ASSAY_LIVE": "1"} if env_set else {}

    locks = check_locks(env=env, cli_flag=cli_flag, checklist_path=checklist_path)
    assert locks.env_set is env_set
    assert locks.cli_flag is cli_flag
    assert locks.checklist_signed is checklist_signed
    assert locks.all_engaged() is (expected_mode is ExecMode.LIVE)

    read_adapter = _make_read_adapter()
    adapter = build_exec_adapter(
        adapter=read_adapter,
        env=env,
        cli_live=cli_flag,
        checklist_path=checklist_path,
        dry_run=True,
    )
    assert adapter.mode is expected_mode, (
        f"case {case_name}: env={env_set} cli={cli_flag} checklist={checklist_signed} "
        f"-> expected {expected_mode}, got {adapter.mode}"
    )


# ---------------------------------------------------------------------
# Purity + independence
# ---------------------------------------------------------------------


def test_check_locks_no_side_effects(tmp_path, monkeypatch):
    """check_locks must not modify os.environ or touch the filesystem."""
    import os

    before_env = dict(os.environ)
    checklist = tmp_path / "not_created"
    assert not checklist.exists()

    locks = check_locks(
        env={"ALPHA_ASSAY_LIVE": "1"},
        cli_flag=True,
        checklist_path=checklist,
    )
    assert locks.checklist_signed is False
    assert not checklist.exists()
    assert dict(os.environ) == before_env


def test_checklist_path_default_is_home_dir(monkeypatch, tmp_path):
    """Default checklist path resolves under Path.home()."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # With no env override, check_locks(path=None) must use ~/.alpha_assay/...
    locks = check_locks(env={}, cli_flag=False, checklist_path=None)
    assert locks.checklist_signed is False

    # Write file at the default path, assert now signed.
    default_path = fake_home / ".alpha_assay" / "go_live_checklist_signed"
    _write_checklist(default_path)
    locks2 = check_locks(env={}, cli_flag=False, checklist_path=None)
    assert locks2.checklist_signed is True


def test_checklist_path_env_override(monkeypatch, tmp_path):
    """ALPHA_ASSAY_CHECKLIST_PATH env overrides the default."""
    override = tmp_path / "override" / "signed"
    _write_checklist(override)
    env = {"ALPHA_ASSAY_CHECKLIST_PATH": str(override)}
    locks = check_locks(env=env, cli_flag=False, checklist_path=None)
    assert locks.checklist_signed is True


def test_live_mode_locks_all_engaged():
    assert LiveModeLocks(True, True, True).all_engaged() is True
    assert LiveModeLocks(True, True, False).all_engaged() is False
    assert LiveModeLocks(True, False, True).all_engaged() is False
    assert LiveModeLocks(False, True, True).all_engaged() is False
    assert LiveModeLocks(False, False, False).all_engaged() is False


def test_build_exec_adapter_warns_and_names_missing_locks(tmp_path, caplog):
    """PAPER fallback must emit a WARN log naming every missing lock."""
    read_adapter = _make_read_adapter()
    with caplog.at_level(logging.WARNING):
        adapter = build_exec_adapter(
            adapter=read_adapter,
            env={"ALPHA_ASSAY_LIVE": "1"},  # env engaged
            cli_live=False,  # cli missing
            checklist_path=tmp_path / "nope",  # checklist missing
            dry_run=True,
        )
    assert adapter.mode is ExecMode.PAPER
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "cli" in joined.lower()
    assert "checklist" in joined.lower()
    # env is engaged, so must NOT appear in the "missing" message
    # (sanity: the warning should reference only the missing ones).


# ---------------------------------------------------------------------
# Observability gauges
# ---------------------------------------------------------------------


def test_live_lock_state_gauges_updated_on_check(tmp_path):
    """The three per-lock gauges reflect the three lock signals."""
    checklist = tmp_path / "signed"
    _write_checklist(checklist)
    check_locks(env={"ALPHA_ASSAY_LIVE": "1"}, cli_flag=True, checklist_path=checklist)
    assert _gauge_value(M.live_lock_state, lock="env") == 1
    assert _gauge_value(M.live_lock_state, lock="cli") == 1
    assert _gauge_value(M.live_lock_state, lock="checklist") == 1

    check_locks(env={}, cli_flag=False, checklist_path=tmp_path / "nope")
    assert _gauge_value(M.live_lock_state, lock="env") == 0
    assert _gauge_value(M.live_lock_state, lock="cli") == 0
    assert _gauge_value(M.live_lock_state, lock="checklist") == 0


def test_exec_mode_gauge_reflects_active_mode(tmp_path):
    """PAPER adapter sets paper=1, live=0. LIVE adapter flips it."""
    read_adapter = _make_read_adapter()
    paper = build_exec_adapter(
        adapter=read_adapter,
        env={},
        cli_live=False,
        checklist_path=tmp_path / "missing",
        dry_run=True,
    )
    assert paper.mode is ExecMode.PAPER
    assert _gauge_value(M.exec_mode, mode="paper") == 1
    assert _gauge_value(M.exec_mode, mode="live") == 0

    checklist = tmp_path / "signed"
    _write_checklist(checklist)
    live = build_exec_adapter(
        adapter=_make_read_adapter(),
        env={"ALPHA_ASSAY_LIVE": "1"},
        cli_live=True,
        checklist_path=checklist,
        dry_run=True,
    )
    assert live.mode is ExecMode.LIVE
    assert _gauge_value(M.exec_mode, mode="paper") == 0
    assert _gauge_value(M.exec_mode, mode="live") == 1


# ---------------------------------------------------------------------
# Order submission
# ---------------------------------------------------------------------


_SPEC = {
    "symbol": "ES",
    "sec_type": "FUT",
    "exchange": "CME",
    "currency": "USD",
    "expiry": "202606",
}


def test_place_bracket_order_submits_three_orders(tmp_path):
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = [101, 102, 103]

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter, mode=ExecMode.PAPER)

    plan = exec_adapter.place_bracket_order(
        contract_spec=_SPEC,
        side="BUY",
        quantity=1,
        entry_type="LIMIT",
        stop_points=4.0,
        target_points=8.0,
        limit_price=5200.0,
    )
    assert ib.placeOrder.call_count == 3

    orders_by_id = {call.args[1].orderId: call.args[1] for call in ib.placeOrder.call_args_list}
    assert set(orders_by_id.keys()) == {101, 102, 103}

    parent = orders_by_id[101]
    target = orders_by_id[102]
    stop = orders_by_id[103]

    assert parent.action == "BUY"
    assert parent.orderType == "LMT"
    assert parent.lmtPrice == 5200.0
    assert parent.transmit is False  # children transmit last

    assert target.action == "SELL"
    assert target.orderType == "LMT"
    assert target.lmtPrice == 5208.0
    assert target.parentId == 101
    assert target.transmit is False

    assert stop.action == "SELL"
    assert stop.orderType == "STP"
    assert stop.auxPrice == 5196.0
    assert stop.parentId == 101
    assert stop.transmit is True  # bracket transmits on last child

    assert plan.parent_id == 101
    assert plan.target_id == 102
    assert plan.stop_id == 103
    assert plan.entry_price == 5200.0
    assert plan.stop_price == 5196.0  # 5200 - 4
    assert plan.target_price == 5208.0  # 5200 + 8


def test_place_bracket_order_sell_side_prices(tmp_path):
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = [201, 202, 203]

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)

    plan = exec_adapter.place_bracket_order(
        contract_spec=_SPEC,
        side="SELL",
        quantity=2,
        entry_type="LIMIT",
        stop_points=5.0,
        target_points=10.0,
        limit_price=5300.0,
    )
    # SELL entry: stop ABOVE entry, target BELOW entry.
    assert plan.stop_price == 5305.0
    assert plan.target_price == 5290.0


def test_place_bracket_order_market_entry_uses_limit_none(tmp_path):
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = [301, 302, 303]

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)

    with pytest.raises(ValueError, match="LIMIT entry requires limit_price"):
        exec_adapter.place_bracket_order(
            contract_spec=_SPEC,
            side="BUY",
            quantity=1,
            entry_type="LIMIT",
            stop_points=4.0,
            target_points=8.0,
            limit_price=None,
        )


def test_place_bracket_order_respects_dry_run():
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = [401, 402, 403]

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter, dry_run=True)

    plan = exec_adapter.place_bracket_order(
        contract_spec=_SPEC,
        side="BUY",
        quantity=1,
        entry_type="LIMIT",
        stop_points=4.0,
        target_points=8.0,
        limit_price=5000.0,
    )
    assert ib.placeOrder.call_count == 0
    assert plan.parent_id == 401
    assert plan.dry_run is True


def test_orders_submitted_counter_increments_with_type_labels():
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = [501, 502, 503]

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)

    before_parent = _counter_value(M.orders_submitted_total, type="parent")
    before_stop = _counter_value(M.orders_submitted_total, type="stop")
    before_target = _counter_value(M.orders_submitted_total, type="target")

    exec_adapter.place_bracket_order(
        contract_spec=_SPEC,
        side="BUY",
        quantity=1,
        entry_type="LIMIT",
        stop_points=4.0,
        target_points=8.0,
        limit_price=5000.0,
    )
    assert _counter_value(M.orders_submitted_total, type="parent") == before_parent + 1
    assert _counter_value(M.orders_submitted_total, type="stop") == before_stop + 1
    assert _counter_value(M.orders_submitted_total, type="target") == before_target + 1


def test_on_fill_callback_receives_canonical_schema():
    import datetime as _dt

    from ib_insync import CommissionReport, Contract, Execution, Fill, Order, Trade

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = [601, 602, 603]

    trades_by_id: dict[int, Trade] = {}

    def _place_order(contract, order):
        trade = Trade(contract=contract, order=order)
        trades_by_id[order.orderId] = trade
        return trade

    ib.placeOrder.side_effect = _place_order

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)

    received: list = []
    exec_adapter.on_fill(lambda evt: received.append(evt))

    exec_adapter.place_bracket_order(
        contract_spec=_SPEC,
        side="BUY",
        quantity=1,
        entry_type="LIMIT",
        stop_points=4.0,
        target_points=8.0,
        limit_price=5000.0,
    )

    fake_fill = Fill(
        contract=Contract(),
        execution=Execution(
            execId="E-1",
            time=_dt.datetime(2026, 5, 1, 14, 31, 0, tzinfo=_dt.UTC),
            shares=1.0,
            price=5000.25,
            side="BOT",
        ),
        commissionReport=CommissionReport(),
        time=_dt.datetime(2026, 5, 1, 14, 31, 0, tzinfo=_dt.UTC),
    )
    one_arg_trade = Trade(contract=Contract(), order=Order(orderId=601, orderType="LMT"))
    trades_by_id[601].filledEvent.emit(one_arg_trade)
    assert received == []

    # Fire only the parent's fillEvent. Each child has its own event,
    # so this exercises the single-order -> single-callback path.
    trades_by_id[601].fillEvent.emit(one_arg_trade, fake_fill)

    assert len(received) == 1
    evt = received[0]
    assert set(evt.keys()) == {
        "order_id",
        "order_type",
        "exec_id",
        "timestamp",
        "quantity",
        "price",
        "side",
    }
    assert evt["order_id"] == 601
    assert evt["exec_id"] == "E-1"
    assert evt["quantity"] == 1.0
    assert evt["price"] == 5000.25


def test_cancel_order_uses_retained_trade_order_and_unknown_logs(caplog):
    from ib_insync import Trade

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = [701, 702, 703]
    ib.trades.return_value = []

    def _place_order(contract, order):
        order.clientId = 23
        return Trade(contract=contract, order=order)

    ib.placeOrder.side_effect = _place_order

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)

    plan = exec_adapter.place_bracket_order(
        contract_spec=_SPEC,
        side="BUY",
        quantity=1,
        entry_type="LIMIT",
        stop_points=4.0,
        target_points=8.0,
        limit_price=5000.0,
    )
    exec_adapter.cancel_order(plan.parent_id)
    assert ib.cancelOrder.call_count == 1
    cancelled_order = ib.cancelOrder.call_args.args[0]
    assert cancelled_order.orderId == plan.parent_id
    assert cancelled_order.clientId == 23

    with caplog.at_level(logging.WARNING):
        exec_adapter.cancel_order(9999)
    assert "cancel_order(9999): no live trade found" in caplog.text


def test_terminal_cancelled_status_dispatches_order_status_event():
    import datetime as _dt

    from ib_insync import OrderStatus, Trade
    from ib_insync.objects import TradeLogEntry

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.return_value = 801

    trades_by_id: dict[int, Trade] = {}

    def _place_order(contract, order):
        trade = Trade(contract=contract, order=order)
        trades_by_id[order.orderId] = trade
        return trade

    ib.placeOrder.side_effect = _place_order

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)
    received: list[dict[str, object]] = []
    exec_adapter.on_order_status(received.append)

    order_id = exec_adapter.place_market_order(contract_spec=_SPEC, side="BUY", quantity=1)
    trade = trades_by_id[order_id]
    trade.orderStatus.status = OrderStatus.Cancelled
    trade.log.append(
        TradeLogEntry(
            time=_dt.datetime(2026, 5, 1, 14, 31, 0, tzinfo=_dt.UTC),
            status=OrderStatus.Cancelled,
            message="Order rejected",
            errorCode=201,
        )
    )
    trade.statusEvent.emit(trade)

    assert received == [
        {
            "order_id": 801,
            "status": OrderStatus.Cancelled,
            "error_code": 201,
            "message": "Order rejected",
        }
    ]


def test_benign_ibkr_notice_cancelled_status_is_suppressed():
    import datetime as _dt

    from ib_insync import OrderStatus, Trade
    from ib_insync.objects import TradeLogEntry

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.return_value = 811

    trades_by_id: dict[int, Trade] = {}

    def _place_order(contract, order):
        trade = Trade(contract=contract, order=order)
        trades_by_id[order.orderId] = trade
        return trade

    ib.placeOrder.side_effect = _place_order

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)
    received: list[dict[str, object]] = []
    exec_adapter.on_order_status(received.append)

    order_id = exec_adapter.place_market_order(contract_spec=_SPEC, side="BUY", quantity=1)
    trade = trades_by_id[order_id]
    trade.orderStatus.status = OrderStatus.Cancelled
    trade.log.append(
        TradeLogEntry(
            time=_dt.datetime(2026, 5, 1, 14, 31, 0, tzinfo=_dt.UTC),
            status=OrderStatus.Cancelled,
            message="Order TIF was set to DAY based on order preset.",
            errorCode=10349,
        )
    )
    trade.statusEvent.emit(trade)

    assert received == []


def test_adapter_logs_mode_prominently_on_connect(caplog):
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = False
    ib.client = MagicMock()
    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter, mode=ExecMode.PAPER)

    with caplog.at_level(logging.WARNING):
        exec_adapter.connect()

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "PAPER" in joined


# ---------------------------------------------------------------------
# CLI: alpha_assay live-check
# ---------------------------------------------------------------------


def test_live_check_cli_exits_2_when_any_lock_missing(tmp_path, monkeypatch):
    """The diagnostic CLI exits 2 when any of the three locks is missing."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("ALPHA_ASSAY_LIVE", raising=False)
    monkeypatch.delenv("ALPHA_ASSAY_CHECKLIST_PATH", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_main, ["live-check"])
    assert result.exit_code == 2
    combined = (result.output or "").lower()
    # Reasonable case: nothing is engaged, so all three locks must be named.
    assert "env" in combined
    assert "cli" in combined
    assert "checklist" in combined


def test_live_check_cli_exits_0_when_all_locks_engaged(tmp_path, monkeypatch):
    checklist = tmp_path / "signed"
    _write_checklist(checklist)
    monkeypatch.setenv("ALPHA_ASSAY_LIVE", "1")
    monkeypatch.setenv("ALPHA_ASSAY_CHECKLIST_PATH", str(checklist))

    runner = CliRunner()
    result = runner.invoke(cli_main, ["live-check", "--live"])
    assert result.exit_code == 0, result.output
    assert "LIVE" in result.output


def test_live_check_cli_flag_alone_is_not_enough(tmp_path, monkeypatch):
    """--live alone (no env, no checklist) must still exit 2."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("ALPHA_ASSAY_LIVE", raising=False)
    monkeypatch.delenv("ALPHA_ASSAY_CHECKLIST_PATH", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_main, ["live-check", "--live"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "env_set,cli_flag,checklist_signed,expected_exit,case_name",
    [
        (False, False, False, 2, "000_all_missing"),
        (True, False, False, 2, "100_env_only"),
        (False, True, False, 2, "010_cli_only"),
        (False, False, True, 2, "001_checklist_only"),
        (True, True, False, 2, "110_env_and_cli"),
        (True, False, True, 2, "101_env_and_checklist"),
        (False, True, True, 2, "011_cli_and_checklist"),
        (True, True, True, 0, "111_all_engaged"),
    ],
)
def test_live_check_cli_8_case_matrix(
    env_set, cli_flag, checklist_signed, expected_exit, case_name, tmp_path, monkeypatch
):
    """`alpha_assay live-check [--live]` exits 0 only when all three locks are engaged.

    The CLI is the operator's pre-flip gate, so cover the full 2^3 matrix
    here too - a two-of-three shortcut in the CLI plumbing would otherwise
    only be caught by the build_exec_adapter path (test_lock_matrix_8_cases).
    Each case sets its own env/flag/checklist; nothing is shared across rows.
    """
    if env_set:
        monkeypatch.setenv("ALPHA_ASSAY_LIVE", "1")
    else:
        monkeypatch.delenv("ALPHA_ASSAY_LIVE", raising=False)
    checklist = tmp_path / f"signed_{case_name}"
    if checklist_signed:
        _write_checklist(checklist)
    # Always pin the checklist path so a real ~/.alpha_assay file can't leak in.
    monkeypatch.setenv("ALPHA_ASSAY_CHECKLIST_PATH", str(checklist))

    args = ["live-check", "--live"] if cli_flag else ["live-check"]
    result = CliRunner().invoke(cli_main, args)

    assert result.exit_code == expected_exit, (
        f"case {case_name}: env={env_set} cli={cli_flag} checklist={checklist_signed} "
        f"-> expected exit {expected_exit}, got {result.exit_code}; output:\n{result.output}"
    )
    if expected_exit == 0:
        assert "mode=LIVE" in result.output
    else:
        assert "mode=PAPER" in result.output
        # The "missing locks:" line must name exactly the un-engaged locks.
        missing_line = next(ln for ln in result.output.splitlines() if ln.startswith("missing locks:"))
        named = {tok.strip() for tok in missing_line.split(":", 1)[1].split(",") if tok.strip()}
        expected_missing = (
            ({"env"} if not env_set else set())
            | ({"cli"} if not cli_flag else set())
            | ({"checklist"} if not checklist_signed else set())
        )
        assert (
            named == expected_missing
        ), f"case {case_name}: missing-locks line names {named}, expected {expected_missing}"


# ---------------------------------------------------------------------
# Loop-affinity marshaling (loop_owner): every ib_insync interaction
# must execute on the connection-owning loop thread, regardless of the
# calling thread. See alpha_assay.exec.loop_marshal for the model.
# ---------------------------------------------------------------------


@pytest.fixture()
def ib_loop():
    from alpha_assay.exec.loop_marshal import IBLoopThread

    lt = IBLoopThread(name="exec-test-ib-loop")
    lt.start()
    yield lt
    lt.stop()


def _make_loop_affine_ib(call_threads: dict):
    """MagicMock IB that records the thread of every loop-affine call."""
    import threading
    from types import SimpleNamespace

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.side_effect = list(range(700, 800))

    def _record(name):
        def _side_effect(*args, **kwargs):
            call_threads.setdefault(name, []).append(threading.current_thread())
            if name == "placeOrder":
                # No filledEvent attr -> handler wiring is skipped.
                return SimpleNamespace(order=args[1] if len(args) > 1 else kwargs.get("order"))
            return []

        return _side_effect

    ib.placeOrder.side_effect = _record("placeOrder")
    ib.cancelOrder.side_effect = _record("cancelOrder")
    ib.positions.side_effect = _record("positions")
    ib.openTrades.side_effect = _record("openTrades")
    ib.disconnect.side_effect = _record("disconnect")
    return ib


def test_exec_calls_from_worker_thread_run_on_loop_owner(ib_loop):
    """place_bracket_order / place_market_order / cancel_order /
    position_quantity / cancel_open_orders called from a NON-owner
    worker thread (the bus-consumer shape) must all execute their
    ib_insync interactions on the owning loop thread."""
    import threading

    call_threads: dict = {}
    ib = _make_loop_affine_ib(call_threads)
    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter, loop_owner=ib_loop)

    errors: list = []

    def worker():
        try:
            plan = exec_adapter.place_bracket_order(
                contract_spec=_SPEC,
                side="BUY",
                quantity=1,
                entry_type="LIMIT",
                stop_points=4.0,
                target_points=8.0,
                limit_price=5000.0,
            )
            assert plan.parent_id == 700
            order_id = exec_adapter.place_market_order(contract_spec=_SPEC, side="SELL", quantity=1)
            exec_adapter.cancel_order(order_id)
            assert exec_adapter.position_quantity(_SPEC) == 0.0
            assert exec_adapter.cancel_open_orders(_SPEC) == 0
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    t = threading.Thread(target=worker, name="bus-worker")
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "exec call from worker thread hung"
    assert errors == []

    loop_thread_name = "exec-test-ib-loop"
    assert len(call_threads["placeOrder"]) == 4  # 3 bracket children + 1 market
    for name in ("placeOrder", "cancelOrder", "positions", "openTrades"):
        threads = {th.name for th in call_threads[name]}
        assert threads == {loop_thread_name}, f"{name} ran on {threads}, expected the IB loop thread"


def test_exec_calls_without_loop_owner_stay_inline():
    """Regression guard for the legacy path: no loop_owner -> calls
    execute on the calling thread (tests with fakes rely on this)."""
    import threading

    call_threads: dict = {}
    ib = _make_loop_affine_ib(call_threads)
    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter)

    assert exec_adapter.position_quantity(_SPEC) == 0.0
    assert call_threads["positions"] == [threading.current_thread()]


def test_connect_marshals_and_fills_deliver_on_dispatcher_thread(ib_loop):
    """connect() must run the connect coroutine ON the owned loop (the
    transport binds to the loop current at connect time), and fill
    callbacks must be delivered on the dispatcher thread - never the IB
    loop thread - so a callback taking the runner lock cannot wedge the
    loop (lock-inversion deadlock; see loop_marshal docstring)."""
    import asyncio
    import threading
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    call_threads: dict = {}
    ib = _make_loop_affine_ib(call_threads)
    ib.isConnected.return_value = False
    connect_loops: list = []

    async def _connect_async(**kwargs):
        connect_loops.append(asyncio.get_running_loop())

    ib.connectAsync = AsyncMock(side_effect=_connect_async)

    read_adapter = IBKRAdapter(ib=ib, read_only=False)
    exec_adapter = IBKRExecAdapter(adapter=read_adapter, loop_owner=ib_loop)

    exec_adapter.connect()
    assert connect_loops == [ib_loop.loop]

    fill_threads: list = []
    delivered = threading.Event()

    def on_fill(evt):
        fill_threads.append(threading.current_thread())
        delivered.set()

    exec_adapter.on_fill(on_fill)
    handler = exec_adapter._make_fill_handler()
    fake_trade = SimpleNamespace(order=SimpleNamespace(orderId=1, orderType="MKT"))
    fake_fill = SimpleNamespace(
        execution=SimpleNamespace(execId="E-9", time=None, shares=1.0, price=5000.0, side="BOT")
    )
    # Fire from the loop thread, exactly where ib_insync fires filledEvent.
    ib_loop.call(handler, fake_trade, fake_fill)

    assert delivered.wait(timeout=5), "fill callback never delivered"
    assert fill_threads[0] is not ib_loop._thread, "fill delivered inline on the IB loop thread"
    assert fill_threads[0] is not threading.current_thread()

    ib.isConnected.return_value = True
    exec_adapter.disconnect()
    assert {th.name for th in call_threads["disconnect"]} == {"exec-test-ib-loop"}
    assert exec_adapter._fill_dispatcher is None
