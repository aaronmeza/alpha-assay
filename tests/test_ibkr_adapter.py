# SPDX-License-Identifier: Apache-2.0
"""Tests for IBKRAdapter. All mock ib_insync end-to-end; no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from alpha_assay.data import ibkr_adapter
from alpha_assay.data.ibkr_adapter import IBKRAdapter
from alpha_assay.observability import metrics as M

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ibkr_bars_fixture.json"


def _load_bar_fixtures() -> list[SimpleNamespace]:
    """Load the JSON fixture into SimpleNamespace objects that quack like
    ib_insync BarData (attribute access on date/open/high/low/close/volume).
    """
    raw = json.loads(_FIXTURE_PATH.read_text())
    bars: list[SimpleNamespace] = []
    for row in raw:
        if "_fixture_docstring" in row:
            continue
        bars.append(
            SimpleNamespace(
                date=pd.Timestamp(row["date"]).to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                average=row["average"],
                barCount=row["barCount"],
            )
        )
    return bars


def _make_adapter(ib: MagicMock | None = None, **kwargs) -> IBKRAdapter:
    if ib is None:
        ib = MagicMock(name="IB")
        ib.isConnected.return_value = False
    return IBKRAdapter(ib=ib, **kwargs)


# --- lifecycle ---------------------------------------------------------


def test_adapter_instantiation_does_not_connect():
    ib = MagicMock(name="IB")
    IBKRAdapter(ib=ib)
    ib.connect.assert_not_called()


def test_connect_passes_host_port_client_id():
    ib = MagicMock(name="IB")
    adapter = IBKRAdapter(ib=ib, host="10.0.0.5", port=4002, client_id=42)
    adapter.connect()
    ib.connect.assert_called_once()
    kwargs = ib.connect.call_args.kwargs
    assert kwargs["host"] == "10.0.0.5"
    assert kwargs["port"] == 4002
    assert kwargs["clientId"] == 42


def test_connect_marks_read_only_true_by_default():
    ib = MagicMock(name="IB")
    adapter = IBKRAdapter(ib=ib)
    adapter.connect()
    assert ib.connect.call_args.kwargs["readonly"] is True


def test_read_only_flag_forwarded_to_ib_client():
    ib = MagicMock(name="IB")
    adapter = IBKRAdapter(ib=ib, read_only=False)
    adapter.connect()
    assert ib.connect.call_args.kwargs["readonly"] is False


def test_disconnect_is_idempotent():
    ib = MagicMock(name="IB")
    ib.isConnected.side_effect = [True, False]
    adapter = IBKRAdapter(ib=ib)
    adapter.disconnect()  # connected path
    adapter.disconnect()  # no-op path
    assert ib.disconnect.call_count == 1


def test_is_connected_delegates_to_ib():
    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True
    adapter = IBKRAdapter(ib=ib)
    assert adapter.is_connected is True
    ib.isConnected.return_value = False
    assert adapter.is_connected is False


# --- contract spec -----------------------------------------------------


def test_contract_spec_to_futures():
    contract = ibkr_adapter._build_contract(
        {
            "symbol": "ES",
            "sec_type": "FUT",
            "exchange": "CME",
            "currency": "USD",
            "expiry": "202606",
        }
    )
    # ib_insync Future sets secType='FUT' on construction
    assert contract.secType == "FUT"
    assert contract.symbol == "ES"
    assert contract.exchange == "CME"
    assert contract.currency == "USD"
    assert contract.lastTradeDateOrContractMonth == "202606"


def test_contract_spec_to_index():
    contract = ibkr_adapter._build_contract(
        {"symbol": "TICK-NYSE", "sec_type": "IND", "exchange": "NYSE", "currency": "USD"}
    )
    assert contract.secType == "IND"
    assert contract.symbol == "TICK-NYSE"
    assert contract.exchange == "NYSE"


def test_contract_spec_to_stock():
    contract = ibkr_adapter._build_contract(
        {"symbol": "SPY", "sec_type": "STK", "exchange": "SMART", "currency": "USD"}
    )
    assert contract.secType == "STK"
    assert contract.symbol == "SPY"


def test_contract_spec_unknown_sec_type_raises_valueerror():
    try:
        ibkr_adapter._build_contract(
            {"symbol": "BTCUSD", "sec_type": "CRYPTO", "exchange": "PAXOS"}
        )
    except ValueError as exc:
        msg = str(exc)
        assert "CRYPTO" in msg
        assert "unsupported sec_type" in msg
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown sec_type")


# --- bar subscription --------------------------------------------------


class _FakeEvent:
    """Minimal eventkit.Event stand-in supporting ``+=`` / ``-=``."""

    def __init__(self):
        self._handlers: list = []

    def __iadd__(self, h):
        self._handlers.append(h)
        return self

    def __isub__(self, h):
        if h in self._handlers:
            self._handlers.remove(h)
        return self

    def fire(self, *args, **kwargs):
        for h in list(self._handlers):
            h(*args, **kwargs)


class _FakeBarDataList(list):
    """Stand-in for ib_insync.BarDataList: list + updateEvent."""

    def __init__(self, bars):
        super().__init__(bars)
        self.updateEvent = _FakeEvent()

    def fire(self, new_bar):
        self.append(new_bar)
        self.updateEvent.fire(self, True)


async def _drain(async_iter, n):
    """Collect up to n events from an async generator then close it."""
    out = []
    try:
        for _ in range(n):
            out.append(await asyncio.wait_for(async_iter.__anext__(), timeout=1.0))
    finally:
        await async_iter.aclose()
    return out


def test_subscribe_bars_yields_canonical_schema():
    fixtures = _load_bar_fixtures()
    fake_list = _FakeBarDataList(fixtures[:3])

    ib = MagicMock(name="IB")
    # Adapter awaits reqHistoricalDataAsync (the async ib_insync API);
    # the AsyncMock makes the awaited result available without invoking
    # the real ib_insync reactor.
    ib.reqHistoricalDataAsync = AsyncMock(return_value=fake_list)

    adapter = IBKRAdapter(ib=ib)
    spec = {
        "symbol": "ES",
        "sec_type": "FUT",
        "exchange": "CME",
        "currency": "USD",
        "expiry": "202606",
    }

    events = asyncio.run(_drain(adapter.subscribe_bars(spec), 3))

    assert len(events) == 3
    for ev in events:
        assert set(ev.keys()) == {"timestamp", "open", "high", "low", "close", "volume", "feed"}
        assert isinstance(ev["timestamp"], pd.Timestamp)
        assert ev["timestamp"].tzinfo is not None
        assert str(ev["timestamp"].tz) == "UTC"
        assert ev["feed"] == "ES-FUT-202606"


def test_subscribe_bars_clamps_ohlc_invariants():
    # Fixture bar index 2 (0-indexed) has high=5200.00 < close=5202.25.
    fixtures = _load_bar_fixtures()
    violator = fixtures[2]
    assert violator.high < violator.close, "fixture bar 3 must violate the OHLC invariant"

    fake_list = _FakeBarDataList([violator])

    ib = MagicMock(name="IB")
    ib.reqHistoricalDataAsync = AsyncMock(return_value=fake_list)

    adapter = IBKRAdapter(ib=ib)
    spec = {"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "202606"}
    events = asyncio.run(_drain(adapter.subscribe_bars(spec), 1))

    ev = events[0]
    # After clamping, high >= max(open, close) and low <= min(open, close).
    assert ev["high"] >= max(ev["open"], ev["close"])
    assert ev["low"] <= min(ev["open"], ev["close"])
    assert ev["high"] == max(violator.high, violator.open, violator.close)


def test_subscribe_bars_increments_freshness_gauge():
    fixtures = _load_bar_fixtures()
    fake_list = _FakeBarDataList(fixtures[:1])

    ib = MagicMock(name="IB")
    ib.reqHistoricalDataAsync = AsyncMock(return_value=fake_list)

    adapter = IBKRAdapter(ib=ib)
    spec = {"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "202606"}
    asyncio.run(_drain(adapter.subscribe_bars(spec), 1))

    # Gauge is set to 0 on each event (adapter resets freshness on tick).
    val = M.ibkr_feed_freshness_seconds.labels(feed="ES-FUT-202606")._value.get()
    assert val == 0.0


# --- breadth subscription ---------------------------------------------


def test_subscribe_breadth_yields_tick_events():
    ib = MagicMock(name="IB")
    ib.pendingTickersEvent = _FakeEvent()

    fake_contract = SimpleNamespace()
    # Stash the contract we'll hand back from reqMktData so the adapter's
    # identity check matches.
    created = {}

    def _req_mkt_data(contract, *args, **kwargs):
        created["contract"] = contract
        return SimpleNamespace(contract=contract)

    ib.reqMktData.side_effect = _req_mkt_data

    adapter = IBKRAdapter(ib=ib)

    async def _run():
        gen = adapter.subscribe_breadth(symbol="TICK-NYSE")
        # Kick the generator so the pending handler is registered.
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)

        # Fire a tick whose contract matches what reqMktData returned.
        tick = SimpleNamespace(
            contract=created["contract"],
            last=245.0,
            close=240.0,
            time=pd.Timestamp("2026-04-21T14:30:00", tz="UTC"),
        )
        ib.pendingTickersEvent.fire([tick])

        ev = await asyncio.wait_for(task, timeout=1.0)
        await gen.aclose()
        return ev

    event = asyncio.run(_run())
    assert event["symbol"] == "TICK-NYSE"
    assert event["value"] == 245.0
    assert isinstance(event["timestamp"], pd.Timestamp)
    assert event["timestamp"].tzinfo is not None

    _ = fake_contract  # kept for readability


def test_subscribe_breadth_uses_bid_ask_midpoint_when_last_is_nan():
    """Regression test for the AD-NYSE field-precedence bug.

    AD-NYSE never populates ``.last`` (it is not a tradeable instrument);
    its live integer value comes through ``.bid`` and ``.ask`` as a
    tight quote pair. Falling through to ``.close`` returns the
    previous-session close (stale), which silently broke any breadth-aware strategy's
    bias filter on first real-Gateway deployment 2026-04-28. This test
    pins the bid/ask midpoint behavior so the regression cannot return.
    """
    import math

    ib = MagicMock(name="IB")
    ib.pendingTickersEvent = _FakeEvent()
    created = {}

    def _req_mkt_data(contract, *args, **kwargs):
        created["contract"] = contract
        return SimpleNamespace(contract=contract)

    ib.reqMktData.side_effect = _req_mkt_data
    adapter = IBKRAdapter(ib=ib)

    async def _run():
        gen = adapter.subscribe_breadth(symbol="AD-NYSE")
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)

        # Real-IBKR shape for AD-NYSE: last is NaN, close is stale, bid/ask
        # carry the live value. Live AD = (1259 + 1264) / 2 = 1261.5.
        tick = SimpleNamespace(
            contract=created["contract"],
            last=math.nan,
            close=209.0,  # previous-session close, must NOT be picked
            bid=1259.0,
            ask=1264.0,
            time=pd.Timestamp("2026-04-28T19:30:00", tz="UTC"),
        )
        ib.pendingTickersEvent.fire([tick])

        ev = await asyncio.wait_for(task, timeout=1.0)
        await gen.aclose()
        return ev

    event = asyncio.run(_run())
    assert event["symbol"] == "AD-NYSE"
    assert event["value"] == 1261.5, (
        f"expected bid/ask midpoint 1261.5, got {event['value']} "
        "(if 209.0 the recorder regressed to stale-close fallback)"
    )


# --- observability -----------------------------------------------------


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


def test_connection_event_counter_increments_on_connect_and_disconnect():
    before_conn = _counter_value(M.ibkr_connection_events_total, event="connected")
    before_disc = _counter_value(M.ibkr_connection_events_total, event="disconnected")

    ib = MagicMock(name="IB")
    ib.isConnected.return_value = True  # so disconnect() does real work
    adapter = IBKRAdapter(ib=ib)
    adapter.connect()
    adapter.disconnect()

    after_conn = _counter_value(M.ibkr_connection_events_total, event="connected")
    after_disc = _counter_value(M.ibkr_connection_events_total, event="disconnected")
    assert after_conn == before_conn + 1
    assert after_disc == before_disc + 1


def test_connection_event_counter_increments_on_error():
    before = _counter_value(M.ibkr_connection_events_total, event="error")
    ib = MagicMock(name="IB")
    ib.connect.side_effect = ConnectionRefusedError("boom")
    adapter = IBKRAdapter(ib=ib)
    try:
        adapter.connect()
    except ConnectionRefusedError:
        pass
    after = _counter_value(M.ibkr_connection_events_total, event="error")
    assert after == before + 1


# --- / boundary ---------------------------------------


def test_adapter_has_no_place_order_attribute():
    """is READ-only. Order submission is . This test
    documents the boundary: if someone adds place_order to this module
    they must do it in a sibling executor, not here.
    """
    adapter = _make_adapter()
    assert hasattr(adapter, "place_order") is False
    assert hasattr(adapter, "submit_order") is False
    assert hasattr(adapter, "send_order") is False
