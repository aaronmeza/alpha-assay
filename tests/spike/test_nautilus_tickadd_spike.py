"""NautilusTrader TICK/ADD ingestion mini-spike (day-1 prereq #3).

Purpose: pick between the three candidate data-path shapes for NYSE breadth
indicators (`$TICK`, `$ADD`) alongside the ES bar stream before writing the
real engine adapter:

(a) Custom `Data` subtypes per feed (`TickIndicator`, `AddIndicator`) published
    alongside bars on the same event queue, subscribed in the strategy.
(b) Secondary synthetic instruments (e.g. an `Equity` mocked as TICK-NYSE)
    publishing Bars the strategy subscribes to. Rejected on a premise
    check: Nautilus `Price` is non-negative; TICK/ADD both go negative.
    Offsetting destroys information and is ugly. No spike needed to falsify.
(c) Actor-side pandas frames held in the strategy, looked up by timestamp
    inside `on_bar`. Simple, but sidesteps the engine's event queue so
    backtest-live parity is broken: IBKR live sends TICK/ADD as independent
    ticks on a separate client, which is option (a)-shaped.

This spike proves option (a) runs end-to-end on the sample_2d fixture.
Acceptance:
    - The strategy receives every TICK event and every ADD event (780 each
      on the 2-day fixture).
    - Custom-data events interleave with bars in time order (no all-bars-
      then-all-indicators artifact).
    - Zero orders placed (always-flat).

Verdict written in the research note; this file is the executable
evidence.
"""

# IMPORTANT: no `from __future__ import annotations` here.
# Nautilus `@customdataclass` inspects `cls.__annotations__` at decoration
# time and dispatches on the annotation's `__name__`. PEP 563 stringified
# annotations make those values plain strings (no `__name__`), which the
# decorator rejects with `Unsupported custom data annotation: 'InstrumentId'`.
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarSpecification, BarType, CustomData, DataType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    AssetClass,
    BarAggregation,
    OmsType,
    PriceType,
)
from nautilus_trader.model.identifiers import ClientId, InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

# Import the production custom-data types directly. The spike originally
# declared its own copies of these classes, but once shipped the
# canonical module, having a second set of @customdataclass registrations
# collided in the serializable-type registry during full-suite runs.
from alpha_assay.engine.custom_data import AddIndicator, TickIndicator

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_2d.csv"


# Nautilus `Price` rejects negative values, so TICK/ADD (which are signed
# integers) cannot ride on the Bar/Quote/Trade machinery. Custom Data
# subtypes are the idiomatic escape hatch; `@customdataclass` wires up
# serialization + Arrow schemas for free. TickIndicator and AddIndicator
# are imported from alpha_assay.engine.custom_data (canonical
# module) rather than redefined here.


def _load_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _make_es_futures(venue: Venue) -> FuturesContract:
    instrument_id = InstrumentId(Symbol("ESM6"), venue)
    activation = pd.Timestamp("2026-03-21", tz="UTC")
    expiration = pd.Timestamp("2026-06-19", tz="UTC")
    return FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol("ESM6"),
        asset_class=AssetClass.INDEX,
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.25"),
        multiplier=Quantity.from_int(50),
        lot_size=Quantity.from_int(1),
        underlying="ES",
        activation_ns=int(activation.value),
        expiration_ns=int(expiration.value),
        ts_event=0,
        ts_init=0,
    )


def _df_to_bars(df: pd.DataFrame, bar_type: BarType) -> list[Bar]:
    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        ts = dt_to_unix_nanos(row.timestamp)
        o = round(row.ES_open, 2)
        h = round(row.ES_high, 2)
        lo = round(row.ES_low, 2)
        c = round(row.ES_close, 2)
        # Defensive OHLC clamp per spike Appendix A: Nautilus
        # rejects bars where high < open after float rounding.
        h = max(h, o, c)
        lo = min(lo, o, c)
        bars.append(
            Bar(
                bar_type,
                Price.from_str(f"{o:.2f}"),
                Price.from_str(f"{h:.2f}"),
                Price.from_str(f"{lo:.2f}"),
                Price.from_str(f"{c:.2f}"),
                Quantity.from_int(int(row.ES_volume)),
                ts,
                ts,
            )
        )
    return bars


TICK_DATA_TYPE = DataType(TickIndicator)
ADD_DATA_TYPE = DataType(AddIndicator)


def _df_to_breadth(df: pd.DataFrame, instrument_id: InstrumentId) -> tuple[list[CustomData], list[CustomData]]:
    # BacktestEngine's DataEngine.process dispatch only routes custom data
    # when wrapped in a CustomData(data_type, data) envelope. Raw custom
    # Data subclasses fall through to a silent log.error("unrecognized
    # type") with logging bypassed. This wrapper is the load-bearing
    # bit that makes option (a) actually work.
    ticks: list[CustomData] = []
    adds: list[CustomData] = []
    for row in df.itertuples(index=False):
        ts = dt_to_unix_nanos(row.timestamp)
        tick = TickIndicator(
            ts_event=ts,
            ts_init=ts,
            instrument_id=instrument_id,
            value=float(row.TICK),
        )
        add = AddIndicator(
            ts_event=ts,
            ts_init=ts,
            instrument_id=instrument_id,
            value=float(row.ADD),
        )
        ticks.append(CustomData(data_type=TICK_DATA_TYPE, data=tick))
        adds.append(CustomData(data_type=ADD_DATA_TYPE, data=add))
    return ticks, adds


@dataclass
class BreadthEvent:
    ts_ns: int
    kind: str  # "bar" | "tick" | "add"
    value: float


class BreadthAwareStrategy(Strategy):
    """Subscribes to ES bars plus TICK and ADD custom data, records
    every event so the test can assert interleaving + totals.
    """

    def __init__(self, bar_type: BarType, instrument_id: InstrumentId) -> None:
        super().__init__()
        self._bar_type = bar_type
        self._instrument_id = instrument_id
        self.events: list[BreadthEvent] = []

    def on_start(self) -> None:
        self.subscribe_bars(self._bar_type)
        self.subscribe_data(
            DataType(TickIndicator),
            instrument_id=self._instrument_id,
        )
        self.subscribe_data(
            DataType(AddIndicator),
            instrument_id=self._instrument_id,
        )

    def on_bar(self, bar: Bar) -> None:
        self.events.append(
            BreadthEvent(ts_ns=bar.ts_event, kind="bar", value=float(bar.close)),
        )

    def on_data(self, data: Data) -> None:
        if isinstance(data, TickIndicator):
            self.events.append(
                BreadthEvent(ts_ns=data.ts_event, kind="tick", value=data.value),
            )
        elif isinstance(data, AddIndicator):
            self.events.append(
                BreadthEvent(ts_ns=data.ts_event, kind="add", value=data.value),
            )


@pytest.fixture
def sim_venue() -> Venue:
    return Venue("SIM")


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return _load_fixture()


def _make_engine(venue: Venue, trader_id: str) -> BacktestEngine:
    config = BacktestEngineConfig(
        trader_id=trader_id,
        logging=LoggingConfig(bypass_logging=True),
    )
    engine = BacktestEngine(config=config)
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
        default_leverage=Decimal(1),
    )
    return engine


def test_custom_breadth_data_flows_to_strategy(sim_venue: Venue, fixture_df: pd.DataFrame) -> None:
    """Option (a) end-to-end: custom TICK + ADD Data subtypes are received
    by a Strategy subscribed via DataType+instrument_id, in bar-time order.
    """
    engine = _make_engine(sim_venue, trader_id="TICKADD-SPIKE")
    try:
        instrument = _make_es_futures(sim_venue)
        engine.add_instrument(instrument)

        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument.id, bar_spec, AggregationSource.EXTERNAL)

        bars = _df_to_bars(fixture_df, bar_type)
        ticks, adds = _df_to_breadth(fixture_df, instrument.id)
        assert len(bars) == len(fixture_df)
        assert len(ticks) == len(fixture_df)
        assert len(adds) == len(fixture_df)

        # Nautilus sorts events by ts_init internally, so order of add_data
        # calls does not affect event-loop ordering. We still exercise
        # mixed ordering to prove that. Custom data needs an explicit
        # client_id since it doesn't have a direct instrument_id at the
        # envelope level - the engine validation reads the inner data's
        # instrument_id for routing but requires a client_id on ingestion.
        engine.add_data(bars)
        client_id = ClientId("BREADTH")
        engine.add_data(ticks, client_id=client_id)
        engine.add_data(adds, client_id=client_id)

        strategy = BreadthAwareStrategy(bar_type=bar_type, instrument_id=instrument.id)
        engine.add_strategy(strategy)
        engine.run()

        by_kind: dict[str, int] = {"bar": 0, "tick": 0, "add": 0}
        for event in strategy.events:
            by_kind[event.kind] += 1
        assert by_kind["bar"] == len(bars), f"expected {len(bars)} bars, got {by_kind['bar']}"
        assert by_kind["tick"] == len(ticks), f"expected {len(ticks)} TICK events, got {by_kind['tick']}"
        assert by_kind["add"] == len(adds), f"expected {len(adds)} ADD events, got {by_kind['add']}"

        # Interleave check: for every bar at ts T, the matching TICK and
        # ADD events occur at the same ts T. If the engine batched all
        # bars first and all breadth events after (or vice versa), we
        # would see long runs of a single kind; assert no such run.
        # Each fixture row produces (bar, tick, add) at the same ts_event,
        # so the max contiguous same-kind run in a well-interleaved log
        # should be small (<= 3 typically for same-ts ties depending on
        # the engine's intra-ts ordering).
        max_run = 0
        current_kind: str | None = None
        current_run = 0
        for event in strategy.events:
            if event.kind == current_kind:
                current_run += 1
            else:
                current_kind = event.kind
                current_run = 1
            max_run = max(max_run, current_run)

        # A batched-all-one-kind artifact would produce a run equal to
        # len(fixture_df); interleaved traffic yields a small run.
        assert max_run < 10, (
            f"events look batched by kind (max_run={max_run}); " "custom data is not interleaving with bars as expected"
        )

        cache = engine.cache
        assert len(cache.orders()) == 0, "always-flat expected zero orders"
    finally:
        engine.dispose()
