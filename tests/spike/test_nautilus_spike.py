"""NautilusTrader engine spike.

Goal: prove end-to-end wiring of NautilusTrader's BacktestEngine against
the project's sample_2d.csv fixture before committing to the engine
adapter work. No signal logic, no orders - just load data, run an
always-flat strategy, confirm zero errors.

Tested against nautilus_trader==1.225.0.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    AssetClass,
    BarAggregation,
    OmsType,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_2d.csv"
EXPECTED_COLUMNS = [
    "timestamp",
    "ES_open",
    "ES_high",
    "ES_low",
    "ES_close",
    "ES_volume",
    "TICK",
    "ADD",
]


def _load_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _make_es_futures(venue: Venue) -> FuturesContract:
    """Build an ES-like futures contract.

    ES spec: $50 multiplier, 0.25 tick, USD settled, CME. The activation
    and expiration dates are illustrative - they just need to bracket the
    fixture's timestamps so bars are accepted.
    """
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
    """Convert the fixture's OHLCV rows to Nautilus `Bar` objects.

    Rounding floats to 2dp can produce high<open or low>close relationships
    that Nautilus rejects, so we re-derive the OHLC envelope from the rounded
    values to guarantee high >= max(open, close) >= min(open, close) >= low.
    """
    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        ts = dt_to_unix_nanos(row.timestamp)
        o = round(row.ES_open, 2)
        h = round(row.ES_high, 2)
        lo = round(row.ES_low, 2)
        c = round(row.ES_close, 2)
        h = max(h, o, c)
        lo = min(lo, o, c)
        bar = Bar(
            bar_type,
            Price.from_str(f"{o:.2f}"),
            Price.from_str(f"{h:.2f}"),
            Price.from_str(f"{lo:.2f}"),
            Price.from_str(f"{c:.2f}"),
            Quantity.from_int(int(row.ES_volume)),
            ts,
            ts,
        )
        bars.append(bar)
    return bars


class AlwaysFlatStrategy(Strategy):
    """Receives every bar, places zero orders."""

    def __init__(self, bar_type: BarType) -> None:
        super().__init__()
        self._bar_type = bar_type
        self.bars_seen: int = 0

    def on_start(self) -> None:
        self.subscribe_bars(self._bar_type)

    def on_bar(self, bar: Bar) -> None:  # noqa: ARG002 - signature required
        self.bars_seen += 1


@pytest.fixture
def sim_venue() -> Venue:
    return Venue("SIM")


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return _load_fixture()


def _make_engine(venue: Venue, trader_id: str = "SPIKE-001") -> BacktestEngine:
    # bypass_logging avoids Rust-side logging re-init aborting the process
    # when multiple BacktestEngine instances are created in a single pytest run.
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


def test_nautilus_engine_runs_against_fixture(sim_venue: Venue, fixture_df: pd.DataFrame) -> None:
    """E3: BacktestEngine instantiates, SIM venue attaches, fixture loads."""
    assert fixture_df.shape[0] >= 300, "fixture must have >= 300 rows"
    assert list(fixture_df.columns) == EXPECTED_COLUMNS, f"unexpected columns: {list(fixture_df.columns)}"

    engine = _make_engine(sim_venue, trader_id="SPIKE-E3")
    try:
        assert engine.trader_id.value == "SPIKE-E3"
    finally:
        engine.dispose()


def test_always_flat_strategy_sees_every_bar(sim_venue: Venue, fixture_df: pd.DataFrame) -> None:
    """E4 + E5: AlwaysFlatStrategy receives every bar from an ES futures
    instrument, places zero orders.
    """
    engine = _make_engine(sim_venue, trader_id="SPIKE-E4")
    try:
        instrument = _make_es_futures(sim_venue)
        engine.add_instrument(instrument)

        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument.id, bar_spec, AggregationSource.EXTERNAL)

        bars = _df_to_bars(fixture_df, bar_type)
        assert len(bars) == len(fixture_df)

        engine.add_data(bars)

        strategy = AlwaysFlatStrategy(bar_type=bar_type)
        engine.add_strategy(strategy)

        engine.run()

        assert strategy.bars_seen == len(bars), f"strategy saw {strategy.bars_seen} bars, expected {len(bars)}"

        cache = engine.cache
        orders = cache.orders()
        assert len(orders) == 0, f"expected zero orders, got {len(orders)}"
    finally:
        engine.dispose()
