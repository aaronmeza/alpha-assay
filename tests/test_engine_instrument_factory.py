import pandas as pd
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import FuturesContract

from alpha_assay.engine.instrument_factory import make_es_futures, make_mes_futures


def test_make_es_futures_returns_futures_contract():
    venue = Venue("SIM")
    es = make_es_futures(venue)
    assert isinstance(es, FuturesContract)


def test_make_es_futures_defaults_to_esm6_50_multiplier():
    venue = Venue("SIM")
    es = make_es_futures(venue)
    assert str(es.raw_symbol) == "ESM6"
    assert int(es.multiplier) == 50
    assert es.asset_class == AssetClass.INDEX
    assert es.price_precision == 2
    assert str(es.price_increment) == "0.25"


def test_make_es_futures_activation_and_expiration_are_uint64_ns():
    venue = Venue("SIM")
    es = make_es_futures(venue)
    # Quarter-front-month: activation roughly 3 months before expiration.
    # Values are unix nanoseconds (int), comparable, activation < expiration.
    assert isinstance(es.activation_ns, int)
    assert isinstance(es.expiration_ns, int)
    assert es.activation_ns < es.expiration_ns


def test_make_es_futures_custom_symbol_and_multiplier():
    venue = Venue("SIM")
    es = make_es_futures(venue, symbol="ESZ6", multiplier=50)
    assert str(es.raw_symbol) == "ESZ6"


def test_make_mes_futures_has_5_multiplier():
    venue = Venue("SIM")
    mes = make_mes_futures(venue)
    assert int(mes.multiplier) == 5
    assert str(mes.raw_symbol) == "MESM6"
    assert mes.asset_class == AssetClass.INDEX


def test_make_mes_futures_brackets_week1_fixture_window():
    # The fixture uses 2026-04-28 bars; activation must precede
    # and expiration must follow, or Nautilus rejects bars with
    # "outside contract life".
    venue = Venue("SIM")
    mes = make_mes_futures(venue)
    fixture_ts = int(pd.Timestamp("2026-04-28", tz="UTC").value)
    assert mes.activation_ns <= fixture_ts
    assert mes.expiration_ns >= fixture_ts
