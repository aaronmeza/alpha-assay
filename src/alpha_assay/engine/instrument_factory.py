# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""FuturesContract factories for ES and MES.

Wraps the Nautilus construction ceremony captured in ADR Appendix A:
activation/expiration as uint64 nanoseconds (not datetime); multiplier
and lot_size as Quantity; AssetClass.INDEX for equity-index futures.
Default contract month ESM6 brackets the synthetic fixture.
"""

from __future__ import annotations

import pandas as pd
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Price, Quantity


def _futures_contract(
    *,
    venue: Venue,
    symbol: str,
    underlying: str,
    multiplier: int,
    activation: pd.Timestamp,
    expiration: pd.Timestamp,
) -> FuturesContract:
    instrument_id = InstrumentId(Symbol(symbol), venue)
    return FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        asset_class=AssetClass.INDEX,
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.25"),
        multiplier=Quantity.from_int(multiplier),
        lot_size=Quantity.from_int(1),
        underlying=underlying,
        activation_ns=int(activation.value),
        expiration_ns=int(expiration.value),
        ts_event=0,
        ts_init=0,
    )


def make_es_futures(
    venue: Venue,
    *,
    symbol: str = "ESM6",
    multiplier: int = 50,
    activation: pd.Timestamp | None = None,
    expiration: pd.Timestamp | None = None,
) -> FuturesContract:
    """Return an ES futures contract. Default ESM6, $50/pt, brackets
    2026 Q2 window."""
    return _futures_contract(
        venue=venue,
        symbol=symbol,
        underlying="ES",
        multiplier=multiplier,
        activation=activation or pd.Timestamp("2026-03-21", tz="UTC"),
        expiration=expiration or pd.Timestamp("2026-06-19", tz="UTC"),
    )


def make_mes_futures(
    venue: Venue,
    *,
    symbol: str = "MESM6",
    multiplier: int = 5,
    activation: pd.Timestamp | None = None,
    expiration: pd.Timestamp | None = None,
) -> FuturesContract:
    """Return an MES futures contract. Default MESM6, $5/pt, brackets
    2026 Q2 window. MES default limits live blast radius 10x per spec
    decision 2."""
    return _futures_contract(
        venue=venue,
        symbol=symbol,
        underlying="ES",
        multiplier=multiplier,
        activation=activation or pd.Timestamp("2026-03-21", tz="UTC"),
        expiration=expiration or pd.Timestamp("2026-06-19", tz="UTC"),
    )
