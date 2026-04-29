# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""pandas DataFrame -> list[CustomData] adapters for TICK + ADD feeds.

Per ADR Appendix C: BacktestEngine.add_data silently drops raw custom
Data subclasses. Events must be wrapped as
CustomData(data_type=DataType(Cls), data=inner) and ingested with an
explicit client_id. This module owns the wrapping. The engine core
(nautilus_runner.py) owns the client_id assignment at add_data time.
"""

from __future__ import annotations

import pandas as pd
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import CustomData, DataType
from nautilus_trader.model.identifiers import InstrumentId

from alpha_assay.engine.custom_data import AddIndicator, TickIndicator

_TICK_DATA_TYPE = DataType(TickIndicator)
_ADD_DATA_TYPE = DataType(AddIndicator)


def df_to_breadth(df: pd.DataFrame, instrument_id: InstrumentId) -> tuple[list[CustomData], list[CustomData]]:
    """Return (tick_events, add_events) wrapped in CustomData envelopes.

    Requires `timestamp`, `TICK`, and `ADD` columns. `timestamp` must be a
    timezone-aware datetime; the adapter converts to unix nanoseconds for
    Nautilus ingestion.
    """
    for col in ("timestamp", "TICK", "ADD"):
        if col not in df.columns:
            raise KeyError(f"df_to_breadth requires column {col!r}; got {list(df.columns)}")

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
        ticks.append(CustomData(data_type=_TICK_DATA_TYPE, data=tick))
        adds.append(CustomData(data_type=_ADD_DATA_TYPE, data=add))
    return ticks, adds
