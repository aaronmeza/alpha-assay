# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Custom NautilusTrader Data subtypes for NYSE breadth indicators.

TICK and ADD are NYSE index aggregates. TICK is the difference between
stocks ticking up minus stocks ticking down on NYSE (roughly -1500 to
+1500). ADD is the advance-decline line on NYSE (roughly -2000 to +2000).
Both values are signed integers; Nautilus `Price` rejects negatives, so
these cannot ride on the Bar/Quote/Trade types. Custom Data subtypes
(option a from the day-1 ingestion spike) are the idiomatic path.

IMPORTANT: this module MUST NOT enable PEP 563 stringified annotations
(the `__future__` annotations import). The `@customdataclass` decorator
inspects `cls.__annotations__` at decoration time and dispatches on the
annotation's `__name__`. Stringified annotations make those values plain
strings (no `__name__`), and the decorator rejects them with
`TypeError: Unsupported custom data annotation`. A regression test reads
this module's source and asserts the forbidden line is absent. This
constraint is captured in ADR Appendix C.
"""

from nautilus_trader.core.data import Data
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.identifiers import InstrumentId


@customdataclass
class TickIndicator(Data):
    """NYSE TICK index value at a given bar boundary."""

    instrument_id: InstrumentId = InstrumentId.from_str("ES.SIM")
    value: float = 0.0


@customdataclass
class AddIndicator(Data):
    """NYSE ADD (advance-decline line) value at a given bar boundary."""

    instrument_id: InstrumentId = InstrumentId.from_str("ES.SIM")
    value: float = 0.0
