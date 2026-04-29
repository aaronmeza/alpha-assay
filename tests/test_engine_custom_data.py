from nautilus_trader.core.data import Data
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import InstrumentId

from alpha_assay.engine.custom_data import AddIndicator, TickIndicator


def test_tick_indicator_is_nautilus_data_subclass():
    assert issubclass(TickIndicator, Data)


def test_add_indicator_is_nautilus_data_subclass():
    assert issubclass(AddIndicator, Data)


def test_tick_indicator_instantiates_with_signed_value():
    # NYSE TICK ranges roughly -1500 to +1500; signed. Proves the
    # customdataclass decoration accepted the float annotation.
    instrument_id = InstrumentId.from_str("ES.SIM")
    t = TickIndicator(
        ts_event=1_700_000_000_000_000_000,
        ts_init=1_700_000_000_000_000_000,
        instrument_id=instrument_id,
        value=-842.0,
    )
    assert t.value == -842.0
    assert t.instrument_id == instrument_id


def test_add_indicator_instantiates_with_signed_value():
    instrument_id = InstrumentId.from_str("ES.SIM")
    a = AddIndicator(
        ts_event=1_700_000_000_000_000_000,
        ts_init=1_700_000_000_000_000_000,
        instrument_id=instrument_id,
        value=1250.0,
    )
    assert a.value == 1250.0


def test_custom_data_types_registerable():
    # DataType(Cls) is how BacktestEngine's DataEngine dispatches
    # custom data. Must not raise.
    tick_dt = DataType(TickIndicator)
    add_dt = DataType(AddIndicator)
    assert tick_dt.type is TickIndicator
    assert add_dt.type is AddIndicator


def test_module_does_not_import_future_annotations():
    # Invariant from ADR Appendix C: PEP 563 stringified annotations break
    # @customdataclass. Read the module source and assert the forbidden
    # import line is absent.
    from pathlib import Path

    import alpha_assay.engine.custom_data as mod

    source = Path(mod.__file__).read_text()
    assert "from __future__ import annotations" not in source, (
        "custom_data.py MUST NOT use `from __future__ import annotations`; "
        "PEP 563 stringified annotations break @customdataclass per ADR Appendix C"
    )
