"""Tests for stream name derivation + Message pack/unpack."""

import time

import pytest

from alpha_assay.bus.streams import (
    MalformedMessageError,
    Message,
    SchemaVersionError,
    pack,
    stream_name_for_bars,
    stream_name_for_ticks,
    unpack,
)


def test_stream_name_for_bars_es_future():
    spec = {"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260618"}
    assert stream_name_for_bars(spec) == "bars.es.cme.20260618"


def test_stream_name_for_bars_normalizes_case():
    # Mixed-case input must produce deterministic lowercase stream name.
    spec = {"symbol": "ES", "sec_type": "FUT", "exchange": "Cme", "expiry": "20260618"}
    assert stream_name_for_bars(spec) == "bars.es.cme.20260618"


def test_stream_name_for_bars_stock_no_expiry():
    spec = {"symbol": "AAPL", "sec_type": "STK", "exchange": "SMART"}
    assert stream_name_for_bars(spec) == "bars.aapl.smart"


def test_stream_name_for_ticks():
    assert stream_name_for_ticks("TICK-NYSE") == "ticks.tick-nyse"
    assert stream_name_for_ticks("AD-NYSE") == "ticks.ad-nyse"


def test_pack_unpack_roundtrip_bars():
    payload = {"open": 7250.5, "high": 7252.5, "low": 7246.5, "close": 7247.25, "volume": 14338}
    msg = Message(
        v=1,
        seq=1234,
        ts_recv_ns=time.time_ns(),
        ts_event_ns=time.time_ns() - 1_000_000,
        stream="bars.es.cme.20260618",
        payload=payload,
    )
    raw = pack(msg)
    decoded = unpack(raw)
    assert decoded.v == 1
    assert decoded.seq == 1234
    assert decoded.stream == "bars.es.cme.20260618"
    assert decoded.payload == payload


def test_pack_unpack_roundtrip_ticks():
    payload = {"value": 142.0, "symbol": "TICK-NYSE"}
    msg = Message(v=1, seq=99, ts_recv_ns=1, ts_event_ns=0, stream="ticks.tick-nyse", payload=payload)
    decoded = unpack(pack(msg))
    assert decoded == msg


def test_unpack_unknown_major_version_raises():
    # Forge a v=2 message; consumers must hard-fail.
    import msgpack

    raw = msgpack.packb(
        {"v": 2, "seq": 0, "ts_recv_ns": 0, "ts_event_ns": 0, "stream": "x", "payload": {}}
    )
    with pytest.raises(SchemaVersionError, match="major version"):
        unpack(raw)


def test_unpack_tolerates_unknown_minor_field():
    # Additive minor evolution: an extra field at top level should not break decoding.
    import msgpack

    raw = msgpack.packb(
        {
            "v": 1,
            "seq": 0,
            "ts_recv_ns": 0,
            "ts_event_ns": 0,
            "stream": "x",
            "payload": {},
            "future_optional_field": "ignored",
        }
    )
    msg = unpack(raw)
    assert msg.v == 1


def test_stream_name_for_bars_raises_on_missing_symbol():
    with pytest.raises(ValueError, match="symbol"):
        stream_name_for_bars({"sec_type": "FUT", "exchange": "CME"})


def test_stream_name_for_bars_raises_on_missing_exchange():
    with pytest.raises(ValueError, match="exchange"):
        stream_name_for_bars({"symbol": "ES", "sec_type": "FUT"})


def test_stream_name_for_bars_raises_on_empty_dict():
    with pytest.raises(ValueError):
        stream_name_for_bars({})


def test_stream_name_for_ticks_raises_on_empty_string():
    with pytest.raises(ValueError):
        stream_name_for_ticks("")


def test_unpack_raises_on_missing_required_field():
    import msgpack

    # Missing 'seq'.
    raw = msgpack.packb({"v": 1, "ts_recv_ns": 0, "ts_event_ns": 0, "stream": "x", "payload": {}})
    with pytest.raises(MalformedMessageError, match="seq"):
        unpack(raw)
