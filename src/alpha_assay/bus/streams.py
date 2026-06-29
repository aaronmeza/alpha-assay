# SPDX-License-Identifier: Apache-2.0
"""Stream name derivation + msgpack message schema for the alpha-assay bus.

Stream names are deterministic from contract specs so the framework
stays strategy-agnostic: adding a new instrument is config, not code.

Schema versioning is enforced strictly: major version mismatch is a
hard fail (consumers exit), additive minor fields are tolerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgpack

if TYPE_CHECKING:
    import redis as redis_pkg

CURRENT_VERSION = 1


class SchemaVersionError(RuntimeError):
    """Raised when a consumer reads a message with an incompatible major version."""


class MalformedMessageError(RuntimeError):
    """Raised when a required wire field is missing from a message."""


@dataclass(frozen=True)
class Message:
    """Wire-format message on the bus.

    All timestamps are nanosecond epoch (UTC). Payload is feed-specific.
    """

    v: int
    seq: int
    ts_recv_ns: int
    ts_event_ns: int
    stream: str
    payload: dict[str, Any]


def stream_name_for_bars(contract_spec: dict[str, Any]) -> str:
    """Build a deterministic stream name for a bar feed.

    Format: ``bars.{symbol}.{venue}.{expiry}`` -- symbol/venue lowercased;
    expiry omitted for non-futures.

    Raises ValueError if 'symbol' or 'exchange' is missing or empty.
    """
    symbol = contract_spec.get("symbol")
    exchange = contract_spec.get("exchange")

    if not symbol or not exchange:
        raise ValueError(f"contract_spec missing required fields: needs 'symbol' and 'exchange'; got {contract_spec!r}")

    symbol_lower = str(symbol).lower()
    venue_lower = str(exchange).lower()
    parts = ["bars", symbol_lower, venue_lower]
    expiry = contract_spec.get("expiry")
    if expiry:
        parts.append(str(expiry))
    return ".".join(p for p in parts if p)


def bars_stream_name(symbol: str, venue: str, expiry: str | None = None) -> str:
    """Build a bar stream name from explicit ``(symbol, venue, expiry)``.

    Thin wrapper over :func:`stream_name_for_bars` for callers that hold
    the parts directly rather than a contract-spec dict - notably the
    roll-aware rebind path, which produces ``bars.<root>.<venue>.<E2>``
    when the front month rolls. Parameterized by ``(symbol, venue)`` with
    no hard-coded root, so a second instrument (e.g. NQ) shares the path.

    Raises ValueError if ``symbol`` or ``venue`` is missing or empty.
    """
    return stream_name_for_bars({"symbol": symbol, "exchange": venue, "expiry": expiry})


def bars_stream_has_data(redis_client: redis_pkg.Redis, symbol: str, venue: str, expiry: str) -> bool:
    """Non-destructively report whether a contract's bar stream has entries.

    Used by the consumer-side roll cutover as the *data gate*: the producer
    writes the new front-month key at the 08:00 CT pre-open re-qualify but
    keeps publishing the OLD per-contract stream until its next restart
    (``infra/feed/run.py`` ``_pre_open_requalify_loop`` - "stays on the
    boot-time stream by design"). A key-only rebind would therefore abandon
    the still-live old stream for an empty new one (silent starvation). The
    producer only begins publishing the new stream after that restart, so
    "the new stream has >=1 entry" is the reliable signal that the producer
    has truly switched and the consumer may cut over.

    Uses ``XLEN`` (O(1), read-only) - it never consumes entries, so it is
    safe to call on a stream the consumer has not yet attached to. A missing
    stream reports 0. Any Redis error is treated as "no data" so a transient
    blip holds the consumer on its current (live) stream rather than cutting
    over to an unverified one.
    """
    try:
        return int(redis_client.xlen(bars_stream_name(symbol, venue, expiry))) > 0
    except Exception:  # noqa: BLE001 - any Redis error -> hold (do not cut over)
        return False


def stream_name_for_ticks(symbol: str) -> str:
    """Build a deterministic stream name for a tick feed.

    Format: ``ticks.{symbol-lowered}``.

    Raises ValueError if symbol is None or empty.
    """
    if not symbol:
        raise ValueError("symbol must be non-empty string")
    return f"ticks.{symbol.lower()}"


def pack(msg: Message) -> bytes:
    """Serialize a Message to msgpack bytes."""
    return msgpack.packb(
        {
            "v": msg.v,
            "seq": msg.seq,
            "ts_recv_ns": msg.ts_recv_ns,
            "ts_event_ns": msg.ts_event_ns,
            "stream": msg.stream,
            "payload": msg.payload,
        },
        use_bin_type=True,
    )


def unpack(raw: bytes) -> Message:
    """Deserialize msgpack bytes back to a Message.

    Hard-fail on major version mismatch. Tolerate unknown additive fields.
    Raises MalformedMessageError if any required field is missing.
    """
    decoded = msgpack.unpackb(raw, raw=False)
    v = int(decoded.get("v", 0))
    if v != CURRENT_VERSION:
        raise SchemaVersionError(
            f"bus message major version {v} != supported {CURRENT_VERSION}; " f"coordinated upgrade required"
        )

    # Validate required fields.
    required_fields = ["seq", "ts_recv_ns", "ts_event_ns", "stream", "payload"]
    for field in required_fields:
        if field not in decoded:
            raise MalformedMessageError(f"required field missing: {field}")

    return Message(
        v=v,
        seq=int(decoded["seq"]),
        ts_recv_ns=int(decoded["ts_recv_ns"]),
        ts_event_ns=int(decoded["ts_event_ns"]),
        stream=str(decoded["stream"]),
        payload=dict(decoded["payload"]),
    )
