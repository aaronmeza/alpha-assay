# SPDX-License-Identifier: Apache-2.0
"""Stream name derivation + msgpack message schema for the alpha-assay bus.

Stream names are deterministic from contract specs so the framework
stays strategy-agnostic: adding a new instrument is config, not code.

Schema versioning is enforced strictly: major version mismatch is a
hard fail (consumers exit), additive minor fields are tolerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgpack

CURRENT_VERSION = 1


class SchemaVersionError(RuntimeError):
    """Raised when a consumer reads a message with an incompatible major version."""


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
    """
    symbol = str(contract_spec.get("symbol", "")).lower()
    venue = str(contract_spec.get("exchange", "")).lower()
    parts = ["bars", symbol, venue]
    expiry = contract_spec.get("expiry")
    if expiry:
        parts.append(str(expiry))
    return ".".join(p for p in parts if p)


def stream_name_for_ticks(symbol: str) -> str:
    """Build a deterministic stream name for a tick feed.

    Format: ``ticks.{symbol-lowered}``.
    """
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
    """
    decoded = msgpack.unpackb(raw, raw=False)
    v = int(decoded.get("v", 0))
    if v != CURRENT_VERSION:
        raise SchemaVersionError(
            f"bus message major version {v} != supported {CURRENT_VERSION}; "
            f"coordinated upgrade required"
        )
    return Message(
        v=v,
        seq=int(decoded["seq"]),
        ts_recv_ns=int(decoded["ts_recv_ns"]),
        ts_event_ns=int(decoded["ts_event_ns"]),
        stream=str(decoded["stream"]),
        payload=dict(decoded["payload"]),
    )
