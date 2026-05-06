# SPDX-License-Identifier: Apache-2.0
"""Append-mode trade-record writer for the paper trial dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_COLUMNS = (
    "timestamp",
    "signal_type",
    "entry_price",
    "stop",
    "target",
    "mock_fill_price",
    "mock_pnl_dollars",
    "account_balance_after",
)


@dataclass(frozen=True)
class TradeRecord:
    timestamp: pd.Timestamp
    signal_type: str
    entry_price: float
    stop: float
    target: float
    mock_fill_price: float
    mock_pnl_dollars: float
    account_balance_after: float


class TradeLog:
    """Buffered append to ``{out_dir}/trades.parquet``.

    Writes entire file on flush (parquet doesn't append cheaply at row
    granularity). Buffer is unbounded for now - flush per session
    rollover or on shutdown.
    """

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._out_dir / "trades.parquet"
        self._buffer: list[TradeRecord] = []
        # Load existing rows if file exists so we don't overwrite history.
        if self._path.exists():
            df = pd.read_parquet(self._path)
            for row in df.to_dict("records"):
                self._buffer.append(TradeRecord(**row))

    def write(self, record: TradeRecord) -> None:
        self._buffer.append(record)

    def flush(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame([asdict(r) for r in self._buffer], columns=list(_COLUMNS))
        df.to_parquet(self._path, index=False)
