# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""End-to-end tests for `alpha_assay backfill`.

The IBKRAdapter is patched at the CLI module's import site so no
network or ib_insync is touched. The mocked adapter returns
deterministic per-chunk bar lists; the test verifies per-day shards
land on disk in the canonical schema accepted by
``databento_adapter.load_parquet`` and that re-running merges rather
than clobbers.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from click.testing import CliRunner

from alpha_assay.cli import main as cli_main
from alpha_assay.data.databento_adapter import load_parquet

# Force a real handle to the submodule. ``alpha_assay.cli.__init__``
# re-binds the name ``backfill`` to the click Command, which shadows
# the submodule on plain attribute access. ``import_module`` returns
# the actual module object regardless.
backfill_module = importlib.import_module("alpha_assay.cli.backfill")


def _mk_bar(ts_iso: str, *, o: float, h: float, low: float, c: float, vol: int) -> dict:
    return {
        "timestamp": pd.Timestamp(ts_iso, tz="UTC"),
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": vol,
        "feed": "ES-FUT-20260618",
    }


def _patched_adapter(monkeypatch, chunk_returns: list[list[dict]]) -> MagicMock:
    """Patch ``alpha_assay.cli.backfill.IBKRAdapter`` to return ``chunk_returns``.

    Each call to ``historical_bars_async`` pops the next list off
    ``chunk_returns``; once exhausted, returns ``[]``. ``connect_async``
    and ``disconnect_async`` are AsyncMocks so the run path completes.
    """
    instance = MagicMock(name="IBKRAdapter")
    instance.connect_async = AsyncMock(return_value=None)
    instance.disconnect_async = AsyncMock(return_value=None)

    pending = list(chunk_returns)

    async def _fake_hist(*args, **kwargs):
        if pending:
            return pending.pop(0)
        return []

    instance.historical_bars_async = AsyncMock(side_effect=_fake_hist)
    factory = MagicMock(return_value=instance)
    # Patch via the imported module object: the dotted-path form of
    # ``alpha_assay.cli.backfill`` resolves to the click Command object
    # (re-exported by ``alpha_assay.cli.__init__``), not the module.
    monkeypatch.setattr(backfill_module, "IBKRAdapter", factory)
    return instance


def test_backfill_writes_per_day_shards(monkeypatch, tmp_path: Path):
    # Two chunks, each returning bars on a single distinct day so we
    # can lock in the per-day grouping without fighting timezone math.
    # 13:30 UTC == 08:30 CT (RTH open).
    chunk_a = [
        _mk_bar("2026-04-20T13:30:00", o=5200.0, h=5201.0, low=5199.0, c=5200.5, vol=10),
        _mk_bar("2026-04-20T13:31:00", o=5200.5, h=5202.0, low=5200.0, c=5201.5, vol=12),
    ]
    chunk_b = [
        _mk_bar("2026-04-21T13:30:00", o=5210.0, h=5211.0, low=5209.0, c=5210.5, vol=15),
        _mk_bar("2026-04-21T13:31:00", o=5210.5, h=5212.0, low=5210.0, c=5211.5, vol=20),
    ]
    adapter = _patched_adapter(monkeypatch, [chunk_a, chunk_b])

    out = tmp_path / "es_bars"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backfill",
            "--feed",
            "es-bars",
            "--days",
            "10",
            "--out",
            str(out),
            "--chunk-duration",
            "1 W",
            "--pace-seconds",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    # The CLI walks at least 2 chunks for a 10-day window with 1 W slices.
    assert adapter.historical_bars_async.await_count >= 2
    adapter.connect_async.assert_awaited_once()
    adapter.disconnect_async.assert_awaited_once()

    shards = sorted(out.glob("*.parquet"))
    shard_names = [s.name for s in shards]
    assert "2026-04-20.parquet" in shard_names
    assert "2026-04-21.parquet" in shard_names

    # Each shard validates against the Databento canonical schema +
    # carries the bars we expect.
    df_a = load_parquet(out / "2026-04-20.parquet")
    df_b = load_parquet(out / "2026-04-21.parquet")
    assert len(df_a) == 2
    assert len(df_b) == 2
    assert list(df_a.columns) == ["open", "high", "low", "close", "volume"]
    assert df_a["open"].iloc[0] == pytest.approx(5200.0)
    assert df_b["close"].iloc[-1] == pytest.approx(5211.5)


def test_backfill_resume_safe_merges_existing_shards(monkeypatch, tmp_path: Path):
    """Re-running over an existing shard sort+dedupes rather than clobbering.

    Pre-write a shard with one bar at 13:30 UTC; the mocked backfill
    returns a second bar at 13:31 UTC plus a *re-pull* of the 13:30
    bar (with a different volume). The post-merge shard must contain
    both timestamps, and the duplicate must keep the new value.
    """
    out = tmp_path / "es_bars"
    out.mkdir(parents=True)

    existing = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-20T13:30:00", tz="UTC"),
                "open": 5200.0,
                "high": 5201.0,
                "low": 5199.0,
                "close": 5200.5,
                "volume": 1,  # stale; backfill should override
            }
        ]
    )
    existing.to_parquet(out / "2026-04-20.parquet", index=False)

    new_bars = [
        _mk_bar("2026-04-20T13:30:00", o=5200.0, h=5201.0, low=5199.0, c=5200.5, vol=999),
        _mk_bar("2026-04-20T13:31:00", o=5200.5, h=5202.0, low=5200.0, c=5201.5, vol=12),
    ]
    _patched_adapter(monkeypatch, [new_bars])

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backfill",
            "--feed",
            "es-bars",
            "--days",
            "1",  # one chunk worth
            "--out",
            str(out),
            "--chunk-duration",
            "1 W",
            "--pace-seconds",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    df = load_parquet(out / "2026-04-20.parquet")
    assert len(df) == 2
    # Duplicate timestamp resolved to the new (backfill) volume.
    first_ts = df.index[0]
    second_ts = df.index[1]
    assert first_ts == pd.Timestamp("2026-04-20T13:30:00", tz="UTC").tz_convert("America/Chicago")
    assert second_ts == pd.Timestamp("2026-04-20T13:31:00", tz="UTC").tz_convert("America/Chicago")
    assert df["volume"].iloc[0] == 999
    # Strictly monotonic - load_parquet would have raised otherwise,
    # but assert explicitly to lock the resume-safety invariant.
    assert df.index.is_monotonic_increasing


def test_backfill_drops_sunday_globex_bars(monkeypatch, tmp_path: Path):
    """End-to-end: bars from the Sunday-evening Globex re-open must not
    produce a Sunday shard.

    Reproduces the actual production bug where the 180-day backfill
    wrote ``2026-04-26.parquet`` with 420 bars from 17:00-23:59 CT
    Sunday. After the fix, those bars are filtered out at write time
    and only the Monday RTH bars in the same chunk land on disk.
    """
    chunk_with_sunday = [
        # Sun 2026-04-26 17:00 CT == 22:00 UTC Sunday. Globex re-open.
        _mk_bar("2026-04-26T22:00:00", o=7185.0, h=7191.0, low=7181.25, c=7182.5, vol=1504),
        # Sun 2026-04-26 23:59 CT == Mon 2026-04-27 04:59 UTC.
        _mk_bar("2026-04-27T04:59:00", o=7200.75, h=7200.75, low=7200.5, c=7200.75, vol=27),
        # Mon 2026-04-27 08:30 CT == 13:30 UTC. RTH open, kept.
        _mk_bar("2026-04-27T13:30:00", o=7210.0, h=7211.0, low=7209.0, c=7210.5, vol=200),
        # Mon 2026-04-27 14:59 CT == 19:59 UTC. Last RTH minute, kept.
        _mk_bar("2026-04-27T19:59:00", o=7212.0, h=7213.0, low=7211.5, c=7212.5, vol=180),
    ]
    _patched_adapter(monkeypatch, [chunk_with_sunday])

    out = tmp_path / "es_bars"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backfill",
            "--feed",
            "es-bars",
            "--days",
            "1",
            "--out",
            str(out),
            "--chunk-duration",
            "1 W",
            "--pace-seconds",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    shard_names = sorted(s.name for s in out.glob("*.parquet"))
    # The bug produced "2026-04-26.parquet"; the fix must not.
    assert "2026-04-26.parquet" not in shard_names
    assert shard_names == ["2026-04-27.parquet"]

    df = load_parquet(out / "2026-04-27.parquet")
    assert len(df) == 2  # only the two RTH bars survived the gate


def test_backfill_drops_saturday_ct_bars(monkeypatch, tmp_path: Path):
    """End-to-end: any Saturday-CT bars in the IBKR response must be
    discarded - ES does not trade Saturday CT.
    """
    chunk = [
        # Sat 2026-04-25 04:00 CT == 09:00 UTC Saturday.
        _mk_bar("2026-04-25T09:00:00", o=1.0, h=1.0, low=1.0, c=1.0, vol=1),
        # Sat 2026-04-25 16:59 CT == 21:59 UTC Saturday.
        _mk_bar("2026-04-25T21:59:00", o=2.0, h=2.0, low=2.0, c=2.0, vol=2),
        # Mon 2026-04-27 09:00 CT == 14:00 UTC. RTH, kept.
        _mk_bar("2026-04-27T14:00:00", o=3.0, h=3.0, low=3.0, c=3.0, vol=3),
    ]
    _patched_adapter(monkeypatch, [chunk])

    out = tmp_path / "es_bars"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backfill",
            "--feed",
            "es-bars",
            "--days",
            "1",
            "--out",
            str(out),
            "--chunk-duration",
            "1 W",
            "--pace-seconds",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    shard_names = sorted(s.name for s in out.glob("*.parquet"))
    assert "2026-04-25.parquet" not in shard_names
    assert shard_names == ["2026-04-27.parquet"]


def test_backfill_rejects_unknown_feed(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "backfill",
            "--feed",
            "tick-nyse",
            "--days",
            "1",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    # click rejects unknown choices itself before our handler runs.
    assert result.exit_code != 0
    assert "tick-nyse" in result.output or "Invalid value" in result.output
