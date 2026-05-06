# tests/test_feed_manifest.py
from __future__ import annotations

from pathlib import Path

import pytest

from infra.feed.feed import FeedManifest, ManifestError


def test_parse_default_manifest(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("""
subscriptions:
  - kind: bars
    contract:
      symbol: ES
      sec_type: FUT
      exchange: CME
      currency: USD
      expiry: "20260618"
    bar_size: "1 min"
    what_to_show: TRADES
  - kind: ticks
    symbol: TICK-NYSE
""")
    m = FeedManifest.from_yaml(p)
    assert len(m.subscriptions) == 2
    bars_sub = m.subscriptions[0]
    assert bars_sub.kind == "bars"
    assert bars_sub.contract["symbol"] == "ES"
    assert bars_sub.bar_size == "1 min"
    ticks_sub = m.subscriptions[1]
    assert ticks_sub.kind == "ticks"
    assert ticks_sub.symbol == "TICK-NYSE"


def test_unknown_kind_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("subscriptions: [{kind: candles, symbol: X}]")
    with pytest.raises(ManifestError, match="kind"):
        FeedManifest.from_yaml(p)
