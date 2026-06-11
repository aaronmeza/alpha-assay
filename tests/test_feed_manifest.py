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


def test_ticks_defaults_are_back_compat(tmp_path: Path):
    """Manifests that predate multi-venue breadth parse to NYSE/USD/required."""
    p = tmp_path / "m.yaml"
    p.write_text("subscriptions: [{kind: ticks, symbol: TICK-NYSE}]")
    sub = FeedManifest.from_yaml(p).subscriptions[0]
    assert sub.exchange == "NYSE"
    assert sub.currency == "USD"
    assert sub.required is True
    assert sub.stream == "ticks.tick-nyse"


def test_ticks_exchange_and_required_parse(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("""
subscriptions:
  - kind: ticks
    symbol: TICK-NASD
    exchange: NASDAQ
    required: false
  - kind: ticks
    symbol: VIX
    exchange: CBOE
    required: false
  - kind: bars
    required: false
    contract:
      symbol: NQ
      sec_type: FUT
      exchange: CME
      currency: USD
""")
    subs = FeedManifest.from_yaml(p).subscriptions
    tick_nasd, vix, nq = subs
    assert tick_nasd.exchange == "NASDAQ" and tick_nasd.required is False
    assert vix.exchange == "CBOE" and vix.stream == "ticks.vix"
    assert nq.kind == "bars" and nq.required is False
    # No expiry yet (resolved at startup) -> stream has no expiry segment.
    assert nq.stream == "bars.nq.cme"


def test_extended_manifest_in_repo_parses(tmp_path: Path):
    """configs/feed-manifest-extended.yaml stays parseable and keeps the
    core feeds required + the new feeds optional."""
    repo_root = Path(__file__).resolve().parent.parent
    m = FeedManifest.from_yaml(repo_root / "configs" / "feed-manifest-extended.yaml")
    by_stream = {s.stream: s for s in m.subscriptions}
    # Core set unchanged and required.
    assert by_stream["ticks.tick-nyse"].required is True
    assert by_stream["ticks.ad-nyse"].required is True
    assert by_stream["bars.es.cme.20260618"].required is True
    # New feeds present and optional.
    assert by_stream["ticks.tick-nasd"].required is False
    assert by_stream["ticks.tick-nasd"].exchange == "NASDAQ"
    assert by_stream["ticks.ad-nasd"].required is False
    assert by_stream["ticks.vix"].exchange == "CBOE"
    assert by_stream["bars.nq.cme"].required is False
