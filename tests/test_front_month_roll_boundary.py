# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Roll-boundary acceptance tests for the consumer-side front-month rebind.

These are the hard part of the fix: a *long-lived* bus consumer (the bars
recorder and the paper-trader), already running, must notice that the
producer has rolled the front month to a new per-contract stream and
re-point its subscription WITHOUT a process restart. The previous
behaviour pinned the expiry at startup and silently read a dead stream
across a quarterly roll (es-bars / nq-bars / paper-trader froze for ~4
trading days at the 2026-06-15 ESM6->ESU6 roll with no error and a
Docker-"healthy" container).

Every test here keeps the component under test running and simulates the
front-month key changing E1->E2 underneath it. Tests that restart the
consumer, or that only inspect stream contents in isolation, do NOT
satisfy the acceptance criteria and are deliberately avoided.

Coverage map (acceptance items a-d + the backstops):

- ``test_recorder_rebinds_across_roll_no_restart`` (ES and NQ):
  (a) re-points to ``bars.<root>.cme.E2`` on the running recorder;
  (b) the boundary minute on disk has EXACTLY ONE bar, the NEW contract's
      price (the rolled-off E1 stream's wrong-price bar never lands);
  (d) no gap beyond one bar;
  + the consumer-front-month gauge moves E1->E2;
  + feed_label stays stable across the roll (series continuity);
  + the NQ case proves no hard-coded 'es' in the switching path.
- ``test_paper_trader_rebinds_across_roll_no_restart_no_replay``:
  (a) the running paper-trader bars loop re-points to E2;
  (c) bar count does NOT reset and buffered E2 bars are NOT replayed
      (start_id='$'); (d) a clean one-bar gap;
  + the paper-trader consumer gauge moves E1->E2.
- ``test_strategy_refuses_to_trade_while_front_month_diverged``:
  a forced divergence raises the divergence state AND the runner refuses
  to submit an order it otherwise would (the trade-refusal backstop),
  with the per-service gauges showing the gap.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from pathlib import Path

import fakeredis
import pandas as pd
import pytest

from alpha_assay.bus.consumer import Consumer
from alpha_assay.bus.producer import Producer
from alpha_assay.bus.streams import bars_stream_name
from alpha_assay.data.front_month import FrontMonthWatcher, write_front_month
from alpha_assay.observability import metrics as M
from infra.recorders.ibkr_es_bars import recorder_metrics as RM
from infra.recorders.ibkr_es_bars.recorder import ESBarsRecorder

# A weekday inside RTH: 2026-05-06 is a Wednesday; 13:30 UTC == 08:30 CDT
# (RTH open). All bar minutes below fall inside the recorder's session mask.
_DAY = "2026-05-06"


def _bar_payload(ts_iso: str, close: float) -> dict:
    secs = pd.Timestamp(ts_iso).value // 10**9
    return {
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 10,
        "ts_minute_utc": secs,
    }


def _publish(producer: Producer, stream: str, ts_iso: str, close: float) -> None:
    producer.publish(stream, _bar_payload(ts_iso, close), ts_event_ns=pd.Timestamp(ts_iso).value)


def _counter(metric, **labels) -> float:
    return metric.labels(**labels)._value.get()


def _gauge(metric, **labels) -> float:
    return metric.labels(**labels)._value.get()


def _wait_until(predicate, timeout: float = 12.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ---------------------------------------------------------------------------
# Recorder: rebind on the running process across a roll (ES + NQ).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,exchange,e1,e2",
    [
        ("ES", "CME", "20260619", "20260918"),
        ("NQ", "CME", "20260619", "20260918"),
    ],
)
def test_recorder_rebinds_across_roll_no_restart(symbol, exchange, e1, e2, tmp_path: Path):
    r = fakeredis.FakeRedis()
    write_front_month(r, symbol=symbol, exchange=exchange, expiry=e1)
    producer = Producer(redis_client=r)
    s1 = bars_stream_name(symbol, exchange, e1)
    s2 = bars_stream_name(symbol, exchange, e2)
    service = f"{symbol.lower()}-bars-recorder"

    # Pre-roll: two correct E1 bars on the E1 stream.
    _publish(producer, s1, f"{_DAY}T13:30:00+00:00", 5000.0)
    _publish(producer, s1, f"{_DAY}T13:31:00+00:00", 5001.0)

    rec = ESBarsRecorder(
        out_dir=tmp_path,
        contract_spec={"symbol": symbol, "sec_type": "FUT", "exchange": exchange, "expiry": e1},
        bus_redis=r,
        bus_consumer_id=service,
        service_label=service,
    )

    # feed_label is the STABLE label (no expiry) - the whole point of the
    # continuity fix. Capture it to assert it does not change across the roll.
    stable_label = rec._feed_label
    assert stable_label == f"{symbol}-FUT", "feed_label must drop the expiry (continuity across rolls)"

    # The running recorder is bound to E1 and consumes the E1 bars.
    asyncio.run(rec.consume_n_messages_for_test(2))
    rec.flush()
    assert rec._stream == s1
    received_pre = _counter(RM.bars_received_total, feed=stable_label)
    # Gauge reflects the bound (E1) expiry from boot.
    assert _gauge(M.consumer_front_month_expiry, service=service, symbol=symbol.lower()) == float(int(e1))

    # ---- THE ROLL (no restart): producer re-resolves; key flips E1 -> E2. ----
    write_front_month(r, symbol=symbol, exchange=exchange, expiry=e2)
    # The race the producer-alias approach could not avoid: the rolled-off E1
    # stream still emits a WRONG-price bar for the boundary minute 13:32.
    _publish(producer, s1, f"{_DAY}T13:32:00+00:00", 9999.0)
    # The producer's NEW E2 stream carries the correct boundary + next bar.
    _publish(producer, s2, f"{_DAY}T13:32:00+00:00", 5002.0)
    _publish(producer, s2, f"{_DAY}T13:33:00+00:00", 5003.0)

    # The SAME running recorder detects the roll and re-points (no new object).
    new_expiry = asyncio.run(rec.check_and_rebind_for_test())

    # (a) re-pointed to bars.<root>.cme.E2 within the running process.
    assert new_expiry == e2
    assert rec._stream == s2
    # Gauge advanced E1 -> E2.
    assert _gauge(M.consumer_front_month_expiry, service=service, symbol=symbol.lower()) == float(int(e2))

    # Consume from the new single-expiry stream. The E1 wrong-price bar is on a
    # stream the recorder no longer reads, so it is never ingested.
    asyncio.run(rec.consume_n_messages_for_test(2))
    rec.flush()

    df = pd.read_parquet(tmp_path / f"{_DAY}.parquet")

    # (b) boundary minute has EXACTLY ONE bar, the NEW contract's price.
    boundary = df[df["timestamp"] == pd.Timestamp(f"{_DAY}T13:32:00+00:00")]
    assert len(boundary) == 1
    assert boundary.iloc[0]["close"] == 5002.0
    assert 9999.0 not in set(df["close"]), "rolled-off E1 wrong-price bar must never land on disk"

    # (d) no gap beyond one bar: 13:30..13:33 contiguous (gap == 0 here).
    assert list(df["timestamp"]) == [pd.Timestamp(f"{_DAY}T13:3{m}:00+00:00") for m in range(4)]
    next_bar = df[df["timestamp"] == pd.Timestamp(f"{_DAY}T13:33:00+00:00")]
    assert len(next_bar) == 1 and next_bar.iloc[0]["close"] == 5003.0

    # feed_label continuity: the stable label is unchanged and the SAME series
    # kept counting across the roll (no per-expiry split / reset).
    assert rec._feed_label == stable_label
    assert _counter(RM.bars_received_total, feed=stable_label) > received_pre


def test_recorder_no_hardcoded_es_in_switching_path():
    """The rebind path derives the root from config, never a literal 'es'.

    Guards against a regression that hard-codes ES in the stream rebuild -
    the NQ recorder instance shares this exact code path.
    """
    import inspect

    src = inspect.getsource(ESBarsRecorder._check_and_rebind)
    lowered = src.lower()
    assert '"es"' not in lowered and "'es'" not in lowered
    assert "bars.es" not in lowered
    # The switch must build the stream from the instance symbol/exchange.
    assert "self._symbol" in src and "self._exchange" in src


def test_recorder_pinned_expiry_never_rebinds(tmp_path: Path):
    """An operator expiry pin freezes the watcher: no auto-rebind."""
    r = fakeredis.FakeRedis()
    write_front_month(r, symbol="ES", exchange="CME", expiry="20260619")
    rec = ESBarsRecorder(
        out_dir=tmp_path,
        contract_spec={"symbol": "ES", "sec_type": "FUT", "exchange": "CME", "expiry": "20260619"},
        bus_redis=r,
        bus_consumer_id="es-bars-recorder",
        front_month_pinned=True,
    )
    write_front_month(r, symbol="ES", exchange="CME", expiry="20260918")
    assert asyncio.run(rec.check_and_rebind_for_test()) is None
    assert rec._stream == bars_stream_name("ES", "CME", "20260619")


# ---------------------------------------------------------------------------
# Paper-trader: rebind on the running consumer loop; no restart, no replay.
# ---------------------------------------------------------------------------


def _load_paper_dryrun():
    name = "paper_dryrun"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent.parent / "scripts" / "paper_dryrun.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _CountingStrategy:
    """Minimal bus-event sink: records every bar close it is fed."""

    def __init__(self) -> None:
        self.bars_seen = 0
        self.ticks_seen = 0
        self.disconnect_count = 0
        self.closes: list[float] = []
        self.consumed_expiry: str | None = None
        self.resolved_expiry: str | None = None

    def on_bar(self, bar, *, feed_label):  # noqa: ARG002 - label unused here
        self.bars_seen += 1
        self.closes.append(float(bar["close"]))

    def update_front_month(self, *, consumed_expiry=None, resolved_expiry=None):
        if consumed_expiry is not None:
            self.consumed_expiry = consumed_expiry
        if resolved_expiry is not None:
            self.resolved_expiry = resolved_expiry


def test_paper_trader_rebinds_across_roll_no_restart_no_replay():
    mod = _load_paper_dryrun()
    e1, e2 = "20260619", "20260918"
    r = fakeredis.FakeRedis()
    write_front_month(r, symbol="ES", exchange="CME", expiry=e1)
    producer = Producer(redis_client=r)
    s1 = bars_stream_name("ES", "CME", e1)
    s2 = bars_stream_name("ES", "CME", e2)

    M.consumer_front_month_expiry.labels(service="paper-trader", symbol="es").set(int(e1))
    strat = _CountingStrategy()
    watcher = FrontMonthWatcher(
        r,
        symbol="ES",
        exchange="CME",
        current_expiry=e1,
    )
    rebinder = mod._BarsRebinder(r, watcher, strat, symbol="ES", exchange="CME", consumer_id="paper-trader-bars")
    # start_id='$' so the strategy never replays buffered history (real config).
    consumer = Consumer(redis_client=r, stream=s1, consumer_id="paper-trader-bars", start_id="$")

    stop = threading.Event()
    th = threading.Thread(
        target=mod._consume_bars_from_bus_sync,
        args=(consumer, strat, stop, rebinder),
        daemon=True,
    )
    th.start()
    try:
        # Let the consumer issue its first XREAD on '$' before publishing.
        time.sleep(0.4)
        _publish(producer, s1, f"{_DAY}T13:30:00+00:00", 5000.0)
        _publish(producer, s1, f"{_DAY}T13:31:00+00:00", 5001.0)
        assert _wait_until(lambda: strat.bars_seen >= 2), "E1 bars not consumed"

        # A bar lands on the NEW stream BEFORE the rebind happens. With
        # start_id='$' the rebind must NOT replay it (the one-bar gap).
        _publish(producer, s2, f"{_DAY}T13:32:00+00:00", 7777.0)

        # THE ROLL: flip the key. The running loop detects it on its next
        # drained-batch poll and re-points to E2 - no restart.
        write_front_month(r, symbol="ES", exchange="CME", expiry=e2)
        assert _wait_until(
            lambda: _gauge(M.consumer_front_month_expiry, service="paper-trader", symbol="es") == float(int(e2))
        ), "paper-trader did not rebind to E2"
        assert strat.consumed_expiry == e2

        # A live E2 bar published AFTER the rebind must be delivered.
        _publish(producer, s2, f"{_DAY}T13:33:00+00:00", 5002.0)
        assert _wait_until(lambda: 5002.0 in strat.closes), "live E2 bar not delivered after rebind"
    finally:
        stop.set()
        th.join(timeout=5)

    # (c) bar count did NOT reset across the rebind: the E1 bars are still
    # counted, plus the post-rebind live bar.
    assert strat.bars_seen >= 3
    assert strat.closes[:2] == [5000.0, 5001.0]
    # (c) no replay: the bar buffered on E2 before the rebind is never seen.
    assert 7777.0 not in strat.closes
    # (d) clean one-bar gap: the strategy's timeline jumps 13:31 -> 13:33, the
    # buffered 13:32 dropped - exactly one bar, never a mixed-contract overlap.
    assert 5002.0 in strat.closes


# ---------------------------------------------------------------------------
# Backstop: the strategy refuses to trade while the front month diverges.
# ---------------------------------------------------------------------------


def test_strategy_refuses_to_trade_while_front_month_diverged():
    # Reuse the runner scaffolding (scripted strategy + fake exec) from the
    # paper-runner unit tests so the only variable is the divergence guard.
    from tests.test_paper_runner import _ct, _feed_minutes, _make_runner

    # Baseline: a signal in-session DOES fire a bracket when consumed ==
    # resolved (no divergence).
    fire = _ct("10:05")
    runner_ok, exec_ok = _make_runner(fire_at={fire: 1})
    assert not runner_ok.front_month_diverged
    _feed_minutes(runner_ok, _ct("10:00"), 8)
    assert len(exec_ok.brackets) == 1, "control: the signal should fire without divergence"

    # Now force a divergence: the producer has rolled (resolved E2) but the
    # consumer is still bound to E1. The identical signal must be REFUSED.
    runner_div, exec_div = _make_runner(fire_at={fire: 1})
    runner_div.update_front_month(consumed_expiry="20260619", resolved_expiry="20260918")
    assert runner_div.front_month_diverged is True

    # The divergence is observable as a gauge gap (consumer behind resolved).
    M.consumer_front_month_expiry.labels(service="paper-trader", symbol="es").set(20260619)
    M.front_month_expiry.labels(symbol="ES", exchange="CME").set(20260918)
    assert _gauge(M.consumer_front_month_expiry, service="paper-trader", symbol="es") != _gauge(
        M.front_month_expiry, symbol="ES", exchange="CME"
    )

    filtered_before = _counter(
        M.signals_filtered_total,
        strategy="ScriptedStrategy",
        filter_name="front_month_diverged",
        reason="stale_contract",
    )
    _feed_minutes(runner_div, _ct("10:00"), 8)

    # REFUSED: no order submitted while diverged, and the refusal is metered.
    assert exec_div.brackets == [], "runner must refuse to trade on a rolled-off contract"
    filtered_after = _counter(
        M.signals_filtered_total,
        strategy="ScriptedStrategy",
        filter_name="front_month_diverged",
        reason="stale_contract",
    )
    assert filtered_after - filtered_before >= 1

    # And once the consumer rebinds (consumed catches up to resolved), trading
    # resumes - the guard is transient, not a latch.
    runner_div.update_front_month(consumed_expiry="20260918")
    assert runner_div.front_month_diverged is False


# ---------------------------------------------------------------------------
# Order-contract rebind: orders must follow the cutover to the live contract.
# ---------------------------------------------------------------------------


def test_runner_repoints_order_contract_on_cutover():
    """After a roll cutover the runner must place orders on the NEW contract.

    Regression for the cross-review finding: the cutover advanced the
    consumed/resolved expiries but left ``_contract_spec`` at the startup
    expiry, so the runner generated signals from new-contract bars yet
    submitted bracket orders against the rolled-off contract.
    """
    from tests.test_paper_runner import _ct, _feed_minutes, _make_runner

    fire = _ct("10:05")
    runner, exec_adapter = _make_runner(fire_at={fire: 1})
    assert runner._contract_spec["expiry"] == "20260618"  # startup contract

    # Roll cutover (flat at roll, pre-open): the order contract re-points too.
    runner.update_front_month(consumed_expiry="20260918", resolved_expiry="20260918")
    assert runner._contract_spec["expiry"] == "20260918"
    assert not runner.front_month_diverged

    # A signal now submits the bracket on the NEW contract - never the old.
    _feed_minutes(runner, _ct("10:00"), 8)
    assert len(exec_adapter.brackets) == 1
    assert exec_adapter.brackets[0]["contract_spec"]["expiry"] == "20260918"
    assert all(b["contract_spec"]["expiry"] != "20260618" for b in exec_adapter.brackets)


def test_runner_defers_contract_repoint_while_position_open_then_applies_on_close():
    """Flat-at-roll invariant: never silently re-point an OPEN position, but
    never latch divergence either.

    The roll is pre-open so the runner is flat; if a position were somehow
    open across a roll, re-pointing the order contract would orphan the
    resting bracket on the old contract. So the cutover is DEFERRED: the
    runner holds the old contract and stays diverged (halting new entries)
    while open, then applies the parked cutover automatically the moment the
    position closes - so divergence can never latch permanently.
    """
    from tests.test_paper_runner import _ct, _feed_minutes, _make_runner

    fire = _ct("10:05")
    runner, exec_adapter = _make_runner(fire_at={fire: 1})
    _feed_minutes(runner, _ct("10:00"), 8)  # opens a position on the E1 contract
    pos = runner.open_position
    assert pos is not None
    assert exec_adapter.brackets[0]["contract_spec"]["expiry"] == "20260618"

    # A cutover arriving while the position is open must NOT re-point now.
    runner.update_front_month(consumed_expiry="20260918", resolved_expiry="20260918")
    assert runner._contract_spec["expiry"] == "20260618"  # held, bracket not orphaned
    assert runner.front_month_diverged  # stays diverged -> new entries halted
    assert runner._pending_consumed_expiry == "20260918"  # cutover parked

    # Close the position (entry fill then target fill). On going flat the
    # deferred cutover applies: the order contract re-points and divergence clears.
    exec_adapter.fire_fill(pos.plan.parent_id, 5000.0, pos.contracts, "2026-06-02 15:06:00+00:00")
    exec_adapter.fire_fill(pos.plan.target_id, 5004.0, pos.contracts, "2026-06-02 15:10:00+00:00")
    assert runner.open_position is None
    assert runner._contract_spec["expiry"] == "20260918"  # deferred cutover applied
    assert runner._pending_consumed_expiry is None
    assert not runner.front_month_diverged  # divergence cleared -> entries resume


def test_round_trip_close_advances_even_if_trade_log_write_raises():
    """Cross-review focus2: a trade-log WRITE failure on the exit fill must NOT
    block in-memory close-state advancement.

    write() only appends to TradeLog's in-memory buffer, so a failure is the
    near-impossible (OOM) case where the record is genuinely unrecoverable. The
    broker-side exit has already closed the round trip, so the position must
    still clear AND any deferred front-month cutover must apply - gating the
    close on the log write would latch the position 'open' forever.
    """
    from tests.test_paper_runner import _ct, _feed_minutes, _make_runner

    class _WriteRaisingTradeLog:
        """Duck-typed trade log whose write always raises (records lost)."""

        def write(self, record):  # noqa: ARG002 - signature match only
            raise OSError("out of memory")

        def flush(self):
            pass

    fire = _ct("10:05")
    runner, exec_adapter = _make_runner(fire_at={fire: 1}, trade_log=_WriteRaisingTradeLog())
    _feed_minutes(runner, _ct("10:00"), 8)  # opens a position on the E1 contract
    pos = runner.open_position
    assert pos is not None

    # Park a deferred cutover while the position is open.
    runner.update_front_month(consumed_expiry="20260918", resolved_expiry="20260918")
    assert runner._pending_consumed_expiry == "20260918"

    # Close via target. The trade-log write raises, but the close must still
    # complete: position cleared, deferred cutover applied, divergence cleared.
    exec_adapter.fire_fill(pos.plan.parent_id, 5000.0, pos.contracts, "2026-06-02 15:06:00+00:00")
    exec_adapter.fire_fill(pos.plan.target_id, 5004.0, pos.contracts, "2026-06-02 15:10:00+00:00")
    assert runner.open_position is None  # close advanced despite log failure
    assert runner._pending_consumed_expiry is None  # deferred cutover applied
    assert runner._contract_spec["expiry"] == "20260918"
    assert not runner.front_month_diverged


def test_round_trip_close_advances_and_record_retained_when_flush_raises(tmp_path: Path):
    """Cross-review confirming pass: a trade-log FLUSH failure (the realistic
    transient I/O case) must advance close-state WITHOUT losing the record.

    write() has already buffered the record, so a flush() failure is
    self-recovering: the whole buffer is rewritten by the next round-trip flush
    or the shutdown flush. This asserts both halves - the close advances AND the
    record stays in the buffer (retained for retry), never silently dropped.
    """
    from alpha_assay.exec.trade_log import TradeLog
    from tests.test_paper_runner import _ct, _feed_minutes, _make_runner

    trade_log = TradeLog(out_dir=tmp_path)

    flush_calls = {"n": 0}

    def _raising_flush():
        flush_calls["n"] += 1
        raise OSError("transient disk error")

    trade_log.flush = _raising_flush  # type: ignore[method-assign]

    fire = _ct("10:05")
    runner, exec_adapter = _make_runner(fire_at={fire: 1}, trade_log=trade_log)
    _feed_minutes(runner, _ct("10:00"), 8)
    pos = runner.open_position
    assert pos is not None

    # Close via target. flush() raises, but write() succeeded (record buffered).
    exec_adapter.fire_fill(pos.plan.parent_id, 5000.0, pos.contracts, "2026-06-02 15:06:00+00:00")
    exec_adapter.fire_fill(pos.plan.target_id, 5004.0, pos.contracts, "2026-06-02 15:10:00+00:00")

    assert runner.open_position is None  # close advanced despite flush failure
    assert flush_calls["n"] >= 1  # flush was attempted
    # The record is RETAINED in the buffer (not dropped): a later flush recovers it.
    assert len(trade_log._buffer) == 1
    # Proof of recovery: invoking the real (unbound) flush persists the retained
    # record - exactly what the next round-trip flush or shutdown flush does.
    TradeLog.flush(trade_log)
    persisted = pd.read_parquet(tmp_path / "trades.parquet")
    assert len(persisted) == 1


def test_front_month_diverged_reads_under_lock():
    """Cross-review focus1: front_month_diverged must read the
    consumed/resolved pair under self._lock so an off-thread ops/health caller
    can never observe a torn (impossible) divergence state.

    A torn-read race is too narrow to catch probabilistically in CPython, so
    this asserts the property deterministically: it must acquire self._lock to
    read. The lock is swapped for an enter-counting wrapper around the real
    RLock; reading the property must enter it at least once.
    """
    from tests.test_paper_runner import _make_runner

    class _CountingLock:
        """Delegates to the real RLock but counts context-manager entries."""

        def __init__(self, inner):
            self._inner = inner
            self.enters = 0

        def __enter__(self):
            self.enters += 1
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def acquire(self, *a, **k):
            return self._inner.acquire(*a, **k)

        def release(self):
            return self._inner.release()

    runner, _ = _make_runner()
    counting = _CountingLock(runner._lock)
    runner._lock = counting

    _ = runner.front_month_diverged
    assert counting.enters >= 1, "front_month_diverged must read under self._lock"
