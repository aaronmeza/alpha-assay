# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""end-to-end live-IBKR paper dry-run test.

OPT-IN ONLY. This module is skipped unless ``RUN_LIVE_E2E=1`` is set
in the environment. It launches ``scripts/paper_dryrun.py`` as a
subprocess against a real IB Gateway / TWS paper account, scrapes the
``/metrics`` endpoint, and asserts:

- ``alpha_assay_bars_processed_total{feed="es"}`` > 0
- ``alpha_assay_bars_processed_total{feed="tick_nyse"}`` > 0
- ``alpha_assay_orders_submitted_total`` is 0 across all ``type`` labels
  (always-flat invariant)
- ``alpha_assay_ibkr_feed_freshness_seconds`` < 30 for both feeds

Operational checklist before running:

- IB Gateway (paper) running on the host / local host, listening on 4002.
- IBKR paper market-data subs active (CME futures bundle + NYSE breadth).
- ``.env.secrets`` populated with ``IBKR_HOST/PORT/CLIENT_ID/ACCOUNT``.
- The test runs during US RTH (Mon-Fri 09:30-16:00 ET); outside RTH the
  freshness gauge will climb above 30s and the test will fail in ways
  that are NOT bugs.
- The ``METRICS_PORT`` exposed by the script under test is 18000 to
  match the docker-compose binding; do not change without updating
  ``docker-compose.yml`` and ``observability/prometheus.yml``.

The subprocess uses the project's own venv interpreter
(the project venv interpreter) so the
test does NOT pip-install at run time.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "paper_dryrun.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

E2E_PORT = 18000
E2E_PORT_STR = str(E2E_PORT)
RUN_DURATION_SECONDS = 45
DRAIN_TIMEOUT_SECONDS = 20


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_E2E") != "1",
    reason="set RUN_LIVE_E2E=1 to run live IBKR paper dry-run",
)


def _interpreter() -> str:
    """Prefer the repo venv's Python; fall back to the current
    interpreter only if the venv has not been bootstrapped (CI on a
    fresh checkout)."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _scrape_metrics(port: int, timeout: float = 5.0) -> str:
    url = f"http://127.0.0.1:{port}/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _parse_metric_samples(body: str, metric_name: str) -> list[tuple[dict[str, str], float]]:
    """Best-effort Prometheus exposition parser for the metrics this
    test needs. We avoid importing ``prometheus_client.parser`` because
    we want to assert on the wire format the operator will see when
    they run ``curl /metrics`` themselves.
    """
    samples: list[tuple[dict[str, str], float]] = []
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(metric_name):
            continue
        # Accept either ``name value`` or ``name{labels} value``.
        rest = line[len(metric_name) :]
        labels: dict[str, str] = {}
        if rest.startswith("{"):
            close = rest.index("}")
            labels_str = rest[1:close]
            rest = rest[close + 1 :]
            for kv in labels_str.split(","):
                if not kv:
                    continue
                k, _, v = kv.partition("=")
                labels[k.strip()] = v.strip().strip('"')
        rest = rest.strip()
        if not rest:
            continue
        value_str = rest.split()[0]
        try:
            samples.append((labels, float(value_str)))
        except ValueError:
            continue
    return samples


def _start_subprocess(
    *,
    duration_seconds: int,
    metrics_port: int,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["DRYRUN_DURATION_SECONDS"] = str(duration_seconds)
    env["METRICS_PORT"] = str(metrics_port)
    # Force the documented IB Gateway paper port if the operator has not
    # already set IBKR_PORT. The test still respects an explicit override.
    env.setdefault("IBKR_PORT", "4002")
    env.setdefault("IBKR_HOST", "127.0.0.1")
    env.setdefault("IBKR_CLIENT_ID", "1")
    env.setdefault("ES_EXPIRY", "20260618")
    return subprocess.Popen(
        [_interpreter(), "-u", str(SCRIPT_PATH)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _drain_or_kill(proc: subprocess.Popen[bytes]) -> int:
    """SIGTERM, then wait up to DRAIN_TIMEOUT_SECONDS, escalate to
    SIGKILL if the script does not drain. Returns the exit code or -9
    if killed.
    """
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=DRAIN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        return -9
    return proc.returncode


# ----------------------------------------------------------------------


def test_dryrun_subscribes_and_emits_metrics():
    """Run the dry-run for ~45s, scrape /metrics, assert per-feed bar
    counts climb and orders_submitted stays zero."""
    proc = _start_subprocess(duration_seconds=RUN_DURATION_SECONDS, metrics_port=E2E_PORT)
    try:
        # Allow the subprocess to bind /metrics + connect to IBKR.
        deadline = time.monotonic() + 25
        body = ""
        while time.monotonic() < deadline:
            try:
                body = _scrape_metrics(E2E_PORT)
                if "alpha_assay_bars_processed_total" in body:
                    break
            except (urllib.error.URLError, ConnectionRefusedError):
                pass
            time.sleep(1)
        assert body, "failed to scrape /metrics within startup window"

        # Wait for at least one bar per feed; bars take >= 1 minute to
        # roll for ES, but breadth ticks fire continuously during RTH.
        bar_deadline = time.monotonic() + RUN_DURATION_SECONDS
        es_count = 0.0
        tick_count = 0.0
        while time.monotonic() < bar_deadline:
            body = _scrape_metrics(E2E_PORT)
            samples = _parse_metric_samples(body, "alpha_assay_bars_processed_total")
            es_count = sum(v for labels, v in samples if labels.get("feed") == "es")
            tick_count = sum(v for labels, v in samples if labels.get("feed") == "tick_nyse")
            if es_count > 0 and tick_count > 0:
                break
            time.sleep(2)

        assert es_count > 0, (
            f"alpha_assay_bars_processed_total{{feed='es'}} stayed 0 over {RUN_DURATION_SECONDS}s; "
            f"check IBKR sub for ES futures bars and that the test runs during RTH"
        )
        assert tick_count > 0, (
            f"alpha_assay_bars_processed_total{{feed='tick_nyse'}} stayed 0 over "
            f"{RUN_DURATION_SECONDS}s; check NYSE breadth subscription"
        )

        order_samples = _parse_metric_samples(body, "alpha_assay_orders_submitted_total")
        assert all(v == 0 for _labels, v in order_samples), (
            "orders_submitted_total must be 0 across all type labels in always-flat dry-run; " f"got {order_samples}"
        )

        freshness_samples = _parse_metric_samples(body, "alpha_assay_ibkr_feed_freshness_seconds")
        seen_es = False
        seen_tick = False
        for labels, value in freshness_samples:
            feed = labels.get("feed", "")
            if feed.startswith("ES"):
                seen_es = True
                assert value < 30.0, f"ES feed freshness {value}s >= 30s; market may be closed or feed stalled"
            elif feed == "TICK-NYSE":
                seen_tick = True
                assert value < 30.0, f"TICK-NYSE freshness {value}s >= 30s; check NYSE breadth feed"
        assert seen_es, "no ES freshness gauge sample observed"
        assert seen_tick, "no TICK-NYSE freshness gauge sample observed"
    finally:
        rc = _drain_or_kill(proc)
        assert rc != -9, (
            "paper_dryrun.py failed to drain within DRAIN_TIMEOUT_SECONDS; " "graceful-shutdown contract is broken"
        )


def test_dryrun_exits_cleanly_on_sigterm():
    """Send SIGTERM and require exit code 0 within the drain window."""
    proc = _start_subprocess(duration_seconds=0, metrics_port=E2E_PORT + 1)
    try:
        # Give the process time to come up and bind /metrics.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                _scrape_metrics(E2E_PORT + 1)
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(1)
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=DRAIN_TIMEOUT_SECONDS)
        assert rc == 0, f"expected clean exit on SIGTERM, got {rc}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_dryrun_metrics_endpoint_responds():
    """Basic /metrics HTTP 200 sanity."""
    proc = _start_subprocess(duration_seconds=0, metrics_port=E2E_PORT + 2)
    try:
        deadline = time.monotonic() + 15
        body = ""
        while time.monotonic() < deadline:
            try:
                body = _scrape_metrics(E2E_PORT + 2)
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(1)
        assert "alpha_assay_" in body, "metrics body missing alpha_assay_ prefix"
    finally:
        rc = _drain_or_kill(proc)
        assert rc != -9, "drain failure on metrics-only smoke test"
