# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Compose-file + prometheus assertions for the ES-bars recorder."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
PROMETHEUS_PATH = REPO_ROOT / "observability" / "prometheus.yml"


def _load_compose() -> dict:
    with COMPOSE_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prometheus() -> dict:
    with PROMETHEUS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_compose_es_bars_recorder_uses_real_dockerfile() -> None:
    data = _load_compose()
    rec = data["services"]["es-bars-recorder"]
    build = rec.get("build")
    assert isinstance(build, dict), f"expected build dict, got {build!r}"
    assert build.get("dockerfile") == "infra/recorders/ibkr_es_bars/Dockerfile"
    assert build.get("context") == "."


def test_compose_es_bars_recorder_profile_gated() -> None:
    data = _load_compose()
    rec = data["services"]["es-bars-recorder"]
    assert "recorder" in (rec.get("profiles") or []), "es-bars-recorder must declare profiles: [recorder]"


def test_compose_es_bars_recorder_has_volume_mount_for_output() -> None:
    data = _load_compose()
    rec = data["services"]["es-bars-recorder"]
    volumes = rec.get("volumes") or []
    mounted = [v for v in volumes if isinstance(v, str) and v.endswith(":/data/es_bars")]
    assert mounted, f"es-bars-recorder must mount a volume at /data/es_bars; got {volumes}"
    assert any(
        v.startswith("es_bars_parquet:") for v in mounted
    ), f"expected named volume `es_bars_parquet`; got {mounted}"

    top_volumes = data.get("volumes", {})
    assert "es_bars_parquet" in top_volumes, (
        f"es_bars_parquet volume must be declared at top level; got " f"{sorted(top_volumes.keys())}"
    )
    # The named volume must use the alphaassay_ prefix to keep the
    # the deployment host host's docker volume namespace distinct from sibling stacks.
    assert (
        top_volumes["es_bars_parquet"].get("name") == "alphaassay_es_bars_parquet"
    ), f"expected alphaassay_es_bars_parquet; got {top_volumes['es_bars_parquet']}"


def test_compose_es_bars_recorder_runs_real_entrypoint() -> None:
    data = _load_compose()
    rec = data["services"]["es-bars-recorder"]
    command = rec.get("command") or []
    if isinstance(command, list):
        joined = " ".join(command)
    else:
        joined = str(command)
    assert (
        "infra/recorders/ibkr_es_bars/run.py" in joined
    ), f"es-bars-recorder command must invoke the real run.py; got {command!r}"


def test_compose_es_bars_recorder_has_unique_metrics_port() -> None:
    """must NOT collide with paper-trader (8000) or breadth (8001)."""
    data = _load_compose()
    rec = data["services"]["es-bars-recorder"]
    env = rec.get("environment") or {}
    assert env.get("METRICS_PORT") == "8002", (
        f"es-bars-recorder METRICS_PORT must be 8002 (paper-trader=8000, "
        f"breadth-recorder=8001); got {env.get('METRICS_PORT')!r}"
    )

    ports = rec.get("ports") or []
    # Loopback-only host binding on 18002.
    assert any(
        isinstance(p, str) and p == "127.0.0.1:18002:8002" for p in ports
    ), f"expected host port 127.0.0.1:18002:8002; got {ports}"


def test_prometheus_scrape_includes_es_bars_recorder() -> None:
    data = _load_prometheus()
    jobs = {job["job_name"]: job for job in data["scrape_configs"]}
    assert "es-bars-recorder" in jobs, f"prometheus scrape jobs missing es-bars-recorder; got {sorted(jobs)}"
    job = jobs["es-bars-recorder"]
    targets: list[str] = []
    for sc in job.get("static_configs", []):
        targets.extend(sc.get("targets", []))
    assert "host.docker.internal:8002" in targets, (
        f"es-bars-recorder job must target host.docker.internal:8002 so the scrape "
        f"works whether the container is on the bridge network or network_mode: host; "
        f"got {targets}"
    )
