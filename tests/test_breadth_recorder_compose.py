# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Compose-file assertions for the breadth recorder service."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"


def _load_compose() -> dict:
    with COMPOSE_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_compose_breadth_recorder_service_uses_real_dockerfile() -> None:
    data = _load_compose()
    recorder = data["services"]["breadth-recorder"]
    build = recorder.get("build")
    assert isinstance(build, dict), f"expected build dict, got {build!r}"
    assert build.get("dockerfile") == "infra/recorders/ibkr_breadth/Dockerfile"
    assert build.get("context") == "."


def test_compose_breadth_recorder_still_profile_gated() -> None:
    data = _load_compose()
    recorder = data["services"]["breadth-recorder"]
    assert "recorder" in (
        recorder.get("profiles") or []
    ), "breadth-recorder must declare profiles: [recorder]"


def test_compose_breadth_recorder_has_volume_mount_for_output() -> None:
    data = _load_compose()
    recorder = data["services"]["breadth-recorder"]
    volumes = recorder.get("volumes") or []
    mounted = [v for v in volumes if isinstance(v, str) and v.endswith(":/data/breadth")]
    assert mounted, f"breadth-recorder must mount a volume at /data/breadth; got {volumes}"
    assert any(
        v.startswith("breadth_parquet:") for v in mounted
    ), f"expected named volume `breadth_parquet`; got {mounted}"

    top_volumes = data.get("volumes", {})
    assert "breadth_parquet" in top_volumes, (
        f"breadth_parquet volume must be declared at the top level; got "
        f"{sorted(top_volumes.keys())}"
    )


def test_compose_breadth_recorder_runs_real_entrypoint() -> None:
    data = _load_compose()
    recorder = data["services"]["breadth-recorder"]
    command = recorder.get("command") or []
    # Accept list or string forms; normalize.
    if isinstance(command, list):
        joined = " ".join(command)
    else:
        joined = str(command)
    assert "infra/recorders/ibkr_breadth/run.py" in joined, (
        "breadth-recorder command must invoke the real run.py; " f"got {command!r}"
    )
