# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""compose-stack YAML-parse tests.

No docker CLI is invoked. Tests only parse the YAML and JSON shipped in
the repo, then cross-check against the metrics catalog imported from
`alpha_assay.observability.metrics`. the deployment host deploy validation is separate.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from prometheus_client import Counter, Gauge, Histogram

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
PROMETHEUS_PATH = REPO_ROOT / "observability" / "prometheus.yml"
DASHBOARD_PATH = REPO_ROOT / "observability" / "grafana" / "dashboards" / "alphaassay.json"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_compose_yaml_is_valid() -> None:
    data = _load_yaml(COMPOSE_PATH)
    assert isinstance(data, dict)

    services = data.get("services", {})
    expected = {"paper-trader", "breadth-recorder", "prometheus", "grafana"}
    assert expected.issubset(services.keys()), f"missing services; found {sorted(services.keys())}"

    # Network must be named and isolated from the factory stack.
    networks = data.get("networks", {})
    assert (
        "alphaassay_observability" in networks
    ), f"expected alphaassay_observability network; got {sorted(networks.keys())}"

    # Volumes required for datastore persistence.
    volumes = data.get("volumes", {})
    for required in ("prometheus_data", "grafana_data"):
        assert required in volumes, f"expected {required} volume; got {sorted(volumes.keys())}"

    # Host-port bindings: non-default + 127.0.0.1 only (loopback-only).
    expected_host_ports = {"13000", "19090", "18000", "18001"}
    seen_host_ports: set[str] = set()
    for svc_name, svc in services.items():
        for port in svc.get("ports", []):
            assert isinstance(port, str), f"service {svc_name} port entry must be long-form string, got {port!r}"
            assert "0.0.0.0" not in port, f"service {svc_name} binds to 0.0.0.0 ({port!r}); must bind 127.0.0.1 only"
            assert port.startswith(
                "127.0.0.1:"
            ), f"service {svc_name} port {port!r} must bind 127.0.0.1 (loopback only)"
            host_port = port.split(":")[1]
            seen_host_ports.add(host_port)
            assert host_port not in {
                "3000",
                "9090",
                "8000",
                "8001",
            }, f"service {svc_name} uses default host port {host_port}; must be non-default"

    assert expected_host_ports.issubset(
        seen_host_ports
    ), f"expected host ports {expected_host_ports}; saw {seen_host_ports}"

    # Container names prefixed alphaassay- to avoid collision with factory stack.
    for svc_name, svc in services.items():
        cname = svc.get("container_name")
        assert cname and cname.startswith(
            "alphaassay-"
        ), f"service {svc_name} container_name must start with alphaassay-; got {cname!r}"

    # Profile-gated breadth-recorder.
    recorder = services["breadth-recorder"]
    assert "recorder" in (
        recorder.get("profiles") or []
    ), "breadth-recorder must declare profiles: [recorder] so default `up` skips it"

    # Restart policies per spec.
    assert services["paper-trader"].get("restart") == "on-failure:3"
    assert services["breadth-recorder"].get("restart") == "on-failure:3"
    assert services["prometheus"].get("restart") == "unless-stopped"
    assert services["grafana"].get("restart") == "unless-stopped"

    # Healthchecks declared for every service.
    for svc_name, svc in services.items():
        assert "healthcheck" in svc, f"service {svc_name} missing healthcheck"

    # paper-trader environment must set METRICS_PORT, ALPHA_ASSAY_ENV, TZ.
    pt_env = services["paper-trader"].get("environment", {})
    # docker compose allows dict or list; normalize.
    if isinstance(pt_env, list):
        pt_env = dict(item.split("=", 1) for item in pt_env)
    assert pt_env.get("METRICS_PORT") == "8000"
    assert pt_env.get("ALPHA_ASSAY_ENV") == "paper-dryrun"
    assert pt_env.get("TZ") == "America/Chicago"


def test_prometheus_scrape_targets() -> None:
    data = _load_yaml(PROMETHEUS_PATH)
    assert isinstance(data, dict)

    scrape_configs = data.get("scrape_configs", [])
    job_names = {job.get("job_name") for job in scrape_configs}
    assert {"paper-trader", "breadth-recorder"}.issubset(job_names), f"missing scrape jobs; got {job_names}"

    for job in scrape_configs:
        if job["job_name"] == "paper-trader":
            targets = job["static_configs"][0]["targets"]
            assert "host.docker.internal:8000" in targets, (
                f"paper-trader job must target host.docker.internal:8000 so the scrape "
                f"works whether the container is on the bridge network or network_mode: host; "
                f"got {targets}"
            )
        elif job["job_name"] == "breadth-recorder":
            targets = job["static_configs"][0]["targets"]
            assert "host.docker.internal:8001" in targets, (
                f"breadth-recorder job must target host.docker.internal:8001; got {targets}"
            )


def test_grafana_dashboard_json_is_valid() -> None:
    with DASHBOARD_PATH.open("r", encoding="utf-8") as f:
        dashboard = json.load(f)

    assert "title" in dashboard and dashboard["title"], "dashboard must have a title"

    # Collect all PromQL expressions from all panels (including nested).
    expressions: list[str] = []

    def _collect(panels: list) -> None:
        for panel in panels:
            for target in panel.get("targets", []):
                expr = target.get("expr")
                if expr:
                    expressions.append(expr)
            # Grafana row panels nest children under "panels".
            if panel.get("panels"):
                _collect(panel["panels"])

    _collect(dashboard.get("panels", []))
    assert expressions, "dashboard must define at least one PromQL target"

    combined_expr = "\n".join(expressions)

    # Enumerate the metric catalog by importing the module.
    from alpha_assay.observability import metrics as M

    metric_names: list[str] = []
    for attr in dir(M):
        obj = getattr(M, attr)
        if isinstance(obj, (Counter, Gauge, Histogram)):
            # prometheus_client counters auto-suffix _total; use the stored _name.
            metric_names.append(obj._name)

    missing = [name for name in metric_names if name not in combined_expr]
    assert not missing, f"dashboard missing PromQL references for metrics: {missing}"


def test_paper_trader_stub_heartbeat_increments() -> None:
    """Import the stub's heartbeat callable and assert the counter ticks."""
    import importlib.util

    stub_path = REPO_ROOT / "scripts" / "paper_trader_stub.py"
    spec = importlib.util.spec_from_file_location("paper_trader_stub", stub_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from alpha_assay.observability import metrics as M

    # Find the stub-labeled counter child.
    before = M.bars_processed_total.labels(feed="stub")._value.get()
    module.heartbeat_once()
    after = M.bars_processed_total.labels(feed="stub")._value.get()
    assert after == before + 1, f"heartbeat should increment counter by 1 (before={before}, after={after})"
