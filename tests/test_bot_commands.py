# tests/test_bot_commands.py
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from infra.bot.commands import (
    BotContext,
    cmd_feed,
    cmd_health,
    cmd_logs,
    cmd_restart,
    cmd_status,
)
from infra.bot.dispatch import Command


def _ctx(**overrides):
    base = {
        "docker_client": MagicMock(),
        "healthcheck_script": "/bin/true",
        "prom_url": "http://prometheus:9090",
        "feed_pause_redis": MagicMock(),
        "restartable_services": ["es-bars-recorder", "ibkr-feed"],
    }
    base.update(overrides)
    return BotContext(**base)


def test_health_runs_script_and_returns_summary(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="PASS: 22 WARN: 0 FAIL: 0\nRESULT: READY", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = cmd_health(Command(name="health", args=[]), _ctx())
    assert "READY" in out
    assert "22" in out


def test_status_lists_alphaassay_containers():
    docker_client = MagicMock()
    container = MagicMock()
    container.name = "alphaassay-redis"
    container.attrs = {"State": {"Status": "running", "Health": {"Status": "healthy"}}}
    docker_client.containers.list.return_value = [container]
    out = cmd_status(Command(name="status", args=[]), _ctx(docker_client=docker_client))
    assert "alphaassay-redis" in out


def test_logs_default_n_50():
    docker_client = MagicMock()
    container = MagicMock()
    container.logs.return_value = b"\n".join(f"line-{i}".encode() for i in range(50))
    docker_client.containers.get.return_value = container
    cmd_logs(Command(name="logs", args=["es-bars-recorder"]), _ctx(docker_client=docker_client))
    container.logs.assert_called_once()
    _, kwargs = container.logs.call_args
    assert kwargs["tail"] == 50


def test_logs_custom_n():
    docker_client = MagicMock()
    container = MagicMock()
    container.logs.return_value = b"x"
    docker_client.containers.get.return_value = container
    cmd_logs(Command(name="logs", args=["es-bars-recorder", "200"]), _ctx(docker_client=docker_client))
    _, kwargs = container.logs.call_args
    assert kwargs["tail"] == 200


def test_logs_usage_when_no_service():
    out = cmd_logs(Command(name="logs", args=[]), _ctx())
    assert "usage" in out.lower()


def test_restart_rejects_non_allowlisted_service():
    out = cmd_restart(Command(name="restart", args=["postgres"]), _ctx())
    assert "not allowed" in out.lower()


def test_restart_calls_docker_for_allowlisted():
    docker_client = MagicMock()
    container = MagicMock()
    docker_client.containers.get.return_value = container
    out = cmd_restart(Command(name="restart", args=["es-bars-recorder"]), _ctx(docker_client=docker_client))
    container.restart.assert_called_once()
    assert "restarted" in out.lower()


def test_feed_pause_sets_redis_flag():
    redis_mock = MagicMock()
    cmd_feed(Command(name="feed", args=["pause"]), _ctx(feed_pause_redis=redis_mock))
    redis_mock.set.assert_called_with("alpha_assay:feed_paused", "1")


def test_feed_resume_clears_redis_flag():
    redis_mock = MagicMock()
    cmd_feed(Command(name="feed", args=["resume"]), _ctx(feed_pause_redis=redis_mock))
    redis_mock.delete.assert_called_with("alpha_assay:feed_paused")


def test_feed_unknown_subcommand():
    out = cmd_feed(Command(name="feed", args=["explode"]), _ctx())
    assert "usage" in out.lower()
