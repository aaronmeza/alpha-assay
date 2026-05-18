# SPDX-License-Identifier: Apache-2.0
"""Telegram bot command handlers for alpha-assay operator interface."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from infra.bot.dispatch import Command


@dataclass(frozen=True)
class BotContext:
    """Per-instance dependencies passed to every command handler."""

    docker_client: Any
    healthcheck_script: str
    prom_url: str
    restartable_services: list[str]


def cmd_health(cmd: Command, ctx: BotContext) -> str:
    """Run the configured healthcheck script and return its summary."""
    proc = subprocess.run(
        [ctx.healthcheck_script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    summary_lines = [ln for ln in output.splitlines() if "PASS:" in ln or "RESULT:" in ln]
    if summary_lines:
        return "\n".join(summary_lines)
    return output[-500:]


def cmd_status(cmd: Command, ctx: BotContext) -> str:
    """Return one-line summary per alphaassay-* container."""
    containers = ctx.docker_client.containers.list(filters={"name": "alphaassay-"}, all=True)
    if not containers:
        return "no alphaassay-* containers found"
    lines = []
    for c in containers:
        state = c.attrs.get("State", {})
        status = state.get("Status", "unknown")
        health = state.get("Health", {}).get("Status", "n/a")
        lines.append(f"{c.name}: {status} ({health})")
    return "\n".join(lines)


def cmd_logs(cmd: Command, ctx: BotContext) -> str:
    """Tail logs from a container. Default 50 lines."""
    if not cmd.args:
        return "usage: /logs <service> [n]"
    service = cmd.args[0]
    n = 50
    if len(cmd.args) >= 2:
        try:
            n = int(cmd.args[1])
        except ValueError:
            return f"invalid n: {cmd.args[1]!r}"
    name = service if service.startswith("alphaassay-") else f"alphaassay-{service}"
    try:
        container = ctx.docker_client.containers.get(name)
    except Exception as e:
        return f"container not found: {name} ({e})"
    raw = container.logs(tail=n)
    text = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
    return text[-3500:]


def cmd_restart(cmd: Command, ctx: BotContext) -> str:
    """Restart a service from the allowlist."""
    if not cmd.args:
        return "usage: /restart <service>"
    service = cmd.args[0]
    if service not in ctx.restartable_services:
        return f"service {service!r} not allowed (allowlist: {ctx.restartable_services})"
    name = service if service.startswith("alphaassay-") else f"alphaassay-{service}"
    try:
        container = ctx.docker_client.containers.get(name)
    except Exception as e:
        return f"container not found: {name} ({e})"
    container.restart()
    return f"{name} restarted"
