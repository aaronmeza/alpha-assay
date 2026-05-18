# SPDX-License-Identifier: Apache-2.0
"""Telegram bot entrypoint: getUpdates poll + dispatch.

Destructive commands (those marked destructive=True in the registry)
require an explicit "yes" reply within 60s before action runs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import docker as docker_pkg

from infra.bot.commands import (
    BotContext,
    cmd_health,
    cmd_logs,
    cmd_restart,
    cmd_status,
)
from infra.bot.dispatch import (
    Command,
    CommandRegistry,
    UnauthorizedError,
    UnknownCommandError,
    parse_command_text,
)

LOG = logging.getLogger("alpha_assay.bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

CONFIRM_WINDOW_SECONDS = 60


@dataclass
class _PendingConfirm:
    cmd: Command
    chat_id: int
    expires_at: float


def _telegram_get_updates(token: str, offset: int) -> list[dict]:
    url = f"https://api.telegram.org/bot{token}/getUpdates?{urlencode({'offset': offset, 'timeout': 30})}"
    req = Request(url)
    with urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    return body.get("result", []) if body.get("ok") else []


def _telegram_send(token: str, chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urlencode({"chat_id": chat_id, "text": text}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req, timeout=10) as r:
        r.read()


def _load_offset(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return 0


def _save_offset(path: Path, offset: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(str(offset))
    tmp.replace(path)


def main() -> int:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    allowed = {int(x) for x in os.environ["ALLOWED_CHAT_IDS"].split(",") if x.strip()}
    restartable = [s.strip() for s in os.environ.get("RESTARTABLE_SERVICES", "").split(",") if s.strip()]
    healthcheck_script = os.environ["HEALTHCHECK_SCRIPT"]
    prom_url = os.environ.get("PROM_URL", "http://prometheus:9090")
    offset_path = Path(os.environ.get("OFFSET_PATH", "/var/lib/alphaassay/bot/offset"))
    offset_path.parent.mkdir(parents=True, exist_ok=True)

    docker_client = docker_pkg.from_env()

    ctx = BotContext(
        docker_client=docker_client,
        healthcheck_script=healthcheck_script,
        prom_url=prom_url,
        restartable_services=restartable,
    )

    registry = CommandRegistry(allowed_chat_ids=allowed)
    registry.register("health", cmd_health)
    registry.register("status", cmd_status)
    registry.register("logs", cmd_logs)
    registry.register("restart", cmd_restart, destructive=True)

    pending: dict[int, _PendingConfirm] = {}

    LOG.info("bot starting: allowed_chat_ids=%s, restartable=%s", allowed, restartable)
    offset = _load_offset(offset_path)

    while True:
        try:
            updates = _telegram_get_updates(token, offset)
        except Exception as e:
            LOG.warning("getUpdates failed: %s", e)
            time.sleep(5)
            continue

        for upd in updates:
            offset = upd["update_id"] + 1
            msg = upd.get("message")
            if not msg:
                continue
            chat_id = msg["chat"]["id"]
            text = msg.get("text", "")

            if chat_id not in allowed:
                LOG.warning("ignoring message from unauthorized chat_id=%s", chat_id)
                continue

            confirm = pending.pop(chat_id, None)
            if confirm and time.time() <= confirm.expires_at and text.strip().lower() in ("yes", "y"):
                try:
                    out = registry.dispatch(confirm.cmd, chat_id=chat_id, ctx=ctx)
                except Exception as e:
                    out = f"error: {e}"
                _telegram_send(token, chat_id, out)
                continue

            if not text.startswith("/"):
                continue

            try:
                cmd = parse_command_text(text)
            except ValueError as e:
                _telegram_send(token, chat_id, f"parse error: {e}")
                continue

            if registry.is_destructive(cmd.name):
                pending[chat_id] = _PendingConfirm(
                    cmd=cmd, chat_id=chat_id, expires_at=time.time() + CONFIRM_WINDOW_SECONDS
                )
                _telegram_send(
                    token,
                    chat_id,
                    f"confirm: /{cmd.name} {' '.join(cmd.args)} - reply 'yes' within 60s to proceed",
                )
                continue

            try:
                out = registry.dispatch(cmd, chat_id=chat_id, ctx=ctx)
            except (UnauthorizedError, UnknownCommandError) as e:
                out = f"error: {e}"
            except Exception as e:
                LOG.exception("handler crashed for %s", cmd)
                out = f"handler error: {e}"
            _telegram_send(token, chat_id, out)

        _save_offset(offset_path, offset)


if __name__ == "__main__":
    sys.exit(main() or 0)
