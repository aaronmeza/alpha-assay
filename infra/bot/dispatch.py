# SPDX-License-Identifier: Apache-2.0
"""Telegram bot command parsing + dispatch.

Security model:
- Only chat IDs in ``allowed_chat_ids`` get responses; everything else
  is logged and ignored (don't leak existence to non-authorized chats).
- Commands are dispatched to typed handlers - no shell string
  interpolation anywhere.
- Destructive commands (registered via ``register(..., destructive=True)``)
  trigger a confirmation flow, not handled here (see main.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Command:
    name: str
    args: list[str] = field(default_factory=list)


class UnauthorizedError(PermissionError):
    pass


class UnknownCommandError(KeyError):
    pass


@dataclass
class _Handler:
    fn: Callable[[Command, Any], str]
    destructive: bool


def parse_command_text(text: str) -> Command:
    """Parse a Telegram message text starting with '/' into a Command."""
    text = text.strip()
    if not text.startswith("/"):
        raise ValueError(f"not a command: {text!r}")
    parts = text[1:].split()
    head = parts[0]
    if "@" in head:
        head = head.split("@", 1)[0]
    return Command(name=head.lower(), args=parts[1:])


class CommandRegistry:
    """Registers + dispatches commands. Enforces chat-ID allowlist."""

    def __init__(self, allowed_chat_ids: set[int]) -> None:
        self._allowed = set(allowed_chat_ids)
        self._handlers: dict[str, _Handler] = {}

    def register(
        self,
        name: str,
        fn: Callable[[Command, Any], str],
        destructive: bool = False,
    ) -> None:
        self._handlers[name] = _Handler(fn=fn, destructive=destructive)

    def is_destructive(self, name: str) -> bool:
        h = self._handlers.get(name)
        return bool(h and h.destructive)

    def dispatch(self, cmd: Command, chat_id: int, ctx: Any = None) -> str:
        if chat_id not in self._allowed:
            raise UnauthorizedError(f"chat_id {chat_id} not allowed")
        handler = self._handlers.get(cmd.name)
        if handler is None:
            raise UnknownCommandError(f"unknown command: /{cmd.name}")
        return handler.fn(cmd, ctx)
