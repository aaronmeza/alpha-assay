# tests/test_bot_dispatch.py
from __future__ import annotations

import pytest

from infra.bot.dispatch import (
    Command,
    CommandRegistry,
    UnauthorizedError,
    UnknownCommandError,
    parse_command_text,
)


def test_parse_simple_command():
    cmd = parse_command_text("/health")
    assert cmd.name == "health"
    assert cmd.args == []


def test_parse_command_with_args():
    cmd = parse_command_text("/logs es-bars-recorder 100")
    assert cmd.name == "logs"
    assert cmd.args == ["es-bars-recorder", "100"]


def test_parse_strips_at_botname():
    # Telegram includes @botname in group chats.
    cmd = parse_command_text("/health@alphaassaybot")
    assert cmd.name == "health"


def test_dispatcher_rejects_unknown_chat():
    reg = CommandRegistry(allowed_chat_ids={1234})
    with pytest.raises(UnauthorizedError):
        reg.dispatch(Command(name="health", args=[]), chat_id=9999)


def test_dispatcher_rejects_unknown_command():
    reg = CommandRegistry(allowed_chat_ids={1234})
    with pytest.raises(UnknownCommandError):
        reg.dispatch(Command(name="bogus", args=[]), chat_id=1234)


def test_dispatcher_runs_registered_handler():
    reg = CommandRegistry(allowed_chat_ids={1234})
    reg.register("ping", lambda cmd, ctx: "pong")
    result = reg.dispatch(Command(name="ping", args=[]), chat_id=1234)
    assert result == "pong"


def test_dispatcher_marks_destructive_handlers():
    reg = CommandRegistry(allowed_chat_ids={1234})
    reg.register("restart", lambda cmd, ctx: "ok", destructive=True)
    reg.register("status", lambda cmd, ctx: "ok")
    assert reg.is_destructive("restart") is True
    assert reg.is_destructive("status") is False
    assert reg.is_destructive("nope") is False
