# SPDX-License-Identifier: Apache-2.0
"""Tests for paper_dryrun._resolve_es_expiry.

Verifies the three-case resolution contract:
  1. ES_EXPIRY env var set   -> use it; do not read Redis.
  2. ES_EXPIRY unset         -> read from the Redis front-month metadata key.
  3. Neither present         -> FrontMonthMissingError (fail closed).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from alpha_assay.data.front_month import FrontMonthMissingError, write_front_month

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "paper_dryrun.py"
_MODULE_NAME = "paper_dryrun"


def _load_script_module():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _resolve_es_expiry
# ---------------------------------------------------------------------------


def test_resolve_expiry_prefers_env_override(fake_redis, monkeypatch):
    """If ES_EXPIRY is set, use it; do not read from Redis."""
    monkeypatch.setenv("ES_EXPIRY", "20260919")
    write_front_month(fake_redis, symbol="ES", exchange="CME", expiry="20260618")
    module = _load_script_module()
    assert module._resolve_es_expiry(fake_redis) == ("20260919", "ES_EXPIRY env")


def test_resolve_expiry_reads_redis_when_unset(fake_redis, monkeypatch):
    """If ES_EXPIRY unset, read from Redis."""
    monkeypatch.delenv("ES_EXPIRY", raising=False)
    write_front_month(fake_redis, symbol="ES", exchange="CME", expiry="20260619")
    module = _load_script_module()
    assert module._resolve_es_expiry(fake_redis) == ("20260619", "Redis metadata key")


def test_resolve_expiry_raises_when_neither_present(fake_redis, monkeypatch):
    """No env + no key -> fail closed."""
    monkeypatch.delenv("ES_EXPIRY", raising=False)
    module = _load_script_module()
    with pytest.raises(FrontMonthMissingError):
        module._resolve_es_expiry(fake_redis)
