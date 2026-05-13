# SPDX-License-Identifier: Apache-2.0
"""Tests for the front-month Redis key helper."""

from __future__ import annotations

import pytest

from alpha_assay.data.front_month import (
    FrontMonthMissingError,
    front_month_redis_key,
    read_front_month,
    write_front_month,
)


def test_front_month_redis_key_format():
    assert front_month_redis_key("ES", "CME") == "alpha_assay:front_month:es.cme"
    assert front_month_redis_key("MES", "CME") == "alpha_assay:front_month:mes.cme"


def test_write_and_read_round_trip(fake_redis):
    write_front_month(fake_redis, symbol="ES", exchange="CME", expiry="20260619")
    assert read_front_month(fake_redis, symbol="ES", exchange="CME") == "20260619"


def test_read_missing_raises(fake_redis):
    with pytest.raises(FrontMonthMissingError, match="ES@CME"):
        read_front_month(fake_redis, symbol="ES", exchange="CME")
