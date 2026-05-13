# SPDX-License-Identifier: Apache-2.0
"""Tests for the front-month Redis key helper."""

from __future__ import annotations

import pytest

from alpha_assay.data.front_month import (
    FrontMonthMissingError,
    InvalidExpiryError,
    front_month_redis_key,
    read_front_month,
    validate_yyyymmdd,
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


# --- validate_yyyymmdd ------------------------------------------------


def test_validate_yyyymmdd_valid_passes():
    assert validate_yyyymmdd("20260619") == "20260619"
    assert validate_yyyymmdd("20250101") == "20250101"


def test_validate_yyyymmdd_wrong_length_raises():
    with pytest.raises(InvalidExpiryError, match="expected 8 digits"):
        validate_yyyymmdd("202606")


def test_validate_yyyymmdd_non_digits_raises():
    with pytest.raises(InvalidExpiryError, match="expected 8 digits"):
        validate_yyyymmdd("2026-06-19")


def test_validate_yyyymmdd_non_string_raises():
    with pytest.raises(InvalidExpiryError, match="expected 8 digits"):
        validate_yyyymmdd(20260619)  # type: ignore[arg-type]


def test_validate_yyyymmdd_invalid_date_raises():
    """8 digits that form an impossible date (Feb 30) must be rejected."""
    with pytest.raises(InvalidExpiryError):
        validate_yyyymmdd("20260230")


def test_validate_yyyymmdd_source_in_error_message():
    with pytest.raises(InvalidExpiryError, match="redis read"):
        validate_yyyymmdd("bad", source="redis read")


# --- write_front_month validation gate --------------------------------


def test_write_front_month_rejects_invalid_expiry(fake_redis):
    """write_front_month must not persist garbage to Redis."""
    with pytest.raises(InvalidExpiryError):
        write_front_month(fake_redis, symbol="ES", exchange="CME", expiry="bad")
    # Key must not have been set.
    with pytest.raises(FrontMonthMissingError):
        read_front_month(fake_redis, symbol="ES", exchange="CME")


# --- read_front_month validation gate ---------------------------------


def test_read_front_month_rejects_corrupt_redis_value(fake_redis):
    """If Redis somehow holds a bad value, read_front_month raises InvalidExpiryError."""
    # Bypass write_front_month's validation by writing directly to Redis.
    key = front_month_redis_key("ES", "CME")
    fake_redis.set(key, "corrupt")
    with pytest.raises(InvalidExpiryError, match="redis read"):
        read_front_month(fake_redis, symbol="ES", exchange="CME")
