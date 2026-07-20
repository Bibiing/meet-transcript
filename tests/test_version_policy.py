"""Kebijakan versi minimum sisi server (W4)."""
from __future__ import annotations

import pytest

from whisper_live.version_policy import (
    is_client_outdated,
    parse_version,
    validate_min_client_version,
)


def test_parse_version_valid() -> None:
    assert parse_version("1.2.3") == (1, 2, 3)
    assert parse_version(" 0.1.0 ") == (0, 1, 0)


@pytest.mark.parametrize("value", ["1.2", "1.2.3.4", "1.2.x", "v1.2.3", "1.2.3-rc1", "", None, 123, "-1.0.0"])
def test_parse_version_rejects_unknown_formats(value) -> None:
    assert parse_version(value) is None


def test_version_comparison_is_numeric_not_lexicographic() -> None:
    # Regresi: sebagai string "0.10.0" < "0.9.0"; sebagai tuple tidak.
    assert is_client_outdated("0.10.0", "0.9.0") is False
    assert is_client_outdated("0.9.0", "0.10.0") is True


def test_client_at_or_above_minimum_is_allowed() -> None:
    assert is_client_outdated("1.2.0", "1.2.0") is False
    assert is_client_outdated("1.2.1", "1.2.0") is False
    assert is_client_outdated("2.0.0", "1.9.9") is False


def test_client_below_minimum_is_outdated() -> None:
    assert is_client_outdated("1.1.9", "1.2.0") is True


def test_enforcement_disabled_when_minimum_empty() -> None:
    # Default (kosong) = penegakan mati: perilaku server tidak berubah.
    for minimum in ("", "   ", None):
        assert is_client_outdated("0.0.0", minimum) is False
        assert is_client_outdated(None, minimum) is False


def test_missing_or_unparsable_client_version_is_outdated_when_enforcing() -> None:
    # Fail-closed: build pra-W4 tidak mengirim client_version.
    assert is_client_outdated(None, "1.0.0") is True
    assert is_client_outdated("", "1.0.0") is True
    assert is_client_outdated("garbage", "1.0.0") is True
    assert is_client_outdated(123, "1.0.0") is True


def test_validate_min_client_version_normalizes_and_accepts_empty() -> None:
    assert validate_min_client_version(None) == ""
    assert validate_min_client_version("  ") == ""
    assert validate_min_client_version(" 1.2.3 ") == "1.2.3"


@pytest.mark.parametrize("value", ["1.2", "v1.2.3", "latest", "1.2.3-rc1"])
def test_validate_min_client_version_rejects_typos(value) -> None:
    # Salah ketik kebijakan harus gagal saat server start, bukan menolak semua client.
    with pytest.raises(ValueError):
        validate_min_client_version(value)
