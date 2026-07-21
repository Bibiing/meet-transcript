from __future__ import annotations

import pytest

from src.permissions import (
    GUIDANCE_DEVICE,
    GUIDANCE_NONE,
    GUIDANCE_PRIVACY,
    PRIVACY_ALLOWED,
    PRIVACY_DENIED,
    PRIVACY_UNKNOWN,
    MicrophoneAccess,
    check_microphone,
    guidance_kind,
    guidance_message,
    microphone_privacy_indicator,
    source_uses_microphone,
)


def _reader(mapping: dict[str, str | None]):
    return lambda subkey: mapping.get(subkey)


def _consent(global_value: str | None, nonpackaged_value: str | None) -> dict[str, str | None]:
    from src.permissions import _CONSENT_SUBKEYS

    return {_CONSENT_SUBKEYS[0]: global_value, _CONSENT_SUBKEYS[1]: nonpackaged_value}


# Indikator registry (BUKAN source of truth) --------------------------------
def test_privacy_indicator_allowed_when_both_toggles_allow() -> None:
    assert microphone_privacy_indicator(_reader(_consent("Allow", "Allow"))) == PRIVACY_ALLOWED


def test_privacy_indicator_denied_when_desktop_apps_toggle_denies() -> None:
    # Aplikasi Win32 juga butuh toggle NonPackaged: satu Deny sudah memblokir.
    assert microphone_privacy_indicator(_reader(_consent("Allow", "Deny"))) == PRIVACY_DENIED


def test_privacy_indicator_unknown_when_registry_unreadable() -> None:
    assert microphone_privacy_indicator(_reader(_consent(None, None))) == PRIVACY_UNKNOWN


def test_privacy_indicator_unknown_on_unexpected_value() -> None:
    assert microphone_privacy_indicator(_reader(_consent("Prompt", "Allow"))) == PRIVACY_UNKNOWN


# Source of truth: buka device ---------------------------------------------
def test_check_microphone_accessible_when_device_opens() -> None:
    access = check_microphone(opener=lambda device: None, privacy_reader=_reader(_consent("Allow", "Allow")))
    assert access.accessible is True
    assert access.privacy == PRIVACY_ALLOWED
    assert access.detail == ""


def test_check_microphone_not_accessible_when_open_raises() -> None:
    def boom(device):
        raise RuntimeError("device unavailable")

    access = check_microphone(opener=boom, privacy_reader=_reader(_consent(None, None)))
    assert access.accessible is False
    assert access.privacy == PRIVACY_UNKNOWN
    assert "device unavailable" in access.detail


def test_check_microphone_passes_device_to_opener() -> None:
    seen = {}
    check_microphone(3, opener=lambda device: seen.setdefault("device", device), privacy_reader=_reader({}))
    assert seen["device"] == 3


# Klasifikasi panduan ------------------------------------------------------
def test_guidance_none_when_accessible_and_privacy_allowed() -> None:
    access = MicrophoneAccess(accessible=True, privacy=PRIVACY_ALLOWED)
    assert guidance_kind(access) == GUIDANCE_NONE
    assert guidance_message(access) == ""


def test_guidance_privacy_when_indicator_denied() -> None:
    access = MicrophoneAccess(accessible=False, privacy=PRIVACY_DENIED)
    assert guidance_kind(access) == GUIDANCE_PRIVACY
    assert "privasi" in guidance_message(access).lower()


def test_guidance_privacy_when_device_opens_but_indicator_denied() -> None:
    # Regresi: privasi ditolak dapat menghasilkan stream terbuka namun senyap.
    access = MicrophoneAccess(accessible=True, privacy=PRIVACY_DENIED)
    assert guidance_kind(access) == GUIDANCE_PRIVACY


def test_guidance_device_when_not_accessible_without_privacy_block() -> None:
    access = MicrophoneAccess(accessible=False, privacy=PRIVACY_ALLOWED)
    assert guidance_kind(access) == GUIDANCE_DEVICE
    assert "mikrofon" in guidance_message(access).lower()


def test_guidance_device_when_not_accessible_and_indicator_unknown() -> None:
    assert guidance_kind(MicrophoneAccess(accessible=False, privacy=PRIVACY_UNKNOWN)) == GUIDANCE_DEVICE


@pytest.mark.parametrize(
    "source, expected",
    [("mic", True), ("both", True), (None, True), ("BOTH", True), ("speaker", False)],
)
def test_source_uses_microphone(source, expected) -> None:
    assert source_uses_microphone(source) is expected


# Regresi W3: kegagalan mikrofon di subprocess TIDAK boleh ditelan diam-diam ---
def test_mic_failure_is_emitted_to_parent_not_swallowed() -> None:
    """capture.py wajib menaikkan kegagalan mic sebagai status, bukan hanya log."""
    from pathlib import Path

    source = Path("src/whisper/capture.py").read_text(encoding="utf-8")
    assert "emit_status(\"mic\", \"CLIENT_MIC_ERROR\"" in source


def test_engine_records_mic_error_and_sets_error_state() -> None:
    from src.core import engine
    from src.utils.status_ipc import format_status_line

    engine.STATE.mic_error = None
    engine._handle_monitor_line(
        format_status_line("mic", "CLIENT_MIC_ERROR", reason="Error opening InputStream")
    )
    assert engine.mic_error_info() == {"reason": "Error opening InputStream"}
    assert engine.connection_status() == "ERROR"
    engine.STATE.mic_error = None


def test_engine_mic_error_none_by_default() -> None:
    from src.core import engine

    engine.STATE.mic_error = None
    assert engine.mic_error_info() is None
