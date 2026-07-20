"""Deteksi permission mikrofon (W3) — logika murni, tanpa Qt.

Kontrak penting (keputusan desain W3):

* **Source of truth** status permission = hasil *percobaan membuka device* mikrofon
  (sounddevice/PortAudio). Hanya ini yang membuktikan aplikasi benar-benar dapat
  meng-capture audio.
* **Registry `ConsentStore` = INDIKATOR**, bukan source of truth. Nilainya hanya
  dipakai untuk mengklasifikasikan penyebab (setelan privasi vs device) sehingga
  panduan yang ditampilkan tepat, dan untuk menangkap kasus Windows yang membuka
  device namun hanya mengirim senyap saat privasi ditolak.

Windows memakai toggle privasi GLOBAL untuk aplikasi desktop (Win32/non-packaged),
bukan prompt per-aplikasi seperti macOS; karena itu panduan mengarahkan pengguna ke
`ms-settings:privacy-microphone`.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)

PRIVACY_ALLOWED = "allowed"
PRIVACY_DENIED = "denied"
PRIVACY_UNKNOWN = "unknown"

GUIDANCE_NONE = "none"
GUIDANCE_PRIVACY = "privacy"
GUIDANCE_DEVICE = "device"

PRIVACY_SETTINGS_URI = "ms-settings:privacy-microphone"

# Aplikasi desktop Win32 butuh toggle global DAN toggle "desktop apps" (NonPackaged).
_CONSENT_SUBKEYS = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone",
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone\NonPackaged",
)

_MESSAGE_PRIVACY = (
    "Aplikasi tidak dapat menggunakan mikrofon karena akses mikrofon untuk aplikasi "
    "desktop sedang dinonaktifkan pada setelan privasi Windows.\n\n"
    "Buka Setelan Privasi, aktifkan \"Microphone access\" dan \"Let desktop apps access "
    "your microphone\", lalu pilih Coba lagi."
)
_MESSAGE_DEVICE = (
    "Aplikasi tidak dapat membuka mikrofon. Mikrofon mungkin belum terpasang, "
    "dinonaktifkan, atau sedang dipakai aplikasi lain.\n\n"
    "Periksa mikrofon Anda (dan setelan privasi Windows bila mikrofon terlihat normal), "
    "lalu pilih Coba lagi."
)


@dataclass(frozen=True)
class MicrophoneAccess:
    """Hasil pemeriksaan mikrofon.

    accessible: source of truth (device berhasil dibuka).
    privacy: indikator registry (allowed/denied/unknown) — hanya untuk klasifikasi.
    detail: pesan error device untuk diagnostik/log (kosong bila berhasil).
    """

    accessible: bool
    privacy: str
    detail: str = ""


def source_uses_microphone(source: Optional[str]) -> bool:
    """True bila source rekaman melibatkan mikrofon (gate hanya relevan untuk ini)."""
    return str(source or "both").strip().lower() in {"mic", "both"}


def _read_consent_value(subkey: str) -> Optional[str]:
    try:
        import winreg
    except ImportError:  # non-Windows
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey) as key:
            value, _ = winreg.QueryValueEx(key, "Value")
    except OSError:
        return None
    return str(value)


def microphone_privacy_indicator(reader: Callable[[str], Optional[str]] | None = None) -> str:
    """INDIKATOR privasi mikrofon dari registry Windows — BUKAN source of truth.

    Kembalikan `denied` bila salah satu toggle (global / desktop apps) memblokir,
    `allowed` bila semua yang terbaca mengizinkan, `unknown` bila tak dapat dibaca
    (registry hilang/OS lain) sehingga pemanggil bergantung pada hasil buka device.
    """
    if reader is None:
        if sys.platform != "win32":
            return PRIVACY_UNKNOWN
        reader = _read_consent_value

    values = [v for v in (reader(subkey) for subkey in _CONSENT_SUBKEYS) if v is not None]
    if not values:
        return PRIVACY_UNKNOWN
    normalized = [v.strip().lower() for v in values]
    if any(v == "deny" for v in normalized):
        return PRIVACY_DENIED
    if all(v == "allow" for v in normalized):
        return PRIVACY_ALLOWED
    return PRIVACY_UNKNOWN


def _default_opener(device: Any) -> None:
    """Buka-tutup stream input singkat: membuktikan device benar-benar dapat dipakai."""
    import sounddevice as sd

    with sd.InputStream(device=device, channels=1):
        pass


def check_microphone(
    device: Any = None,
    *,
    opener: Callable[[Any], None] | None = None,
    privacy_reader: Callable[[str], Optional[str]] | None = None,
) -> MicrophoneAccess:
    """Periksa mikrofon: buka device (source of truth) + baca indikator registry."""
    open_stream = opener or _default_opener
    accessible = True
    detail = ""
    try:
        open_stream(device)
    except Exception as exc:  # PortAudioError & turunan lain dari backend audio
        accessible = False
        detail = str(exc)
        _log.info("microphone open test failed (device=%r): %s", device, exc)

    privacy = microphone_privacy_indicator(privacy_reader) if privacy_reader else microphone_privacy_indicator()
    _log.info("microphone check: accessible=%s privacy_indicator=%s", accessible, privacy)
    return MicrophoneAccess(accessible=accessible, privacy=privacy, detail=detail)


def guidance_kind(access: MicrophoneAccess) -> str:
    """Panduan apa yang perlu ditampilkan.

    Indikator `denied` menang walau device "terbuka": pada Windows, privasi yang
    ditolak dapat menghasilkan stream terbuka namun senyap — kondisi yang tak dapat
    dibedakan dari ruangan sunyi tanpa indikator ini.
    """
    if access.privacy == PRIVACY_DENIED:
        return GUIDANCE_PRIVACY
    if not access.accessible:
        return GUIDANCE_DEVICE
    return GUIDANCE_NONE


def guidance_message(access: MicrophoneAccess) -> str:
    """Pesan panduan sesuai penyebab; kosong bila mikrofon siap."""
    kind = guidance_kind(access)
    if kind == GUIDANCE_PRIVACY:
        return _MESSAGE_PRIVACY
    if kind == GUIDANCE_DEVICE:
        return _MESSAGE_DEVICE
    return ""


def open_privacy_settings() -> bool:
    """Buka halaman Setelan Privasi mikrofon Windows. False bila tak dapat dibuka."""
    if sys.platform != "win32":
        return False
    try:
        os.startfile(PRIVACY_SETTINGS_URI)  # type: ignore[attr-defined]
    except OSError as exc:
        _log.warning("failed to open privacy settings: %s", exc)
        return False
    return True
