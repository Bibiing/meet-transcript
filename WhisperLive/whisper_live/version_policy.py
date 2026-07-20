"""Kebijakan versi minimum client (W4).

Server adalah titik penegakan otoritatif: client mengirim `client_version` pada
handshake, server menolak yang di bawah minimum sebelum sesi dimulai.

Kebijakan bersifat opt-in: minimum kosong = penegakan mati (dipakai pengembangan
lokal, di mana build dev memakai versi `0.0.0`). Nilai minimum divalidasi saat
server start agar salah ketik saat deploy gagal cepat, bukan menolak seluruh
client diam-diam saat runtime.
"""
from __future__ import annotations

from typing import Optional, Tuple

Version = Tuple[int, int, int]


def parse_version(value: object) -> Optional[Version]:
    """Parse "X.Y.Z" menjadi tuple integer; None bila bukan format yang dikenal.

    Perbandingan wajib memakai tuple, bukan string: sebagai string "0.10.0" < "0.9.0".
    """
    if not isinstance(value, str):
        return None
    parts = value.strip().split(".")
    if len(parts) != 3:
        return None
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError:
        return None
    if major < 0 or minor < 0 or patch < 0:
        return None
    return (major, minor, patch)


def validate_min_client_version(value: Optional[str]) -> str:
    """Normalisasi & validasi kebijakan saat start. ValueError bila tak valid."""
    if value is None:
        return ""
    normalized = value.strip()
    if not normalized:
        return ""
    if parse_version(normalized) is None:
        raise ValueError(
            f"min_client_version must be in X.Y.Z format, got {value!r}"
        )
    return normalized


def is_client_outdated(client_version: object, min_client_version: Optional[str]) -> bool:
    """True bila client harus ditolak.

    Fail-closed: saat kebijakan aktif, versi yang hilang (build pra-W4) atau tak
    dapat diparse dianggap usang.
    """
    minimum = parse_version((min_client_version or "").strip())
    if minimum is None:
        return False  # kebijakan mati (kosong) atau tidak valid -> jangan blokir
    client = parse_version(client_version)
    if client is None:
        return True
    return client < minimum
