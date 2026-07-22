"""Lokasi data per-user yang PERSISTEN (tanpa Qt; dipakai GUI maupun subprocess CLI).

Motivasi: sebelumnya transkrip ditulis relatif terhadap `__file__` (root repo).
Pada build Nuitka onefile, root itu adalah direktori ekstraksi sementara di
`%TEMP%\\onefile_xxx\\` yang DIHAPUS saat aplikasi ditutup dan bernama acak tiap
peluncuran. Akibatnya seluruh transkrip hilang saat aplikasi ditutup dan tombol
History tidak punya berkas yang bisa dibuka.

Modul ini menyediakan direktori data per-user yang stabil dan bertahan antar-sesi,
tanpa memerlukan hak admin (sejalan dengan installer per-user).

Path dihitung dari environment/HOME saat runtime (bukan `__file__`), sehingga sama
di jalur dev maupun packaged, dan sama antara proses GUI dan subprocess-nya.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "ListenPLN"


def app_data_dir() -> Path:
    """Direktori data per-user milik aplikasi (belum tentu sudah dibuat)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME")
        root = Path(base) if base else Path.home() / ".local" / "share"
    return root / APP_DIR_NAME


def history_dir() -> Path:
    """Folder khusus riwayat transkrip (satu berkas JSON per sesi rapat)."""
    return app_data_dir() / "history"


def logs_dir() -> Path:
    return app_data_dir() / "logs"


def current_transcript_log() -> Path:
    """Berkas transkrip sesi yang sedang berjalan (di-arsip ke history saat selesai)."""
    return app_data_dir() / "transcript_log.json"


def ensure_dir(path: Path) -> Path:
    """Buat direktori bila belum ada; kembalikan path-nya. Degradasi anggun bila gagal."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path
