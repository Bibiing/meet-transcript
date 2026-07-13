"""Deteksi mode produksi vs development berdasarkan Nuitka compilation status.

Fungsi ini mendeteksi apakah aplikasi dijalankan sebagai:
- Development: Source code Python langsung (via `python -m` atau `uv run`)
- Production: Compiled binary dari Nuitka (.exe di Windows, .bin/.app di macOS)
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_production_build() -> bool:
    """
    Deteksi apakah aplikasi dijalankan sebagai bundle Nuitka (production) atau source code (development).
    
    Returns:
        bool: True jika dijalankan sebagai Nuitka bundle (production), False jika dari source code (development).
    
    Detection logic:
    - Nuitka bundle memiliki atribut __compiled__ yang diset ke nilai non-string
    - Aplikasi compiled juga akan berjalan sebagai single executable file (.exe, .bin, .app)
    - Source code akan memiliki __file__ yang menunjuk ke module Python biasa
    """
    
    # Cara 1: Cek atribut __compiled__ yang ditambahkan Nuitka
    if hasattr(sys.modules.get("__main__"), "__compiled__"):
        return True
    
    # Cara 2: Cek apakah sys.argv[0] adalah .exe (Windows) atau tidak ada ekstensi .py/.pyw (executable binary)
    executable = Path(sys.argv[0]).resolve()
    
    # Jika file executable adalah .exe atau .bin atau .app, itu Nuitka bundle
    if executable.suffix.lower() in {".exe", ".bin", ".app"}:
        return True
    
    # Jika nama file sudah bersih tanpa ekstensi Python, mungkin itu executable
    if not executable.suffix or executable.suffix.lower() not in {".py", ".pyw", ".pyc"}:
        # Cek apakah file ada dan readable, tanpa ekstensi .py
        if executable.exists() and not executable.suffix:
            return True
    
    return False


def get_production_safe_log_level() -> str:
    """
    Tentukan log level yang aman untuk production vs development.
    
    Returns:
        str: "WARNING" untuk production, "INFO" untuk development.
    """
    return "WARNING" if is_production_build() else "INFO"


def should_enable_debug_features() -> bool:
    """
    Tentukan apakah debug features (audio archive, raw audio debug) harus diaktifkan.
    
    Returns:
        bool: False untuk production, True untuk development.
    """
    return not is_production_build()
