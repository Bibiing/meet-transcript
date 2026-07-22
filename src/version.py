"""Single source of truth versi aplikasi client (W4).

Versi ini dikirim ke server pada handshake (`client_version`) dan dipakai untuk
penegakan versi minimum. CI memvalidasi tag rilis `vX.Y.Z` == `__version__` dan
mem-pass nilai yang sama ke installer, sehingga versi yang tampil di Windows,
versi di handshake, dan tag rilis tidak dapat menyimpang diam-diam.

Naikkan nilai ini pada rilis; test menjaga `pyproject.toml` tetap sinkron.
"""
from __future__ import annotations

__version__ = "1.0.0"


def app_version() -> str:
    """Versi client yang sedang berjalan."""
    return __version__
