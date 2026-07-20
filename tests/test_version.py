"""SSOT versi aplikasi (W4)."""
from __future__ import annotations

import re
from pathlib import Path

from src.version import __version__, app_version
from whisper_live.version_policy import parse_version


def test_app_version_matches_module_constant() -> None:
    assert app_version() == __version__


def test_version_is_parseable_by_server_policy() -> None:
    # Versi yang tak dapat diparse server = client selalu ditolak saat penegakan aktif.
    assert parse_version(__version__) is not None


def test_pyproject_version_stays_in_sync_with_ssot() -> None:
    # CI menjaga tag rilis == __version__; test ini menjaga pyproject tidak menyimpang.
    content = Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, flags=re.MULTILINE)
    assert match is not None, "pyproject.toml has no version field"
    assert match.group(1) == __version__
