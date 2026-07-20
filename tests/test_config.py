from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import ConfigProvider, default_bundled_path


class DictStore:
    """UserStore palsu (Qt-free) untuk menguji presedensi tanpa QSettings."""

    def __init__(self, data: dict | None = None) -> None:
        self._d = dict(data or {})

    def get(self, key: str) -> str | None:
        return self._d.get(key)

    def set(self, key: str, value: str) -> None:
        self._d[key] = value


def _bundled(tmp_path: Path, **server_and_transcription) -> Path:
    data = {
        "version": 1,
        "server": {"host": "bundled-host", "port": 8000, "use_tls": False},
        "transcription": {"model": "medium", "language": "id", "initial_prompt": "bundled prompt", "hotwords": ""},
    }
    for section, values in server_and_transcription.items():
        data.setdefault(section, {}).update(values)
    p = tmp_path / "app_config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_precedence_override_beats_all(tmp_path: Path) -> None:
    cfg = ConfigProvider(
        bundled_path=_bundled(tmp_path),
        user_store=DictStore({"server/host": "user-host"}),
        env={"WHISPERLIVE_HOST": "env-host"},
    )
    assert cfg.server_host(override="explicit-host") == "explicit-host"


def test_precedence_env_over_user_over_bundled(tmp_path: Path) -> None:
    # env menang atas user & bundled
    cfg = ConfigProvider(
        bundled_path=_bundled(tmp_path),
        user_store=DictStore({"server/host": "user-host"}),
        env={"WHISPERLIVE_HOST": "env-host"},
    )
    assert cfg.server_host() == "env-host"

    # tanpa env: user menang atas bundled
    cfg2 = ConfigProvider(
        bundled_path=_bundled(tmp_path),
        user_store=DictStore({"server/host": "user-host"}),
        env={},
    )
    assert cfg2.server_host() == "user-host"

    # tanpa env & user: bundled
    cfg3 = ConfigProvider(bundled_path=_bundled(tmp_path), user_store=DictStore(), env={})
    assert cfg3.server_host() == "bundled-host"


def test_code_fallback_when_bundled_missing(tmp_path: Path) -> None:
    cfg = ConfigProvider(bundled_path=tmp_path / "nonexistent.json", user_store=None, env={})
    assert cfg.server_host() == "localhost"  # code fallback
    assert cfg.server_port() == 9090
    assert cfg.model() == "medium"
    assert cfg.language() == "id"


def test_bundled_corrupt_json_degrades_gracefully(tmp_path: Path) -> None:
    p = tmp_path / "app_config.json"
    p.write_text("{ not valid json", encoding="utf-8")
    cfg = ConfigProvider(bundled_path=p, user_store=None, env={})
    assert cfg.server_host() == "localhost"  # jatuh ke fallback, tidak crash


def test_typed_casts(tmp_path: Path) -> None:
    cfg = ConfigProvider(
        bundled_path=_bundled(tmp_path),
        env={"WHISPERLIVE_PORT": "1234", "WHISPERLIVE_USE_WSS": "true"},
    )
    assert cfg.server_port() == 1234 and isinstance(cfg.server_port(), int)
    assert cfg.use_tls() is True


def test_set_user_persists_via_store(tmp_path: Path) -> None:
    store = DictStore()
    cfg = ConfigProvider(bundled_path=_bundled(tmp_path), user_store=store, env={})
    cfg.set_user("server_host", "new-host")
    cfg.set_user("server_port", 7777)
    assert store.get("server/host") == "new-host"
    assert ConfigProvider(bundled_path=_bundled(tmp_path), user_store=store, env={}).server_host() == "new-host"
    assert ConfigProvider(bundled_path=_bundled(tmp_path), user_store=store, env={}).server_port() == 7777


def test_set_user_without_store_raises(tmp_path: Path) -> None:
    cfg = ConfigProvider(bundled_path=_bundled(tmp_path), user_store=None, env={})
    with pytest.raises(RuntimeError):
        cfg.set_user("server_host", "x")


def test_real_bundled_config_has_domain_adaptation_defaults() -> None:
    # Fix inti milestone: paket tanpa .env tetap punya prompt (domain adaptation).
    cfg = ConfigProvider(user_store=None, env={})
    assert cfg.initial_prompt().strip() != ""
    assert cfg.model() == "medium"
    assert default_bundled_path().exists()


def test_qsettings_user_store_persists_across_instances(tmp_path: Path) -> None:
    # DoD "persisten lintas restart" untuk backend QSettings nyata.
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from src.config import QtUserStore

    # Redirect QSettings ke file Ini di tmp agar tidak mengotori registry/plist.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    store1 = QtUserStore(organization="PLNTest_M2", application="AppTest_M2")
    store1.set("server/host", "persisted-host")

    # Instance baru = simulasi restart aplikasi.
    store2 = QtUserStore(organization="PLNTest_M2", application="AppTest_M2")
    assert store2.get("server/host") == "persisted-host"
