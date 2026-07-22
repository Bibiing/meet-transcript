"""Single Config Provider (single source of truth) untuk GUI, CLI, dan subprocess.

Presedensi (tertinggi -> terendah):
  1. Explicit runtime override (arg CLI eksplisit / nilai yang di-pass GUI ke subprocess)
  2. Environment variable (.env / OS env, WHISPERLIVE_*)
  3. User config (QSettings per-user, persisten dari Settings GUI)
  4. Bundled app defaults (config/app_config.json, org-editable)
  5. Code fallback (konstanta aman)

Registry kunci `_KEYS` adalah SSOT pemetaan: env <-> qsettings <-> path bundled <->
fallback <-> tipe. `user_store` injectable agar dapat di-unit-test tanpa Qt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Protocol

CONFIG_VERSION = 1


def default_bundled_path() -> Path:
    """Path app_config.json ter-bundle, di-resolve via __file__ (dev & Nuitka)."""
    return Path(__file__).resolve().parent.parent / "config" / "app_config.json"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class UserStore(Protocol):
    """Backend override per-user (produksi: QSettings; test: dict)."""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...


class _KeySpec:
    __slots__ = ("env", "qs", "bundled", "fallback", "cast")

    def __init__(self, env: str, qs: str, bundled: tuple[str, ...], fallback: Any, cast: Callable[[Any], Any]) -> None:
        self.env = env
        self.qs = qs
        self.bundled = bundled
        self.fallback = fallback
        self.cast = cast


# SSOT: satu-satunya definisi kunci konfigurasi runtime yang dikelola.
_KEYS: dict[str, _KeySpec] = {
    "server_host": _KeySpec("WHISPERLIVE_HOST", "server/host", ("server", "host"), "localhost", str),
    "server_port": _KeySpec("WHISPERLIVE_PORT", "server/port", ("server", "port"), 9090, int),
    "use_tls": _KeySpec("WHISPERLIVE_USE_WSS", "server/use_tls", ("server", "use_tls"), False, _as_bool),
    "model": _KeySpec("WHISPERLIVE_MODEL", "transcription/model", ("transcription", "model"), "medium", str),
    "language": _KeySpec("WHISPERLIVE_LANGUAGE", "transcription/language", ("transcription", "language"), "id", str),
    "initial_prompt": _KeySpec("WHISPERLIVE_INITIAL_PROMPT", "transcription/initial_prompt", ("transcription", "initial_prompt"), "", str),
    "hotwords": _KeySpec("WHISPERLIVE_HOTWORDS", "transcription/hotwords", ("transcription", "hotwords"), "", str),
    # W4: URL unduh versi terbaru (org-editable lewat app_config.json ter-bundle).
    "download_url": _KeySpec("PLN_DOWNLOAD_URL", "update/download_url", ("update", "download_url"), "", str),
}


class ConfigProvider:
    def __init__(
        self,
        *,
        bundled_path: Path | None = None,
        user_store: UserStore | None = None,
        env: dict | None = None,
    ) -> None:
        self._bundled = self._load_bundled(bundled_path or default_bundled_path())
        self._user = user_store
        self._env = env if env is not None else os.environ

    @staticmethod
    def _load_bundled(path: Path) -> dict:
        """Muat app_config.json; degradasi anggun ke {} bila hilang/rusak."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _bundled_value(self, path_tuple: tuple[str, ...]) -> Any:
        node: Any = self._bundled
        for key in path_tuple:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    def get(self, key: str, *, override: Any = None) -> Any:
        spec = _KEYS[key]
        # 1. explicit override
        if override is not None:
            return spec.cast(override)
        # 2. environment variable
        env_val = self._env.get(spec.env)
        if env_val is not None and str(env_val) != "":
            return spec.cast(env_val)
        # 3. user config (QSettings)
        if self._user is not None:
            uv = self._user.get(spec.qs)
            if uv is not None and str(uv) != "":
                return spec.cast(uv)
        # 4. bundled default
        bv = self._bundled_value(spec.bundled)
        if bv is not None:
            return spec.cast(bv)
        # 5. code fallback
        return spec.fallback

    def set_user(self, key: str, value: Any) -> None:
        if self._user is None:
            raise RuntimeError("ConfigProvider has no user store; cannot persist")
        self._user.set(_KEYS[key].qs, str(value))

    # Accessor bertipe -----------------------------------------------------
    def server_host(self, override: Any = None) -> str:
        return self.get("server_host", override=override)

    def server_port(self, override: Any = None) -> int:
        return self.get("server_port", override=override)

    def use_tls(self, override: Any = None) -> bool:
        return self.get("use_tls", override=override)

    def model(self, override: Any = None) -> str:
        return self.get("model", override=override)

    def language(self, override: Any = None) -> str:
        return self.get("language", override=override)

    def initial_prompt(self, override: Any = None) -> str:
        return self.get("initial_prompt", override=override)

    def hotwords(self, override: Any = None) -> str:
        return self.get("hotwords", override=override)

    def download_url(self, override: Any = None) -> str:
        return self.get("download_url", override=override)


class QtUserStore:
    """UserStore berbasis QSettings (per-user, lokasi native OS). Qt di-import lazy."""

    def __init__(self, organization: str = "ListenPLN", application: str = "ListenPLN") -> None:
        from PySide6.QtCore import QSettings

        self._settings = QSettings(organization, application)

    def get(self, key: str) -> str | None:
        value = self._settings.value(key)
        return None if value is None else str(value)

    def set(self, key: str, value: str) -> None:
        self._settings.setValue(key, value)
        self._settings.sync()
