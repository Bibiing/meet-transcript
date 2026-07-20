from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path = Path(".env")) -> None:
    # memuat environtment variabel dari file .env ke dalam os.environ

    # .env bersifat opsional (dev). Paket produksi memakai config provider +
    # QSettings, jadi ketiadaan .env adalah kondisi normal — jangan berisik.
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip() # menghapus spasi

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1) # memisahkan key dan value
        key = key.strip()
        value = value.strip().strip('"').strip("'") # menghapus tanda kutip di awal dan akhir value
        if key:
            os.environ.setdefault(key, value) # menambahkan key dan value ke os.environ


def env_optional(name: str) -> str | None:
    value = os.getenv(name) # mengambil nilai dari environtment variabel
    return value if value else None # mengembalikan nilai jika ada, jika tidak mengembalikan None


def env_str(name: str, default: str) -> str:
    return os.getenv(name) or default 


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
