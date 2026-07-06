"""Structured process log untuk event server WhisperLive.

Server menulis JSONL ke `WHISPERLIVE_PROCESS_LOG` atau `/app/logs/process.log`.
Log ini berisi koneksi client, opsi stream, audio diterima, ASR, local agreement,
TVE score/drop/emit, dan pengiriman hasil ke client.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_TEXT_LIMIT = 500


def log_process_event(event: str, **fields: Any) -> None:
    """Tambahkan satu event JSONL ke process log server."""

    path = Path(os.getenv("WHISPERLIVE_PROCESS_LOG", "/app/logs/process.log"))
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        **{key: _safe_value(value) for key, value in fields.items()},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with _LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # pragma: no cover - observability must not break ASR
        _LOG.debug("failed to write process log event=%s error=%s", event, exc)


def preview_text(text: Any, limit: int = 160) -> str:
    """Potong teks transcript agar log tetap ringkas dan aman."""
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _safe_value(value: Any) -> Any:
    """Ubah field log menjadi JSON-safe dan batasi string panjang."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > _TEXT_LIMIT:
            return value[:_TEXT_LIMIT] + "...[truncated]"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    return str(value)
