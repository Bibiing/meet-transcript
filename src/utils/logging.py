from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.whisper.models import TranscriptionResult
from src.utils.formatter import format_transcript_line, result_to_line

__all__ = [
    "configure_logging",
    "configure_process_logging",
    "clear_process_log_context",
    "flush_process_log_summaries",
    "PROCESS_LOG_PATH",
    "log_process_event",
    "load_transcript_entries",
    "TranscriptLog",
    "STABILITY_STABLE",
    "STABILITY_CANDIDATE",
]

# Nilai stability entry transcript (dipusatkan agar tidak ada magic string tersebar)
STABILITY_STABLE = "stable"
STABILITY_CANDIDATE = "candidate"

_managed_handlers: list[logging.Handler] = []
_APP_LOG_MAX_BYTES = int(os.getenv("TRANSCRIBER_LOG_MAX_BYTES", "5000000") or 5_000_000)
_APP_LOG_BACKUP_COUNT = int(os.getenv("TRANSCRIBER_LOG_BACKUP_COUNT", "5") or 5)

# 1. Konfigurasi Logging Aplikasi
def configure_logging(
    level: str = "INFO",
    log_file: Path = Path("logs") / "transcriber.log",
) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()              # mengambil logger root
    
    # hapus hanya handler yang dibuat configure_logging
    for handler in _managed_handlers:
        if handler in root.handlers:
            root.removeHandler(handler)
    _managed_handlers.clear()

    root.setLevel(_coerce_level(level))     # mengatur level logging sesuai dengan level yang diberikan

    # format log: waktu level nama_logger: pesan
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # file handler untuk menulis log ke file dengan rotasi agar log tidak tumbuh tanpa batas
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max(1, _APP_LOG_MAX_BYTES),
        backupCount=max(0, _APP_LOG_BACKUP_COUNT),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)                            # mengatur format log
    root.addHandler(file_handler)                                   # menambahkan handler ke logger root

    # console handler untuk menampilkan log ke console (stderr)
    console_handler = logging.StreamHandler(sys.stderr)     # handler menulis log ke console
    console_handler.setFormatter(formatter)                 # mengatur format log
    console_handler.setLevel(logging.WARNING)               # level warning ke atas akan ditampilkan di console
    root.addHandler(console_handler)                        # menambahkan handler ke logger root

    _managed_handlers.extend([file_handler, console_handler])  # menyimpan handler yang dikelola agar bisa dihapus nanti

    # membuat logger dengan nama "transcriber"
    logger = logging.getLogger("transcriber")
    logger.info("logging configured level=%s file=%s", level.upper(), log_file)
    return logger

# mengubah level log dari string ke integer
def _coerce_level(level: str) -> int:
    normalized = level.upper()
    if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    return int(getattr(logging, normalized))


# 2. Structured Process Log (JSONL, best-effort)
# melacak alur detail: capture, VAD pass/drop, queue, WebSocket, transcript masuk, dan finalisasi. Logging dibuat best-effort agar tidak pernah menghentikan audio.

_LOG = logging.getLogger("process_log") 
_LOCK = threading.Lock()
PROCESS_LOG_PATH = Path("logs") / "process.log"
PROCESS_LOG_MAX_BYTES = int(os.getenv("PROCESS_LOG_MAX_BYTES", "5000000") or 5_000_000)
PROCESS_LOG_BACKUP_COUNT = int(os.getenv("PROCESS_LOG_BACKUP_COUNT", "5") or 5)
_TEXT_LIMIT = 500
_HOT_PATH_EVENTS = {
    "client.chunk_created",
    "client.vad_pass",
    "client.vad_drop",
    "client.chunk_queued",
    "client.chunk_sent",
}

_process_log_dir_ready = False  # cache: direktori logs/ hanya perlu dibuat sekali per proses
_process_log_path = PROCESS_LOG_PATH
_process_log_context: dict[str, Any] = {}
_process_log_include_hot_path = False
_process_log_summary_interval_seconds = 5.0
_process_log_last_summary_at = 0.0
_process_log_summaries: dict[tuple[str, str, str], dict[str, Any]] = {}


def configure_process_logging(
    *,
    session_id: str | None = None,
    include_hot_path: bool = False,
    summary_interval_seconds: float = 5.0,
    process_log_path: Path | None = None,
) -> None:
    """Atur context process log untuk satu live session.

    Context disisipkan ke semua baris JSONL. Event hot-path dapat diringkas
    agar `process.log` tetap terbaca ketika capture berjalan lama.
    """

    global _process_log_context
    global _process_log_include_hot_path
    global _process_log_summary_interval_seconds
    global _process_log_last_summary_at
    global _process_log_path
    global _process_log_dir_ready

    with _LOCK:
        _flush_process_log_summaries_locked(force=True)
        _process_log_context = {}
        if session_id:
            _process_log_context["session_id"] = session_id
        _process_log_include_hot_path = include_hot_path
        _process_log_summary_interval_seconds = max(0.1, float(summary_interval_seconds))
        _process_log_last_summary_at = monotonic()
        if process_log_path is not None:
            _process_log_path = process_log_path
            _process_log_dir_ready = False


def clear_process_log_context() -> None:
    """Bersihkan context session dan kembalikan path default process log."""

    global _process_log_context
    global _process_log_include_hot_path
    global _process_log_summary_interval_seconds
    global _process_log_last_summary_at
    global _process_log_path
    global _process_log_dir_ready

    with _LOCK:
        _flush_process_log_summaries_locked(force=True)
        _process_log_context = {}
        _process_log_include_hot_path = False
        _process_log_summary_interval_seconds = 5.0
        _process_log_last_summary_at = 0.0
        _process_log_path = PROCESS_LOG_PATH
        _process_log_dir_ready = False


def flush_process_log_summaries() -> None:
    """Tulis ringkasan hot-path yang masih tertahan."""

    with _LOCK:
        _flush_process_log_summaries_locked(force=True)

# event log untuk proses capture, VAD, queue, WebSocket, dan transkripsi. Ditulis ke logs/process.log dalam format JSONL.
def log_process_event(event: str, **fields: Any) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    try:
        # gunakan lock agar penulisan log thread-safe
        with _LOCK:
            safe_fields = {key: _safe_value(value) for key, value in fields.items()}
            if event in _HOT_PATH_EVENTS and not _process_log_include_hot_path:
                _aggregate_hot_path_event(event, now, safe_fields)
                _flush_process_log_summaries_locked(force=False)
                return

            record = {
                "ts": now,
                **{key: _safe_value(value) for key, value in _process_log_context.items()},
                "event": event,
                **safe_fields,
            }
            _write_process_record_locked(record)

    except Exception as exc:  # pragma: no cover - logging must never break capture
        _LOG.debug("failed to write process log event=%s error=%s", event, exc)


def _aggregate_hot_path_event(event: str, ts: str, fields: dict[str, Any]) -> None:
    key = (
        event,
        str(fields.get("source") or ""),
        str(fields.get("reason") or ""),
    )
    summary = _process_log_summaries.setdefault(
        key,
        {
            "event": event,
            "source": fields.get("source"),
            "reason": fields.get("reason"),
            "count": 0,
            "first_ts": ts,
            "last_ts": ts,
            "last": {},
            "metrics": {},
        },
    )
    summary["count"] = int(summary["count"]) + 1
    summary["last_ts"] = ts
    summary["last"] = {
        key: fields[key]
        for key in (
            "passed",
            "queue_size",
            "chunks_sent",
            "chunks_buffered",
            "chunks_enqueued",
            "sent",
            "buffered",
            "dropped",
            "final_drain",
            "final_partial",
        )
        if key in fields
    }

    metrics = summary["metrics"]
    for metric_name in (
        "input_rms_db",
        "output_rms_db",
        "input_rms",
        "duration_seconds",
        "queue_size",
    ):
        value = fields.get(metric_name)
        if not isinstance(value, (int, float)):
            continue
        metric = metrics.setdefault(
            metric_name,
            {"min": float(value), "max": float(value), "sum": 0.0, "count": 0},
        )
        metric["min"] = min(float(metric["min"]), float(value))
        metric["max"] = max(float(metric["max"]), float(value))
        metric["sum"] = float(metric["sum"]) + float(value)
        metric["count"] = int(metric["count"]) + 1


def _flush_process_log_summaries_locked(*, force: bool) -> None:
    global _process_log_last_summary_at

    if not _process_log_summaries:
        return
    now = monotonic()
    if not force and now - _process_log_last_summary_at < _process_log_summary_interval_seconds:
        return

    summaries: list[dict[str, Any]] = []
    for summary in _process_log_summaries.values():
        item = {
            key: value
            for key, value in summary.items()
            if key != "metrics" and value is not None
        }
        metrics: dict[str, dict[str, Any]] = {}
        for name, metric in summary["metrics"].items():
            count = int(metric["count"])
            if count <= 0:
                continue
            metrics[name] = {
                "min": round(float(metric["min"]), 6),
                "max": round(float(metric["max"]), 6),
                "avg": round(float(metric["sum"]) / count, 6),
            }
        if metrics:
            item["metrics"] = metrics
        summaries.append(item)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        **{key: _safe_value(value) for key, value in _process_log_context.items()},
        "event": "client.hot_path_summary",
        "summary_interval_seconds": round(max(0.0, now - _process_log_last_summary_at), 3),
        "summaries": summaries,
    }
    _process_log_summaries.clear()
    _process_log_last_summary_at = now
    _write_process_record_locked(record)


def _write_process_record_locked(record: dict[str, Any]) -> None:
    global _process_log_dir_ready

    if not _process_log_dir_ready:
        _process_log_path.parent.mkdir(parents=True, exist_ok=True)
        _process_log_dir_ready = True

    _rotate_process_log_if_needed_locked()
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with _process_log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _rotate_process_log_if_needed_locked() -> None:
    if PROCESS_LOG_BACKUP_COUNT <= 0 or PROCESS_LOG_MAX_BYTES <= 0:
        return
    try:
        if not _process_log_path.exists() or _process_log_path.stat().st_size < PROCESS_LOG_MAX_BYTES:
            return
        oldest = _process_log_path.with_name(f"{_process_log_path.name}.{PROCESS_LOG_BACKUP_COUNT}")
        if oldest.exists():
            oldest.unlink()
        for index in range(PROCESS_LOG_BACKUP_COUNT - 1, 0, -1):
            source = _process_log_path.with_name(f"{_process_log_path.name}.{index}")
            target = _process_log_path.with_name(f"{_process_log_path.name}.{index + 1}")
            if source.exists():
                source.replace(target)
        _process_log_path.replace(_process_log_path.with_name(f"{_process_log_path.name}.1"))
    except OSError as exc:
        _LOG.debug("failed to rotate process log path=%s error=%s", _process_log_path, exc)

# karena dapat menerima tipe data apa saja, fungsi ini untuk mengubah nilai arbitrary (nilai yang bisa berupa tipe data apa saja, termasuk tipe data yang tidak bisa langsung di-serialize ke JSON, seperti objek custom, Path, atau float NaN/Infinity.) menjadi bentuk JSON-safe dan batasi teks panjang.
def _safe_value(value: Any) -> Any:
    # Jika nilai adalah float dan tidak finite (NaN, Infinity, -Infinity), ubah menjadi string.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    # Jika nilai adalah None, bool, int, float, atau str, kembalikan nilai tersebut.
    if value is None or isinstance(value, (bool, int, float, str)):
        # Jika nilai adalah string dan panjangnya melebihi batas, potong dan tambahkan indikator truncation.
        if isinstance(value, str) and len(value) > _TEXT_LIMIT:
            return value[:_TEXT_LIMIT] + "...[truncated]"
        return value
    # Jika nilai adalah Path, ubah menjadi string.
    if isinstance(value, Path):
        return str(value)
    # Jika nilai adalah dict, ubah setiap key menjadi string dan rekursif memanggil _safe_value untuk setiap item.
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    # Jika nilai adalah list, tuple, atau set, ubah setiap item menjadi bentuk JSON-safe secara rekursif.
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    return str(value)


# 3. Transcript Log (persisten, JSONL append-only)
# TranscriptLog menyimpan backup transcript append-only setelah setiap result.
@dataclass(slots=True)
class TranscriptLog:
    path: Path
    entries: list[dict[str, object]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    # Membaca transcript log lama untuk mode resume/append.
    # Format utama sekarang JSONL append-only, tetapi JSON list lama tetap diterima
    # agar upgrade aplikasi tidak merusak file transcript yang sudah ada.
    @classmethod
    def load(cls, path: Path) -> "TranscriptLog":
        return cls(path=path, entries=load_transcript_entries(path))

    # Tambahkan satu hasil transcript dan langsung simpan ke disk (O(1)).
    def append_result(
        self,
        result: TranscriptionResult,
        *,
        label: str | None = None,
        display: str | None = None,
        completed: bool = True,
        stability: str | None = None,
        reliability_score: float | None = None,
        reliability_action: str | None = None,
    ) -> dict[str, object]:
        line = result_to_line(result) # ubah result menjadi TranscriptLine untuk memformat display
        
        # entry dict yang berisi informasi lengkap
        entry: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": result.source,
            "label": label or line.label,
            "completed": completed,
            "stability": stability or ("stable" if completed else "candidate"),
            "start_seconds": result.start_seconds,
            "end_seconds": result.end_seconds,
            "duration_seconds": result.duration_seconds,
            "text": result.text,
            "display": display or format_transcript_line(line),
            "model_name": result.model_name,
            "language": result.language,
            "segments": [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in result.segments
            ],
        }

        # reliability_score atau reliability_action diberikan, tambahkan ke entry
        if reliability_score is not None:
            entry["reliability_score"] = reliability_score
        if reliability_action is not None:
            entry["reliability_action"] = reliability_action

        self.entries.append(entry)
        self._append_line(entry)
        return entry

    # Tambahkan satu baris JSON ke akhir file tanpa menulis ulang isi lain.
    def _append_line(self, entry: dict[str, object]) -> None:
        target_path = self.path.resolve()  
        target_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(entry, ensure_ascii=False)

        with self._lock:
            with target_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    # tulis seluruh log ke disk (overwrite). Gunakan ini untuk menyimpan snapshot lengkap.
    def save(self) -> Path:
        target_path = self.path.resolve() # mengubah path menjadi absolute path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # tulis ke file sementara, lalu pindahkan ke target_path untuk menghindari file korup jika proses berhenti saat menulis.
        payload = "".join(
            json.dumps(entry, ensure_ascii=False) + "\n" for entry in self.entries
        )

        # NamedTemporaryFile untuk membuat file sementara di direktori yang sama dengan target_path. File sementara ini akan dihapus secara otomatis saat ditutup, kecuali jika delete=False.
        with self._lock:
            # NamedTemporaryFile untuk menulis payload ke file sementara. File ini akan dibuat di direktori yang sama dengan target_path agar bisa dipindahkan dengan os.replace. 
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=str(target_path.parent),
                prefix=f".{target_path.name}.",
                suffix=".tmp",
            ) as tmp_file:
                tmp_file.write(payload)
                tmp_path = Path(tmp_file.name)
 
            try:
                # Pindahkan file sementara ke target_path. os.replace akan menggantikan file target jika sudah ada, sehingga aman untuk overwrite.
                os.replace(str(tmp_path), str(target_path))
            except PermissionError:
                target_path.write_text(payload, encoding="utf-8")
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return self.path


def load_transcript_entries(path: Path) -> list[dict[str, object]]:
    """Baca transcript dari JSONL baru atau JSON list lama.

    Fungsi ini dipakai oleh runtime dan UI agar semua komponen memakai parser
    yang sama. Baris JSONL yang rusak dilewati karena biasanya hanya terjadi
    jika proses berhenti ketika sedang menulis baris terakhir.
    """

    if not path.exists():
        return []

    raw_payload = path.read_text(encoding="utf-8").strip()
    if not raw_payload:
        return []

    if raw_payload.startswith("["):
        payload = json.loads(raw_payload)
        if not isinstance(payload, list):
            raise ValueError("transcript log JSON list is invalid")
        return [entry for entry in payload if isinstance(entry, dict)]

    entries: list[dict[str, object]] = []
    for raw_line in raw_payload.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries
