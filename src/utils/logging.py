from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from src.engine.whisper import TranscriptionResult
from src.utils.formatter import format_transcript_line, result_to_line

__all__ = [
    "configure_logging",
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

    # file handler untuk menulis log ke file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")  # handler menulis log
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
_TEXT_LIMIT = 500

_process_log_dir_ready = False  # cache: direktori logs/ hanya perlu dibuat sekali per proses

# event log untuk proses capture, VAD, queue, WebSocket, dan transkripsi. Ditulis ke logs/process.log dalam format JSONL.
def log_process_event(event: str, **fields: Any) -> None:
    global _process_log_dir_ready

    # buat record log dengan timestamp, event, dan fields tambahan. Fields akan diubah menjadi bentuk JSON-safe.
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        **{key: _safe_value(value) for key, value in fields.items()},
    }

    try:
        # buat direktori logs/ jika belum ada
        if not _process_log_dir_ready:
            PROCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _process_log_dir_ready = True
        # tulis record log ke file dalam format JSONL
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

        # gunakan lock agar penulisan log thread-safe
        with _LOCK:
            with PROCESS_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                
    except Exception as exc:  # pragma: no cover - logging must never break capture
        _LOG.debug("failed to write process log event=%s error=%s", event, exc)

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
