from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.capture.mic_stream import list_input_devices
from src.capture.win_loopback import list_soundcard_loopback_devices
from src.utils.logging import load_transcript_entries
from src.utils.os_detector import AudioBackend, get_audio_backend
from src.app import is_frozen
from src import paths
from src.utils.status_ipc import format_command_line, parse_level_line, parse_status_line
from src.version import app_version


ROOT_DIR = Path(__file__).resolve().parents[2]
# Transkrip & log HARUS di direktori data per-user yang persisten, BUKAN relatif
# terhadap __file__: pada Nuitka onefile, __file__ ada di direktori ekstraksi temp
# yang dihapus saat keluar (data hilang). Lihat src/paths.py.
DEFAULT_TRANSCRIPT_LOG = paths.current_transcript_log()
DEFAULT_LOG_FILE = paths.logs_dir() / "transcriber.log"

# Margin di atas final_drain_seconds subprocess saat menunggu graceful stop.
# Menutup batas atas deterministik yang berjalan seri dengan drain: join reader
# thread (<=3s/thread), flush reconnect buffer (<=5s), plus slack. Lihat BUG-001.
STOP_DRAIN_MARGIN_SECONDS = 12.0

# Policy pemetaan kode status subprocess -> state koneksi UI (BUG-002).
# Kode berasal dari handle_status/reconnect client di subprocess; state adalah
# 4 nilai yang divalidasi set_connection_status. Kode di luar tabel ini
# (mis. CLIENT_AUDIO_FINISHED, CLIENT_CLOSING, WAIT) tidak mengubah status.
_CONNECTION_STATE_BY_CODE: dict[str, str] = {
    "CLIENT_CONNECTING": "CONNECTING",
    "CLIENT_CONNECTING_RETRY": "CONNECTING",
    "CLIENT_RECONNECTING": "CONNECTING",
    "CLIENT_SOCKET_OPEN": "CONNECTING",
    "CLIENT_OPTIONS_SENT": "CONNECTING",
    "CLIENT_WAITING_SERVER_READY": "CONNECTING",
    "SERVER_READY": "CONNECTED",
    "CLIENT_CONNECTED_READY": "CONNECTED",
    "CLIENT_RECONNECTED": "CONNECTED",
    "CLIENT_RECV_ERROR": "ERROR",
    "CLIENT_READY_TIMEOUT": "ERROR",
    "OUTDATED_CLIENT": "ERROR",
    "CLIENT_TLS_ERROR": "ERROR",
    "CLIENT_MIC_ERROR": "ERROR",
    "ERROR": "ERROR",
    "DISCONNECT": "DISCONNECTED",
    "CLIENT_REMOTE_CLOSED": "DISCONNECTED",
    "CLIENT_RECONNECT_FAILED": "DISCONNECTED",
    "CLIENT_CLOSED": "DISCONNECTED",
}


@dataclass(slots=True)
class UiOptions:
    """Options for starting a live transcription subprocess."""

    host: str | None = None
    port: int | None = None
    # W2: TLS (wss://) untuk sesi live. Diisi dari Config Provider; produksi wajib TRUE.
    use_tls: bool = False
    source: str = "both"
    model: str = "medium"
    language: str = "id"
    # Domain adaptation dari config provider (di-pass ke subprocess agar paket
    # tanpa .env tetap berkualitas). Tidak di-edit di UI pada milestone ini.
    initial_prompt: str = ""
    hotwords: str = ""
    chunk_seconds: float = 0.5
    mic_device: int | str | None = None
    speaker_device: int | str | None = None
    mic_server_vad: bool = True
    speaker_server_vad: bool = False
    mic_webrtc_ns: bool = False
    mic_target_rms_db: float = -20.0
    mic_max_normalization_gain_db: float = 18.0
    speaker_target_rms_db: float = -23.0
    speaker_max_normalization_gain_db: float = 18.0
    vad_threshold: float = 0.55
    no_speech_thresh: float = 0.75
    ready_timeout: float = 300.0
    final_drain_seconds: float = 10.0
    auto_reconnect: bool = True
    reconnect_initial_backoff_seconds: float = 1.0
    reconnect_max_backoff_seconds: float = 30.0
    reconnect_buffer_seconds: float = 30.0
    local_agreement: bool = True
    local_agreement_window_seconds: float = 20.0
    local_agreement_hop_seconds: float = 3.0
    dynamic_prompt: bool = True
    speech_boundary_detection: bool = True
    speech_boundary_silence_seconds: float = 0.8
    speech_boundary_max_wait_seconds: float = 5.0
    debug_chunk_archive: bool = False
    rolling_audio_archive: bool = False
    rolling_audio_segment_seconds: float = 60.0
    log_level: str = "INFO"
    process_log_hot_path_detail: bool = False
    process_log_summary_interval_seconds: float = 5.0
    hide_partials: bool = True
    reset_transcript: bool = False
    transcript_log: Path = DEFAULT_TRANSCRIPT_LOG


@dataclass
class LiveProcessState:
    """State for the live transcription subprocess."""

    process: subprocess.Popen[str] | None = None
    started_at: float | None = None
    command: list[str] = field(default_factory=list)
    transcript_log: Path = DEFAULT_TRANSCRIPT_LOG
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=2_000))
    exit_code: int | None = None
    stop_requested: bool = False
    last_error: str | None = None
    connection_status: str = "DISCONNECTED"
    # Anggaran drain sesi aktif; dipakai stop_live untuk menurunkan timeout tunggu.
    final_drain_seconds: float = 10.0
    # Level audio nyata per source (RMS dB), diperbarui dari sinyal level subprocess.
    audio_levels: dict[str, float] = field(default_factory=dict)
    # W4: detail penolakan versi oleh server (None = tidak ada). Bertahan setelah
    # sesi berhenti agar GUI sempat membacanya lewat polling.
    outdated_client: dict | None = None
    # W2: detail kegagalan verifikasi TLS (None = tidak ada), pola sama dengan di atas.
    tls_error: dict | None = None
    # W3: detail kegagalan membuka mikrofon di subprocess (None = tidak ada).
    mic_error: dict | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def running(self) -> bool:
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def get_connection_status(self) -> str:
        with self.lock:
            return self.connection_status

    def set_connection_status(self, status: str) -> None:
        if status not in ("CONNECTING", "CONNECTED", "ERROR", "DISCONNECTED"):
            raise ValueError(f"Invalid status: {status}")
        with self.lock:
            self.connection_status = status

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            elapsed = None if self.started_at is None else round(time.time() - self.started_at, 1)
            return {
                "running": running,
                "connection_status": self.connection_status,
                "pid": None if self.process is None else self.process.pid,
                "elapsed_seconds": elapsed,
                "command": self.command,
                "exit_code": self.exit_code,
                "stop_requested": self.stop_requested,
                "last_error": self.last_error,
                "transcript_log": str(self.transcript_log.relative_to(ROOT_DIR))
                if self.transcript_log.is_relative_to(ROOT_DIR)
                else str(self.transcript_log),
            }

    def append_log(self, line: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.log_lines.append(f"{timestamp} {line.rstrip()}")

    def logs(self, limit: int = 300) -> list[str]:
        with self.lock:
            return list(self.log_lines)[-limit:]


STATE = LiveProcessState()


def app_executable() -> str:
    """Executable yang dipakai untuk men-spawn ulang aplikasi ini (jalur live).

    JANGAN memakai `sys.executable` pada build frozen. Pada Nuitka **onefile**,
    `sys.executable` menunjuk ke `<dir-ekstraksi-temp>\\python.exe` — berkas yang
    TIDAK PERNAH ADA (diverifikasi: `os.path.exists` -> False, dan direktori
    ekstraksi memang tidak memuat satu pun .exe). Men-spawn path itu membuat
    CreateProcess gagal dengan `WinError 2: The system cannot find the file
    specified`. Path exe yang sebenarnya berada di `sys.argv[0]`.

    Dev (tidak frozen) tetap memakai `sys.executable` — di sana nilainya benar.
    """
    if not is_frozen():
        return sys.executable

    for candidate in (sys.argv[0], sys.executable):
        if not candidate:
            continue
        resolved = Path(candidate).resolve()
        if resolved.is_file():
            return str(resolved)

    # Guard: gagal dengan diagnosis yang dapat ditindaklanjuti, bukan WinError 2 mentah.
    raise RuntimeError(
        "Tidak dapat menemukan executable aplikasi untuk memulai sesi. "
        f"sys.argv[0]={sys.argv[0]!r}, sys.executable={sys.executable!r}. "
        "Jalankan aplikasi dari lokasi instalasinya, lalu coba lagi."
    )


def build_live_command(options: UiOptions) -> list[str]:
    # Packaged (Nuitka): exe aplikasi men-dispatch pada args -> "exe --mode live ..."
    # (Nuitka tak mendukung `-m modul`; percobaan itu diblokir guard self-execution).
    # Dev: lewat dispatcher tunggal -> "python -m src.app --mode live ...".
    executable = app_executable()
    command = [executable] if is_frozen() else [executable, "-m", "src.app"]
    command += [
        "--mode",
        "live",
        "--source",
        options.source,
    ]
    if options.host is not None:
        command.extend(["--server-host", options.host])
    if options.port is not None:
        command.extend(["--server-port", str(options.port)])
    if options.use_tls:
        # W2: menutup celah — tanpa ini GUI tidak pernah dapat memakai wss://.
        command.append("--server-wss")


    command.extend([
        "--server-ready-timeout",
        _num(options.ready_timeout),
        "--whisper-model",
        options.model,
        "--whisper-language",
        options.language,
        "--initial-prompt",
        options.initial_prompt or "",
        "--hotwords",
        options.hotwords or "",
        "--live-chunk-seconds",
        _num(options.chunk_seconds),
        "--mic-server-vad" if options.mic_server_vad else "--no-mic-server-vad",
        "--speaker-server-vad" if options.speaker_server_vad else "--no-speaker-server-vad",
        "--mic-target-rms-db",
        _num(options.mic_target_rms_db),
        "--mic-max-normalization-gain-db",
        _num(options.mic_max_normalization_gain_db),
        "--speaker-target-rms-db",
        _num(options.speaker_target_rms_db),
        "--speaker-max-normalization-gain-db",
        _num(options.speaker_max_normalization_gain_db),
        "--vad-threshold",
        _num(options.vad_threshold),
        "--whisperlive-no-speech-thresh",
        _num(options.no_speech_thresh),
        "--final-drain-seconds",
        _num(options.final_drain_seconds),
        "--auto-reconnect" if options.auto_reconnect else "--no-auto-reconnect",
        "--reconnect-initial-backoff-seconds",
        _num(options.reconnect_initial_backoff_seconds),
        "--reconnect-max-backoff-seconds",
        _num(options.reconnect_max_backoff_seconds),
        "--reconnect-buffer-seconds",
        _num(options.reconnect_buffer_seconds),
        "--local-agreement-window-seconds",
        _num(options.local_agreement_window_seconds),
        "--local-agreement-hop-seconds",
        _num(options.local_agreement_hop_seconds),
        "--speech-boundary-silence-seconds",
        _num(options.speech_boundary_silence_seconds),
        "--speech-boundary-max-wait-seconds",
        _num(options.speech_boundary_max_wait_seconds),
        "--transcript-log",
        str(options.transcript_log),
        "--log-level",
        options.log_level,
        "--log-file",
        str(DEFAULT_LOG_FILE),
        "--rolling-audio-segment-seconds",
        _num(options.rolling_audio_segment_seconds),
        "--process-log-summary-interval-seconds",
        _num(options.process_log_summary_interval_seconds),
    ])
    if options.mic_device is not None:
        command.extend(["--mic-device", str(options.mic_device)])
    if options.speaker_device is not None:
        command.extend(["--speaker-device", str(options.speaker_device)])
    if options.hide_partials:
        command.append("--hide-partials")
    command.append("--local-agreement" if options.local_agreement else "--no-local-agreement")
    command.append("--dynamic-prompt" if options.dynamic_prompt else "--no-dynamic-prompt")
    command.append("--speech-boundary-detection" if options.speech_boundary_detection else "--no-speech-boundary-detection")
    if options.debug_chunk_archive:
        command.append("--debug-chunk-archive")
    if options.rolling_audio_archive:
        command.extend(["--rolling-audio-archive", "--rolling-audio-dir", str(ROOT_DIR / "audio" / "rolling")])
    if options.process_log_hot_path_detail:
        command.append("--process-log-hot-path-detail")
    return command


def start_live(options: UiOptions) -> dict[str, Any]:
    _await_stop_in_progress()
    with STATE.lock:
        if STATE.process is not None and STATE.process.poll() is None:
            raise RuntimeError("live session is already running")

        if options.reset_transcript:
            archive_transcript(options.transcript_log)

        options.transcript_log.parent.mkdir(parents=True, exist_ok=True)
        paths.ensure_dir(DEFAULT_LOG_FILE.parent)  # subprocess menulis --log-file ke sini
        command = build_live_command(options)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdin=subprocess.PIPE,  # kanal kontrol masuk (true mute) tanpa restart sesi
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=creationflags,
        )
        STATE.process = process
        STATE.started_at = time.time()
        STATE.command = command
        STATE.transcript_log = options.transcript_log
        STATE.final_drain_seconds = options.final_drain_seconds
        STATE.exit_code = None
        STATE.stop_requested = False
        STATE.last_error = None
        STATE.connection_status = "CONNECTING"
        STATE.audio_levels.clear()
        STATE.outdated_client = None
        STATE.tls_error = None
        STATE.mic_error = None
        STATE.log_lines.clear()
        STATE.append_log("UI started live client")

    thread = threading.Thread(target=_monitor_process, args=(process,), name="core-live-monitor", daemon=True)
    thread.start()
    return STATE.snapshot()


def stop_live(*, force: bool = False, wait_timeout_seconds: float | None = None) -> dict[str, Any]:
    with STATE.lock:
        process = STATE.process
        if process is None or process.poll() is not None:
            STATE.connection_status = "DISCONNECTED"
            return STATE.snapshot()
        STATE.stop_requested = True
        STATE.connection_status = "DISCONNECTED"
        STATE.append_log("UI stop requested")
        if wait_timeout_seconds is None:
            # Turunkan anggaran tunggu dari final_drain_seconds subprocess, bukan
            # konstanta pendek. Subprocess butuh >= final_drain_seconds untuk
            # menerima hasil akhir server (END_OF_AUDIO) dan menuliskan flush
            # merger terakhir; terminate() dini akan membuang transcript penutup.
            wait_timeout_seconds = STATE.final_drain_seconds + STOP_DRAIN_MARGIN_SECONDS

    if force:
        process.terminate()
        return STATE.snapshot()

    try:
        if os.name == "nt":
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            process.send_signal(signal.SIGINT)
    except Exception as exc:
        STATE.append_log(f"graceful stop failed, terminating process: {exc}")
        process.terminate()

    try:
        process.wait(timeout=wait_timeout_seconds)
        STATE.append_log(f"Process stopped gracefully (waited {wait_timeout_seconds}s)")
    except subprocess.TimeoutExpired:
        STATE.append_log(f"Process cleanup timeout after {wait_timeout_seconds}s, terminating forcefully")
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            STATE.append_log("Force terminate timed out, killing process")
            process.kill()

    return STATE.snapshot()


def _await_stop_in_progress() -> None:
    """Tunggu sesi sebelumnya yang sedang di-stop hingga prosesnya benar-benar exit.

    Konsekuensi dari graceful stop yang lebih panjang (BUG-001): proses lama bisa
    masih hidup selama drain. Tanpa penantian ini, restart cepat (toggle mute atau
    Start tepat setelah Stop) gagal dengan "live session is already running".
    Hanya menunggu bila stop memang sedang berjalan; sesi aktif tanpa permintaan
    stop tetap dibiarkan menolak start seperti semula.
    """
    with STATE.lock:
        process = STATE.process
        stopping = STATE.stop_requested
        budget = STATE.final_drain_seconds + STOP_DRAIN_MARGIN_SECONDS
    if process is None or not stopping:
        return
    deadline = time.time() + budget
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.05)


def connection_status() -> str:
    """Status koneksi sesi live saat ini (CONNECTING/CONNECTED/ERROR/DISCONNECTED).

    Accessor tingkat-modul agar konsumen (GUI) tidak perlu menyentuh singleton
    STATE secara langsung. Dipelihara oleh _monitor_process dari sinyal stdout
    subprocess (lihat BUG-002).
    """
    return STATE.get_connection_status()


def outdated_client_info() -> dict | None:
    """Detail penolakan versi oleh server, atau None (W4).

    Berisi `client_version` (versi terpasang) dan `min_version` (minimum server).
    Dikonsumsi GUI untuk menampilkan panduan unduh; di-reset saat sesi baru dimulai.
    """
    with STATE.lock:
        return dict(STATE.outdated_client) if STATE.outdated_client else None


def tls_error_info() -> dict | None:
    """Detail kegagalan verifikasi TLS, atau None (W2).

    Berisi `reason` (sebab yang dapat ditindaklanjuti) dan `url` server.
    Di-reset saat sesi baru dimulai.
    """
    with STATE.lock:
        return dict(STATE.tls_error) if STATE.tls_error else None


def mic_error_info() -> dict | None:
    """Detail kegagalan membuka mikrofon di sesi live, atau None (W3)."""
    with STATE.lock:
        return dict(STATE.mic_error) if STATE.mic_error else None


def set_mute(source: str, muted: bool) -> bool:
    """Mute/unmute satu source pada sesi live yang sedang berjalan (true mute).

    Mengirim perintah lewat stdin subprocess sehingga koneksi WebSocket dan state
    ASR server (termasuk local-agreement) tetap dipertahankan — tidak ada restart
    sesi. Mengembalikan False bila tidak ada sesi aktif atau pipe tidak tersedia.
    """
    with STATE.lock:
        process = STATE.process
        if process is None or process.poll() is not None or process.stdin is None:
            return False
        try:
            process.stdin.write(format_command_line("set_mute", source=source, muted=muted))
            process.stdin.flush()
        except (OSError, ValueError) as exc:
            STATE.append_log(f"set_mute({source}) failed: {exc}")
            return False
        STATE.append_log(f"Mute {source}: {muted}")
        return True


def audio_levels() -> dict[str, float]:
    """Level audio nyata per source (RMS dB) untuk indikator GUI real-time.

    Dikosongkan saat tidak ada sesi berjalan. Diperbarui _monitor_process dari
    sinyal level subprocess.
    """
    with STATE.lock:
        return dict(STATE.audio_levels)


def archive_transcript(path: Path = DEFAULT_TRANSCRIPT_LOG) -> Path | None:
    path = path.resolve()
    if not path.exists():
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = archive_dir / f"{path.stem}.{stamp}{path.suffix}"
    shutil.move(str(path), str(target))
    return target


def archive_transcript_to_history(path: Path = DEFAULT_TRANSCRIPT_LOG) -> Path | None:
    """Arsipkan transkrip sesi ke folder history (satu berkas per rapat, bertanggal).

    Berkas `current` berformat JSONL (satu entri per baris); di-parse dengan parser
    resmi lalu ditulis ke history sebagai `{"entries": [...]}` yang self-describing,
    seragam dengan yang dibaca list_history/SummaryWindow. Hanya entri final
    (completed) yang disimpan — sama dengan yang tampil pada live view/export.
    Transkrip kosong tidak diarsipkan.
    """
    path = path.resolve()
    if not path.exists():
        return None
    try:
        entries = [e for e in load_transcript_entries(path) if isinstance(e, dict)]
    except Exception:
        return None
    final = [e for e in entries if bool(e.get("completed", True))]
    if not final:
        return None  # jangan arsipkan sesi tanpa transkrip final

    hist = paths.ensure_dir(paths.history_dir())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = hist / f"transcript-{stamp}.json"
    # Hindari menimpa arsip lain bila dua sesi berakhir pada detik yang sama.
    suffix = 1
    while target.exists():
        target = hist / f"transcript-{stamp}-{suffix}.json"
        suffix += 1
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "entries": final}
    try:
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return None
    return target


def list_history() -> list[dict[str, Any]]:
    """Daftar riwayat transkrip (terbaru dulu) dari folder history."""
    hist = paths.history_dir()
    if not hist.exists():
        return []
    items: list[dict[str, Any]] = []
    for file in hist.glob("transcript-*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            entries = data.get("entries", []) if isinstance(data, dict) else []
        except (OSError, ValueError):
            entries = []
        preview = ""
        for entry in entries:
            text = str(entry.get("text") or "").strip() if isinstance(entry, dict) else ""
            if text:
                preview = text
                break
        items.append({
            "path": str(file),
            "name": file.stem,
            "mtime": file.stat().st_mtime,
            "entry_count": len(entries),
            "preview": preview,
        })
    items.sort(key=lambda it: it["mtime"], reverse=True)
    return items


def transcript_payload(path: Path = DEFAULT_TRANSCRIPT_LOG) -> dict[str, Any]:
    path = path.resolve()
    entries: list[dict[str, Any]] = []
    error = None
    if path.exists():
        try:
            entries = [entry for entry in load_transcript_entries(path) if isinstance(entry, dict)]
        except Exception as exc:
            error = str(exc)

    final_entries = [entry for entry in entries if bool(entry.get("completed", True))]
    counts = {"mic": 0, "speaker": 0, "other": 0}
    stability_counts = {"stable": 0, "candidate": 0, "other": 0}
    for entry in entries:
        source = str(entry.get("source") or "other")
        counts[source if source in counts else "other"] += 1
        stability = str(entry.get("stability") or ("stable" if bool(entry.get("completed", True)) else "candidate"))
        stability_counts[stability if stability in stability_counts else "other"] += 1

    final_counts = {"mic": 0, "speaker": 0, "other": 0}
    for entry in final_entries:
        source = str(entry.get("source") or "other")
        final_counts[source if source in final_counts else "other"] += 1

    stable = stability_counts["stable"]
    candidate = stability_counts["candidate"]
    total_quality = stable + candidate

    return {
        "path": str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path),
        "exists": path.exists(),
        "entries": final_entries,
        "count": len(final_entries),
        "counts": final_counts,
        "audit_entries": entries,
        "audit_count": len(entries),
        "audit_counts": counts,
        "quality": {
            "stable": stable,
            "candidate": candidate,
            "stable_ratio": round(stable / total_quality, 3) if total_quality else None,
            "stability_counts": stability_counts,
        },
        "error": error,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
    }


def audio_devices_payload() -> dict[str, Any]:
    backend = get_audio_backend()
    result: dict[str, Any] = {
        "mic": [],
        "speaker": [],
        "errors": {},
        "diagnostics": {
            "audio_backend": backend.value,
            "speaker_capture": _speaker_capture_status(backend),
        },
    }
    try:
        raw_mic_devices = list_input_devices(include_system_aliases=True)
        mic_devices = list_input_devices(concise=True)
        result["diagnostics"]["raw_mic_count"] = len(raw_mic_devices)
        result["diagnostics"]["mic_count"] = len(mic_devices)
        result["mic"] = [
            {
                "id": str(device.get("index")),
                "index": device.get("index"),
                "name": str(device.get("name", f"device-{device.get('index')}")),
                "label": str(device.get("label") or device.get("name", f"device-{device.get('index')}")),
                "hostapi": str(device.get("hostapi_name", "")),
                "is_default": bool(device.get("is_default", False)),
                "channels": int(device.get("max_input_channels", 0)),
                "sample_rate": int(float(device.get("default_samplerate", 0) or 0)),
            }
            for device in mic_devices
        ]
    except Exception as exc:
        result["errors"]["mic"] = str(exc)
    if backend is AudioBackend.WASAPI_LOOPBACK:
        try:
            result["speaker"] = [
                {
                    "id": str(device.index),
                    "index": device.index,
                    "name": device.name,
                    "channels": device.channels,
                }
                for device in list_soundcard_loopback_devices()
            ]
        except Exception as exc:
            result["errors"]["speaker"] = str(exc)
    return result


def _speaker_capture_status(backend: AudioBackend) -> dict[str, Any]:
    if backend is AudioBackend.WASAPI_LOOPBACK:
        return {
            "supported": True,
            "status": "supported",
            "message": "Windows WASAPI loopback is available for speaker capture.",
        }
    if backend is AudioBackend.SCREENCAPTUREKIT:
        return {
            "supported": False,
            "status": "experimental",
            "message": "macOS speaker capture boundary exists, but native ScreenCaptureKit wiring is not production-ready.",
        }
    if backend is AudioBackend.SOUNDDEVICE_INPUT:
        return {
            "supported": False,
            "status": "deferred",
            "message": "Linux speaker capture is deferred until PipeWire/PulseAudio monitor capture is validated.",
        }
    return {
        "supported": False,
        "status": "unsupported",
        "message": "Speaker/system-audio capture is unsupported on this platform.",
    }


def _handle_monitor_line(line: str) -> None:
    """Proses satu baris stdout subprocess.

    Status koneksi diambil HANYA dari baris sinyal terstruktur (BUG-002);
    prosa/transcript tidak pernah discan kata kunci sehingga ucapan yang memuat
    kata seperti "error" tidak lagi memicu status palsu.
    """
    parsed = parse_status_line(line)
    if parsed is not None:
        source, code, details = parsed
        if code == "OUTDATED_CLIENT":
            # W4: simpan detail agar GUI dapat menampilkan versi minimum + URL unduh.
            with STATE.lock:
                STATE.outdated_client = {
                    "client_version": app_version(),
                    "min_version": str(details.get("min_version") or ""),
                }
        elif code == "CLIENT_MIC_ERROR":
            with STATE.lock:
                STATE.mic_error = {"reason": str(details.get("reason") or "")}
        elif code == "CLIENT_TLS_ERROR":
            # W2: simpan sebab agar GUI menjelaskan kegagalan koneksi aman.
            with STATE.lock:
                STATE.tls_error = {
                    "reason": str(details.get("reason") or ""),
                    "url": str(details.get("url") or ""),
                }
        ui_state = _CONNECTION_STATE_BY_CODE.get(code)
        if ui_state is not None:
            STATE.set_connection_status(ui_state)
            STATE.append_log(f"Connection status: {ui_state} ({source}:{code})")
        return
    level = parse_level_line(line)
    if level is not None:
        source, rms_db = level
        with STATE.lock:
            STATE.audio_levels[source] = rms_db
        return  # baris level tidak di-log (frekuensi tinggi)
    STATE.append_log(line)


def _monitor_process(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        raise RuntimeError("live client stdout is not captured")

    try:
        for line in process.stdout:
            _handle_monitor_line(line)
    finally:
        exit_code = process.wait()
        with STATE.lock:
            STATE.exit_code = exit_code
            STATE.connection_status = "DISCONNECTED"
            STATE.audio_levels.clear()
            STATE.append_log(f"live client exited with code {exit_code}")


def _safe_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _choice(value: str, allowed: set[str], default: str) -> str:
    return value if value in allowed else default


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _device_value(value: object) -> int | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)
