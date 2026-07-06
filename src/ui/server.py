"""Server web UI lokal untuk mengontrol client live transcriber.

Server ini bukan server ASR. Ia hanya menyajikan HTML/JS lokal, endpoint status,
daftar device audio, transcript log, dan menjalankan `python -m src.main --mode live`
sebagai subprocess.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

from src.capture.mic_stream import list_input_devices
from src.capture.win_loopback import list_soundcard_loopback_devices
from src.utils.logging import load_transcript_entries
from src.utils.os_detector import AudioBackend, get_audio_backend


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_TRANSCRIPT_LOG = ROOT_DIR / "audio" / "transcript_log.json"
DEFAULT_LOG_FILE = ROOT_DIR / "logs" / "transcriber.log"


@dataclass(slots=True)
class UiOptions:
    """Opsi start live session yang diterima dari form web UI."""

    host: str = "localhost"
    port: int = 9090
    source: str = "both"
    model: str = "small"
    language: str = "id"
    chunk_seconds: float = 0.5
    mic_device: int | str | None = None
    speaker_device: int | str | None = None
    mic_client_vad: bool = True
    speaker_client_vad: bool = False
    mic_vad_rms_threshold: float = 0.025
    mic_vad_peak_threshold: float = 0.08
    mic_vad_speech_fraction: float = 0.35
    mic_min_input_rms_db: float = -38.0
    mic_target_rms_db: float = -20.0
    mic_max_normalization_gain_db: float = 18.0
    speaker_vad_rms_threshold: float = 0.008
    speaker_vad_peak_threshold: float = 0.05
    speaker_vad_speech_fraction: float = 0.20
    speaker_min_input_rms_db: float = -48.0
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
    """State proses client live yang sedang dijalankan oleh web UI."""

    process: subprocess.Popen[str] | None = None
    started_at: float | None = None
    command: list[str] = field(default_factory=list)
    transcript_log: Path = DEFAULT_TRANSCRIPT_LOG
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=2_000))
    exit_code: int | None = None
    stop_requested: bool = False
    last_error: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def running(self) -> bool:
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            elapsed = None if self.started_at is None else round(time.time() - self.started_at, 1)
            return {
                "running": running,
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


def build_live_command(options: UiOptions) -> list[str]:
    """Bangun argv yang dipakai UI untuk menjalankan CLI client."""

    command = [
        sys.executable,
        "-m",
        "src.main",
        "--mode",
        "live",
        "--source",
        options.source,
        "--server-host",
        options.host,
        "--server-port",
        str(options.port),
        "--server-ready-timeout",
        _num(options.ready_timeout),
        "--whisper-model",
        options.model,
        "--whisper-language",
        options.language,
        "--live-chunk-seconds",
        _num(options.chunk_seconds),
        "--mic-client-vad" if options.mic_client_vad else "--no-mic-client-vad",
        "--speaker-client-vad" if options.speaker_client_vad else "--no-speaker-client-vad",
        "--mic-vad-rms-threshold",
        _num(options.mic_vad_rms_threshold),
        "--mic-vad-peak-threshold",
        _num(options.mic_vad_peak_threshold),
        "--mic-vad-speech-fraction",
        _num(options.mic_vad_speech_fraction),
        "--mic-min-input-rms-db",
        _num(options.mic_min_input_rms_db),
        "--mic-target-rms-db",
        _num(options.mic_target_rms_db),
        "--mic-max-normalization-gain-db",
        _num(options.mic_max_normalization_gain_db),
        "--speaker-vad-rms-threshold",
        _num(options.speaker_vad_rms_threshold),
        "--speaker-vad-peak-threshold",
        _num(options.speaker_vad_peak_threshold),
        "--speaker-vad-speech-fraction",
        _num(options.speaker_vad_speech_fraction),
        "--speaker-min-input-rms-db",
        _num(options.speaker_min_input_rms_db),
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
    ]
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
    """Start subprocess client live dan mulai thread monitor stdout."""
    with STATE.lock:
        if STATE.process is not None and STATE.process.poll() is None:
            raise RuntimeError("live session is already running")

        if options.reset_transcript:
            archive_transcript(options.transcript_log)

        options.transcript_log.parent.mkdir(parents=True, exist_ok=True)
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
        STATE.exit_code = None
        STATE.stop_requested = False
        STATE.last_error = None
        STATE.log_lines.clear()
        STATE.append_log("UI started live client")

    thread = threading.Thread(target=_monitor_process, args=(process,), name="ui-live-monitor", daemon=True)
    thread.start()
    return STATE.snapshot()


def stop_live(*, force: bool = False) -> dict[str, Any]:
    """Stop client live secara graceful, atau terminate jika force=True."""
    with STATE.lock:
        process = STATE.process
        if process is None or process.poll() is not None:
            return STATE.snapshot()
        STATE.stop_requested = True
        STATE.append_log("UI stop requested")

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
    return STATE.snapshot()


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


def server_health(host: str = "localhost", port: int = 9090) -> dict[str, Any]:
    address = f"{host}:{port}"
    try:
        with socket.create_connection((host, port), timeout=1.2):
            return {"healthy": True, "url": address, "status": "tcp_open", "error": None}
    except OSError as exc:
        return {"healthy": False, "url": address, "status": None, "error": str(exc)}


def audio_devices_payload() -> dict[str, Any]:
    """Payload daftar mic/speaker untuk dropdown UI."""
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


class UiRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler kecil untuk static UI dan API kontrol client."""

    server_version = "PLNTranscriberUI/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            self._json(
                {
                    "session": STATE.snapshot(),
                    "server": server_health(),
                    "transcript": transcript_payload(STATE.transcript_log),
                    "logs": STATE.logs(120),
                }
            )
            return
        if self.path == "/api/transcript":
            self._json(transcript_payload(STATE.transcript_log))
            return
        if self.path == "/api/logs":
            self._json({"lines": STATE.logs(500)})
            return
        if self.path == "/api/audio-devices":
            self._json(audio_devices_payload())
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/start":
                self._json(start_live(_parse_options(self._read_json())))
                return
            if self.path == "/api/stop":
                self._json(stop_live())
                return
            if self.path == "/api/force-stop":
                self._json(stop_live(force=True))
                return
            if self.path == "/api/archive-transcript":
                archived = archive_transcript(STATE.transcript_log)
                self._json({"archived": None if archived is None else str(archived)})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
        except Exception as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            pass  # Klien (browser) sudah menutup koneksi, kita abaikan saja error ini


def _parse_options(payload: dict[str, Any]) -> UiOptions:
    transcript_log = _safe_path(str(payload.get("transcriptLog") or DEFAULT_TRANSCRIPT_LOG))
    return UiOptions(
        host=str(payload.get("host") or "localhost"),
        port=_int(payload.get("port"), 9090),
        source=_choice(str(payload.get("source") or "both"), {"mic", "speaker", "both"}, "both"),
        model=str(payload.get("model") or "small"),
        language=str(payload.get("language") or "id"),
        chunk_seconds=_float(payload.get("chunkSeconds"), 0.5),
        mic_device=_device_value(payload.get("micDevice")),
        speaker_device=_device_value(payload.get("speakerDevice")),
        mic_client_vad=bool(payload.get("micClientVad", True)),
        speaker_client_vad=bool(payload.get("speakerClientVad", False)),
        mic_vad_rms_threshold=_float(payload.get("micVadRmsThreshold"), 0.025),
        mic_vad_peak_threshold=_float(payload.get("micVadPeakThreshold"), 0.08),
        mic_vad_speech_fraction=_float(payload.get("micVadSpeechFraction"), 0.35),
        mic_min_input_rms_db=_float(payload.get("micMinInputRmsDb"), -38.0),
        mic_target_rms_db=_float(payload.get("micTargetRmsDb"), -20.0),
        mic_max_normalization_gain_db=_float(payload.get("micMaxNormalizationGainDb"), 18.0),
        speaker_vad_rms_threshold=_float(payload.get("speakerVadRmsThreshold"), 0.008),
        speaker_vad_peak_threshold=_float(payload.get("speakerVadPeakThreshold"), 0.05),
        speaker_vad_speech_fraction=_float(payload.get("speakerVadSpeechFraction"), 0.20),
        speaker_min_input_rms_db=_float(payload.get("speakerMinInputRmsDb"), -48.0),
        speaker_target_rms_db=_float(payload.get("speakerTargetRmsDb"), -23.0),
        speaker_max_normalization_gain_db=_float(payload.get("speakerMaxNormalizationGainDb"), 18.0),
        vad_threshold=_float(payload.get("vadThreshold"), 0.55),
        no_speech_thresh=_float(payload.get("noSpeechThresh"), 0.75),
        ready_timeout=_float(payload.get("readyTimeout"), 300.0),
        final_drain_seconds=_float(payload.get("finalDrainSeconds"), 10.0),
        auto_reconnect=bool(payload.get("autoReconnect", True)),
        reconnect_initial_backoff_seconds=_float(payload.get("reconnectInitialBackoffSeconds"), 1.0),
        reconnect_max_backoff_seconds=_float(payload.get("reconnectMaxBackoffSeconds"), 30.0),
        reconnect_buffer_seconds=_float(payload.get("reconnectBufferSeconds"), 30.0),
        local_agreement=bool(payload.get("localAgreement", True)),
        local_agreement_window_seconds=_float(payload.get("localAgreementWindowSeconds"), 20.0),
        local_agreement_hop_seconds=_float(payload.get("localAgreementHopSeconds"), 3.0),
        dynamic_prompt=bool(payload.get("dynamicPrompt", True)),
        speech_boundary_detection=bool(payload.get("speechBoundaryDetection", True)),
        speech_boundary_silence_seconds=_float(payload.get("speechBoundarySilenceSeconds"), 0.8),
        speech_boundary_max_wait_seconds=_float(payload.get("speechBoundaryMaxWaitSeconds"), 5.0),
        debug_chunk_archive=bool(payload.get("debugChunkArchive", False)),
        rolling_audio_archive=bool(payload.get("rollingAudioArchive", False)),
        rolling_audio_segment_seconds=_float(payload.get("rollingAudioSegmentSeconds"), 60.0),
        log_level=_choice(str(payload.get("logLevel") or "INFO").upper(), {"DEBUG", "INFO", "WARNING", "ERROR"}, "INFO"),
        process_log_hot_path_detail=bool(payload.get("processLogHotPathDetail", False)),
        process_log_summary_interval_seconds=_float(payload.get("processLogSummaryIntervalSeconds"), 5.0),
        hide_partials=bool(payload.get("hidePartials", True)),
        reset_transcript=bool(payload.get("resetTranscript", False)),
        transcript_log=transcript_log,
    )


def _monitor_process(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    try:
        for line in process.stdout:
            STATE.append_log(line)
    finally:
        exit_code = process.wait()
        with STATE.lock:
            STATE.exit_code = exit_code
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local web UI for PLN Meeting Transcriber")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), UiRequestHandler)
    print(f"PLN Transcriber UI: http://{args.host}:{args.port}")
    print("Tekan Ctrl+C untuk berhenti.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_live(force=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
