"""Small local web UI for controlling the live transcriber client."""

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
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_TRANSCRIPT_LOG = ROOT_DIR / "audio" / "transcript_log.json"
DEFAULT_LOG_FILE = ROOT_DIR / "logs" / "transcriber.log"


@dataclass(slots=True)
class UiOptions:
    host: str = "localhost"
    port: int = 9090
    metrics_port: int = 9091
    source: str = "both"
    model: str = "large-v3-turbo"
    language: str = "id"
    vad_threshold: float = 0.55
    no_speech_thresh: float = 0.45
    ready_timeout: float = 300.0
    final_drain_seconds: float = 8.0
    local_agreement: bool = True
    local_agreement_window_seconds: float = 15.0
    local_agreement_hop_seconds: float = 2.0
    dynamic_prompt: bool = True
    log_level: str = "INFO"
    hide_partials: bool = True
    reset_transcript: bool = False
    transcript_log: Path = DEFAULT_TRANSCRIPT_LOG


@dataclass
class LiveProcessState:
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
    """Build the argv used by the UI to start the CLI client."""

    command = [
        sys.executable,
        "-m",
        "src.main",
        "--live",
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
        "--vad-threshold",
        _num(options.vad_threshold),
        "--whisperlive-no-speech-thresh",
        _num(options.no_speech_thresh),
        "--final-drain-seconds",
        _num(options.final_drain_seconds),
        "--local-agreement-window-seconds",
        _num(options.local_agreement_window_seconds),
        "--local-agreement-hop-seconds",
        _num(options.local_agreement_hop_seconds),
        "--transcript-log",
        str(options.transcript_log),
        "--log-level",
        options.log_level,
        "--log-file",
        str(DEFAULT_LOG_FILE),
    ]
    if options.hide_partials:
        command.append("--hide-partials")
    command.append("--local-agreement" if options.local_agreement else "--no-local-agreement")
    command.append("--dynamic-prompt" if options.dynamic_prompt else "--no-dynamic-prompt")
    return command


def start_live(options: UiOptions) -> dict[str, Any]:
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
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                entries = [entry for entry in payload if isinstance(entry, dict)]
            else:
                error = "transcript file is not a JSON list"
        except Exception as exc:
            error = str(exc)

    counts = {"mic": 0, "speaker": 0, "other": 0}
    for entry in entries:
        source = str(entry.get("source") or "other")
        counts[source if source in counts else "other"] += 1

    return {
        "path": str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path),
        "exists": path.exists(),
        "entries": entries,
        "count": len(entries),
        "counts": counts,
        "error": error,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
    }


def server_health(host: str = "localhost", metrics_port: int = 9091) -> dict[str, Any]:
    url = f"http://{host}:{metrics_port}/metrics"
    try:
        with urlopen(url, timeout=1.2) as response:
            healthy = response.status == 200
            return {"healthy": healthy, "url": url, "status": response.status, "error": None}
    except (OSError, URLError) as exc:
        return {"healthy": False, "url": url, "status": None, "error": str(exc)}


class UiRequestHandler(SimpleHTTPRequestHandler):
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
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _parse_options(payload: dict[str, Any]) -> UiOptions:
    transcript_log = _safe_path(str(payload.get("transcriptLog") or DEFAULT_TRANSCRIPT_LOG))
    return UiOptions(
        host=str(payload.get("host") or "localhost"),
        port=_int(payload.get("port"), 9090),
        metrics_port=_int(payload.get("metricsPort"), 9091),
        source=_choice(str(payload.get("source") or "both"), {"mic", "speaker", "both"}, "both"),
        model=str(payload.get("model") or "large-v3-turbo"),
        language=str(payload.get("language") or "id"),
        vad_threshold=_float(payload.get("vadThreshold"), 0.55),
        no_speech_thresh=_float(payload.get("noSpeechThresh"), 0.45),
        ready_timeout=_float(payload.get("readyTimeout"), 300.0),
        final_drain_seconds=_float(payload.get("finalDrainSeconds"), 8.0),
        local_agreement=bool(payload.get("localAgreement", True)),
        local_agreement_window_seconds=_float(payload.get("localAgreementWindowSeconds"), 15.0),
        local_agreement_hop_seconds=_float(payload.get("localAgreementHopSeconds"), 2.0),
        dynamic_prompt=bool(payload.get("dynamicPrompt", True)),
        log_level=_choice(str(payload.get("logLevel") or "INFO").upper(), {"DEBUG", "INFO", "WARNING", "ERROR"}, "INFO"),
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
