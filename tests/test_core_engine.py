from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from src.core import engine
from src.core.engine import (
    STATE,
    STOP_DRAIN_MARGIN_SECONDS,
    UiOptions,
    _await_stop_in_progress,
    archive_transcript,
    audio_devices_payload,
    build_live_command,
    stop_live,
    transcript_payload,
)
from src.utils.os_detector import AudioBackend
from src.utils.status_ipc import format_status_line


class _FakeStdin:
    """Pipe stdin tiruan; merekam perintah yang ditulis parent ke subprocess."""

    def __init__(self, *, fail: bool = False) -> None:
        self.writes: list[str] = []
        self.flushed = False
        self._fail = fail

    def write(self, data: str) -> int:
        if self._fail:
            raise OSError("broken pipe")
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushed = True


class _FakeProcess:
    """Popen tiruan deterministik untuk menguji jalur stop tanpa proses nyata."""

    def __init__(
        self,
        *,
        pid: int = 4321,
        exit_on_wait: bool = True,
        exit_after_terminate: bool = True,
        stdin: "_FakeStdin | None" = -1,  # type: ignore[assignment]
    ) -> None:
        self.pid = pid
        # Popen selalu punya atribut stdin (pipe atau None); tiruan harus setia.
        self.stdin = _FakeStdin() if stdin == -1 else stdin
        self._alive = True
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []
        self._exit_on_wait = exit_on_wait
        self._exit_after_terminate = exit_after_terminate

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._exit_on_wait or not self._alive:
            self._alive = False
            return 0
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    def terminate(self) -> None:
        self.terminated = True
        if self._exit_after_terminate:
            self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def send_signal(self, sig: int) -> None:  # POSIX path
        pass


@pytest.fixture
def reset_state(monkeypatch):
    """Isolasi singleton STATE dan cegah sinyal OS nyata pada pid tiruan."""
    monkeypatch.setattr(engine.os, "kill", lambda pid, sig: None)
    with STATE.lock:
        STATE.process = None
        STATE.stop_requested = False
        STATE.final_drain_seconds = 10.0
        STATE.connection_status = "DISCONNECTED"
        STATE.audio_levels.clear()
    yield
    with STATE.lock:
        STATE.process = None
        STATE.stop_requested = False


def test_stop_live_waits_full_drain_budget_and_avoids_terminate(reset_state) -> None:
    proc = _FakeProcess(exit_on_wait=True)
    with STATE.lock:
        STATE.process = proc
        STATE.final_drain_seconds = 10.0

    stop_live()

    # Timeout tunggu diturunkan dari final_drain_seconds, bukan konstanta 3s lama.
    assert proc.wait_timeouts[0] == pytest.approx(10.0 + STOP_DRAIN_MARGIN_SECONDS)
    # Proses exit graceful dalam anggaran => tidak ada hard kill.
    assert proc.terminated is False
    assert proc.killed is False


# Regresi: pada aplikasi paket (--windows-console-mode=disable) tidak ada console,
# sehingga CTRL_BREAK_EVENT gagal dan stop selalu jatuh ke TerminateProcess —
# finalisasi (END_OF_AUDIO + flush merger) tak pernah jalan dan transcript
# penutup hilang. Stop WAJIB lewat kanal stdin lebih dulu.
def test_stop_live_requests_graceful_stop_via_stdin(reset_state, monkeypatch) -> None:
    from src.utils.status_ipc import parse_command_line

    kills: list[int] = []
    monkeypatch.setattr(engine.os, "kill", lambda pid, sig: kills.append(sig))

    proc = _FakeProcess(exit_on_wait=True)
    with STATE.lock:
        STATE.process = proc

    stop_live()

    assert proc.stdin is not None
    assert proc.stdin.writes, "perintah stop tidak dikirim lewat stdin"
    assert parse_command_line(proc.stdin.writes[-1])[0] == "stop"
    assert proc.stdin.flushed is True
    # stdin berhasil => tidak perlu console control event, dan tidak ada hard kill.
    assert kills == []
    assert proc.terminated is False
    assert proc.killed is False


def test_stop_live_falls_back_to_console_signal_when_stdin_missing(reset_state, monkeypatch) -> None:
    kills: list[int] = []
    monkeypatch.setattr(engine.os, "kill", lambda pid, sig: kills.append(sig))

    proc = _FakeProcess(exit_on_wait=True, stdin=None)
    with STATE.lock:
        STATE.process = proc

    stop_live()

    if os.name == "nt":
        assert kills, "fallback sinyal console tidak dijalankan"
    assert proc.terminated is False


def test_stop_live_falls_back_when_stdin_write_fails(reset_state, monkeypatch) -> None:
    kills: list[int] = []
    monkeypatch.setattr(engine.os, "kill", lambda pid, sig: kills.append(sig))

    proc = _FakeProcess(exit_on_wait=True, stdin=_FakeStdin(fail=True))
    with STATE.lock:
        STATE.process = proc

    stop_live()

    if os.name == "nt":
        assert kills, "pipe rusak harus jatuh ke sinyal console"
    assert proc.terminated is False


def test_stop_live_terminates_hung_process_as_backstop(reset_state) -> None:
    proc = _FakeProcess(exit_on_wait=False, exit_after_terminate=True)
    with STATE.lock:
        STATE.process = proc

    stop_live(wait_timeout_seconds=0.05)

    # Proses menggantung melewati anggaran => backstop terminate() tetap terpicu.
    assert proc.terminated is True


def test_await_stop_in_progress_returns_after_previous_exit(reset_state) -> None:
    proc = _FakeProcess(exit_on_wait=False)
    with STATE.lock:
        STATE.process = proc
        STATE.stop_requested = True
        STATE.final_drain_seconds = 5.0

    def _finish_shutdown() -> None:
        time.sleep(0.1)
        proc._alive = False

    threading.Thread(target=_finish_shutdown, daemon=True).start()
    started = time.time()
    _await_stop_in_progress()
    elapsed = time.time() - started

    # Menunggu hingga proses lama exit, jauh di bawah deadline (5 + margin).
    assert proc.poll() is not None
    assert elapsed < 2.0


def test_await_stop_in_progress_no_wait_when_not_stopping(reset_state) -> None:
    proc = _FakeProcess(exit_on_wait=False)
    with STATE.lock:
        STATE.process = proc
        STATE.stop_requested = False  # sesi aktif, bukan sedang di-stop

    started = time.time()
    _await_stop_in_progress()
    elapsed = time.time() - started

    # Tanpa permintaan stop, guard tidak menahan start (semantik lama dipertahankan).
    assert elapsed < 0.2


def test_monitor_line_ignores_prose_and_transcript_keywords(reset_state) -> None:
    from src.core.engine import _handle_monitor_line

    _handle_monitor_line(format_status_line("mic", "SERVER_READY"))
    assert STATE.connection_status == "CONNECTED"

    # BUG-002: baris transcript/prosa yang memuat kata kunci TIDAK boleh mengubah status.
    _handle_monitor_line("[00:00:01 - 00:00:02] [Me] tidak ada error di laporan\n")
    assert STATE.connection_status == "CONNECTED"

    _handle_monitor_line("[MIC] websocket connected\n")
    assert STATE.connection_status == "CONNECTED"

    _handle_monitor_line("[00:00:03 - 00:00:04] [Me] koneksi sempat disconnected tadi\n")
    assert STATE.connection_status == "CONNECTED"

    # Hanya sinyal terstruktur yang menggerakkan status.
    _handle_monitor_line(format_status_line("mic", "CLIENT_RECV_ERROR"))
    assert STATE.connection_status == "ERROR"


def test_monitor_line_maps_lifecycle_codes(reset_state) -> None:
    from src.core.engine import _handle_monitor_line

    for code, expected in [
        ("CLIENT_CONNECTING", "CONNECTING"),
        ("SERVER_READY", "CONNECTED"),
        ("CLIENT_REMOTE_CLOSED", "DISCONNECTED"),
        ("CLIENT_RECONNECTED", "CONNECTED"),
    ]:
        _handle_monitor_line(format_status_line("speaker", code))
        assert STATE.connection_status == expected

    # Kode di luar tabel policy tidak mengubah status terakhir.
    _handle_monitor_line(format_status_line("speaker", "CLIENT_AUDIO_FINISHED"))
    assert STATE.connection_status == "CONNECTED"


def test_set_mute_writes_command_without_restarting_session(reset_state) -> None:
    from src.core.engine import set_mute
    from src.utils.status_ipc import parse_command_line

    class _StdinPipe:
        def __init__(self) -> None:
            self.written: list[str] = []

        def write(self, data: str) -> None:
            self.written.append(data)

        def flush(self) -> None:
            pass

    proc = _FakeProcess(exit_on_wait=False)
    proc.stdin = _StdinPipe()  # type: ignore[attr-defined]
    with STATE.lock:
        STATE.process = proc

    assert set_mute("mic", True) is True
    cmd, payload = parse_command_line(proc.stdin.written[0])  # type: ignore[attr-defined]
    assert cmd == "set_mute"
    assert payload == {"cmd": "set_mute", "source": "mic", "muted": True}
    # Sesi tidak di-restart / tidak dimatikan oleh mute.
    assert proc.terminated is False
    assert proc.poll() is None


def test_set_mute_returns_false_without_active_session(reset_state) -> None:
    from src.core.engine import set_mute

    assert set_mute("mic", True) is False


def test_monitor_line_updates_audio_levels_and_skips_log(reset_state) -> None:
    from src.core.engine import _handle_monitor_line, audio_levels
    from src.utils.status_ipc import format_level_line

    logs_before = len(STATE.logs())
    _handle_monitor_line(format_level_line("mic", -22.5))
    _handle_monitor_line(format_level_line("speaker", -35.0))

    levels = audio_levels()
    assert levels["mic"] == -22.5
    assert levels["speaker"] == -35.0
    # Baris level tidak boleh membanjiri log (frekuensi tinggi).
    assert len(STATE.logs()) == logs_before


def test_monitor_process_sets_disconnected_and_exit_code_on_exit(reset_state) -> None:
    class _FakeStreamProcess:
        def __init__(self, lines: list[str]) -> None:
            self.stdout = iter(lines)

        def wait(self, timeout: float | None = None) -> int:
            return 7

    proc = _FakeStreamProcess([format_status_line("mic", "SERVER_READY")])
    engine._monitor_process(proc)  # type: ignore[arg-type]

    # finally selalu menutup ke DISCONNECTED dan mencatat exit code.
    assert STATE.connection_status == "DISCONNECTED"
    assert STATE.exit_code == 7


def test_build_live_command_uses_python_module_and_transcript_log(tmp_path: Path) -> None:
    transcript_log = tmp_path / "meeting.json"
    command = build_live_command(
        UiOptions(
            host="localhost",
            port=9090,
            source="speaker",
            model="medium",
            chunk_seconds=0.5,
            mic_device=3,
            speaker_device="Headset",
            mic_server_vad=True,
            speaker_server_vad=False,
            vad_threshold=0.5,
            no_speech_thresh=0.45,
            local_agreement=True,
            local_agreement_window_seconds=15.0,
            local_agreement_hop_seconds=2.0,
            dynamic_prompt=True,
            rolling_audio_archive=True,
            rolling_audio_segment_seconds=30.0,
            process_log_hot_path_detail=True,
            process_log_summary_interval_seconds=7.0,
            transcript_log=transcript_log,
            hide_partials=True,
        )
    )

    assert "-m" in command
    assert "src.app" in command
    assert "--mode" in command
    assert "live" in command
    assert "--source" in command
    assert "speaker" in command
    assert "--whisper-model" in command
    assert "medium" in command
    assert "--live-chunk-seconds" in command
    assert "0.5" in command
    assert "--mic-device" in command
    assert "3" in command
    assert "--speaker-device" in command
    assert "Headset" in command
    assert "--mic-server-vad" in command
    assert "--no-speaker-server-vad" in command
    assert "--transcript-log" in command
    assert str(transcript_log) in command
    assert "--hide-partials" in command
    assert "--local-agreement" in command
    assert "--dynamic-prompt" in command
    assert "--speech-boundary-detection" in command
    assert "--speech-boundary-silence-seconds" in command
    assert "--local-agreement-window-seconds" in command
    assert "--process-log-hot-path-detail" in command
    assert "--process-log-summary-interval-seconds" in command
    assert "--rolling-audio-archive" in command
    assert "--rolling-audio-dir" in command
    assert "--rolling-audio-segment-seconds" in command
    assert "30" in command
    assert "7" in command


def test_build_live_command_round_trips_through_main_argparse() -> None:
    """Guard drift (F2/M2): setiap flag yang di-emit build_live_command harus
    diterima argparse main.py. Bila engine meng-emit flag yang tak dikenal
    (rename/typo), parse_args akan SystemExit dan test ini gagal."""
    from src.main import parse_args

    opts = UiOptions(
        host="server-x",
        port=9091,
        source="mic",
        model="medium",
        language="en",
        chunk_seconds=0.25,
        mic_server_vad=False,
        speaker_server_vad=True,
        vad_threshold=0.4,
        no_speech_thresh=0.5,
        final_drain_seconds=8.0,
        auto_reconnect=False,
        local_agreement=False,
        dynamic_prompt=False,
        speech_boundary_detection=False,
        debug_chunk_archive=True,
        rolling_audio_archive=True,
        process_log_hot_path_detail=True,
        hide_partials=False,
        mic_device=3,
        speaker_device="Headset",
    )
    command = build_live_command(opts)
    argv = command[command.index("src.app") + 1:]

    args = parse_args(argv)  # tidak boleh SystemExit

    # Round-trip beberapa nilai kunci untuk memastikan pemetaan benar, bukan hanya lolos parse.
    assert args.mode == "live"
    assert args.source == "mic"
    assert args.server_host == "server-x"
    assert args.server_port == 9091
    assert args.whisper_model == "medium"
    assert args.whisper_language == "en"
    assert args.live_chunk_seconds == 0.25
    assert args.vad_threshold == 0.4
    # M2: domain adaptation (prompt/hotwords) HARUS diteruskan ke subprocess agar
    # paket tanpa .env tetap terkonfigurasi penuh.
    assert "--initial-prompt" in command
    assert "--hotwords" in command


def test_build_live_command_forwards_prompt_and_hotwords() -> None:
    from src.main import parse_args

    opts = UiOptions(initial_prompt="rapat PLN Batam", hotwords="gardu, trafo, meteran")
    command = build_live_command(opts)
    argv = command[command.index("src.app") + 1:]
    args = parse_args(argv)
    assert args.initial_prompt == "rapat PLN Batam"
    assert args.hotwords == "gardu, trafo, meteran"


def test_transcript_payload_counts_sources(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text(
        json.dumps(
            [
                {"source": "mic", "text": "halo"},
                {"source": "speaker", "text": "selamat pagi"},
                {"source": "speaker", "text": "lanjut"},
            ]
        ),
        encoding="utf-8",
    )

    payload = transcript_payload(path)

    assert payload["count"] == 3
    assert payload["counts"]["mic"] == 1
    assert payload["counts"]["speaker"] == 2
    assert payload["exists"] is True


def test_transcript_payload_counts_jsonl_sources(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text(
        "\n".join(
            [
                json.dumps({"source": "mic", "text": "halo"}),
                json.dumps({"source": "speaker", "text": "selamat pagi"}),
            ]
        ),
        encoding="utf-8",
    )

    payload = transcript_payload(path)

    assert payload["count"] == 2
    assert payload["counts"]["mic"] == 1
    assert payload["counts"]["speaker"] == 1
    assert payload["error"] is None


def test_transcript_payload_keeps_candidates_in_audit_not_default_entries(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text(
        "\n".join(
            [
                json.dumps({"source": "speaker", "text": "draft", "completed": False, "stability": "candidate"}),
                json.dumps({"source": "speaker", "text": "final", "completed": True, "stability": "stable"}),
            ]
        ),
        encoding="utf-8",
    )

    payload = transcript_payload(path)

    assert payload["count"] == 1
    assert payload["entries"][0]["text"] == "final"
    assert payload["audit_count"] == 2
    assert payload["quality"]["candidate"] == 1
    assert payload["quality"]["stable"] == 1
    assert payload["quality"]["stable_ratio"] == 0.5


def test_archive_transcript_moves_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text("[]", encoding="utf-8")

    archived = archive_transcript(path)

    assert archived is not None
    assert archived.exists()
    assert not path.exists()


def test_audio_devices_payload_marks_linux_speaker_as_deferred(monkeypatch) -> None:
    monkeypatch.setattr("src.core.engine.get_audio_backend", lambda: AudioBackend.SOUNDDEVICE_INPUT)
    monkeypatch.setattr("src.core.engine.list_input_devices", lambda include_system_aliases=False, concise=False: [])

    payload = audio_devices_payload()

    assert payload["speaker"] == []
    assert payload["diagnostics"]["speaker_capture"]["supported"] is False
    assert payload["diagnostics"]["speaker_capture"]["status"] == "deferred"
