"""Penegakan versi sisi client (W4): handshake, deteksi penolakan, state terminal."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from src.core import engine
from src.preprocessing.core import PreprocessedAudioChunk
from src.version import __version__
from src.whisper.client_base import (
    OutdatedClientError,
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
    WhisperLiveStreamClient,
)
from src.whisper.client_reconnect import _BufferedReconnectClient
from src.whisper.models import ReconnectPolicy
from src.utils.status_ipc import format_status_line


class _RejectingWebSocket:
    """Server yang menolak versi: kirim OUTDATED_CLIENT lalu tutup (seperti server.py)."""

    def __init__(self, min_version: str = "1.2.0") -> None:
        self.sent_text: list[str] = []
        self.closed = False
        self._min_version = min_version
        self._replied = False

    def send(self, message: str) -> None:
        self.sent_text.append(message)

    def send_binary(self, message: bytes) -> None:
        pass

    def recv(self) -> str:
        if self._replied:
            raise RuntimeError("connection closed by server")
        self._replied = True
        uid = json.loads(self.sent_text[0])["uid"]
        return json.dumps({
            "uid": uid,
            "status": "ERROR",
            "code": "OUTDATED_CLIENT",
            "min_version": self._min_version,
            "message": f"Client version 0.1.0 is older than the minimum required version {self._min_version}.",
        })

    def settimeout(self, timeout: float | None) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _chunk() -> PreprocessedAudioChunk:
    return PreprocessedAudioChunk(
        source="mic",
        samples=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        start_seconds=0.0,
        duration_seconds=0.1,
        rms_db=-12.0,
    )


# Handshake --------------------------------------------------------------
def test_handshake_sends_client_version() -> None:
    fake = _RejectingWebSocket()
    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile(), ready_timeout=0.2),
        websocket_factory=lambda url, timeout, header: fake,
        on_status=lambda source, message: None,
        uid="mic-test",
    )
    with pytest.raises(OutdatedClientError):
        client.connect()
    assert json.loads(fake.sent_text[0])["client_version"] == __version__


def test_server_rejection_is_raised_as_outdated_client_status() -> None:
    fake = _RejectingWebSocket(min_version="9.9.9")
    statuses: list[dict] = []
    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile(), ready_timeout=0.2),
        websocket_factory=lambda url, timeout, header: fake,
        on_status=lambda source, message: statuses.append(message),
        uid="mic-test",
    )
    with pytest.raises(OutdatedClientError):
        client.connect()

    outdated = [m for m in statuses if m["status"] == "OUTDATED_CLIENT"]
    assert len(outdated) == 1
    assert outdated[0]["min_version"] == "9.9.9"


def test_rejection_fails_fast_without_waiting_for_ready_timeout() -> None:
    # Penolakan datang seketika; menunggu ready_timeout (produksi: 300s) berarti
    # pengguna menatap "Connecting..." selama 5 menit tanpa alasan.
    fake = _RejectingWebSocket()
    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile(), ready_timeout=30.0),
        websocket_factory=lambda url, timeout, header: fake,
        on_status=lambda source, message: None,
        uid="mic-test",
    )
    started = time.perf_counter()
    with pytest.raises(OutdatedClientError):
        client.connect()
    assert time.perf_counter() - started < 2.0
    assert client.is_ready is False  # dibangunkan, tetapi tidak pernah "siap"


# State terminal ---------------------------------------------------------
def test_version_rejection_stops_reconnect_attempts(monkeypatch) -> None:
    # Penolakan versi bersifat permanen: mencoba ulang hanya menghantam server.
    import src.whisper.client_reconnect as reconnect_module

    attempts: list[int] = []

    def factory(url, timeout, header):
        attempts.append(1)
        return _RejectingWebSocket()

    original_client = reconnect_module.WhisperLiveStreamClient

    class RejectedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["websocket_factory"] = factory
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(reconnect_module, "WhisperLiveStreamClient", RejectedClient)

    wrapper = _BufferedReconnectClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile(), ready_timeout=0.2),
        policy=ReconnectPolicy(enabled=True, initial_backoff_seconds=0.0, buffer_seconds=5.0),
        on_transcript=lambda source, segments, message: None,
        on_status=lambda source, message: None,
    )
    wrapper.connect_initial()
    assert attempts == [1]  # satu percobaan, lalu berhenti selamanya
    assert wrapper._fatal_status == "OUTDATED_CLIENT"

    # Chunk berikutnya tidak boleh memicu koneksi baru.
    for _ in range(5):
        wrapper.send(_chunk())
    assert attempts == [1]


def test_transient_disconnect_still_reconnects() -> None:
    # Regresi: state terminal tidak boleh membekukan reconnect biasa.
    wrapper = _BufferedReconnectClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile()),
        policy=ReconnectPolicy(enabled=True),
        on_transcript=lambda source, segments, message: None,
        on_status=lambda source, message: None,
    )
    wrapper._handle_status("mic", {"status": "CLIENT_REMOTE_CLOSED"})
    assert wrapper._fatal_status is None


# Jalur engine -----------------------------------------------------------
def test_engine_records_outdated_client_details_and_error_state() -> None:
    engine.STATE.outdated_client = None
    engine._handle_monitor_line(format_status_line("mic", "OUTDATED_CLIENT", min_version="2.0.0"))

    info = engine.outdated_client_info()
    assert info == {"client_version": __version__, "min_version": "2.0.0"}
    assert engine.connection_status() == "ERROR"
    engine.STATE.outdated_client = None


def test_engine_outdated_info_is_none_by_default() -> None:
    engine.STATE.outdated_client = None
    assert engine.outdated_client_info() is None
