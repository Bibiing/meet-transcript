from __future__ import annotations

import json
import time

import numpy as np

from src.engine.preprocessing import PreprocessedAudioChunk
from src.engine.whisperlive_client import (
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
    WhisperLiveStreamClient,
)
from src.engine.whisperlive_session import _events_from_segments, _results_from_segments


class FakeWebSocket:
    def __init__(self, messages: list[str] | None = None) -> None:
        self.sent_text: list[str] = []
        self.sent_binary: list[bytes] = []
        self.closed = False
        self.timeouts: list[float | None] = []
        self._messages = messages or [
            json.dumps({"uid": "mic-test", "message": "SERVER_READY", "backend": "faster_whisper"})
        ]

    def send(self, message: str) -> None:
        self.sent_text.append(message)

    def send_binary(self, message: bytes) -> None:
        self.sent_binary.append(message)

    def recv(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        raise RuntimeError("done")

    def settimeout(self, timeout: float | None) -> None:
        self.timeouts.append(timeout)

    def close(self) -> None:
        self.closed = True


def test_whisperlive_client_sends_compatible_initial_options() -> None:
    fake = FakeWebSocket()
    statuses: list[tuple[str, dict]] = []

    def factory(url, timeout, header):
        assert url == "ws://localhost:9090"
        assert timeout == 10.0
        assert header == []
        return fake

    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile()),
        websocket_factory=factory,
        on_status=lambda source, message: statuses.append((source, message)),
        uid="mic-test",
    )

    client.connect()
    options = json.loads(fake.sent_text[0])
    client.close()

    assert options["uid"] == "mic-test"
    assert options["source"] == "mic"
    assert options["model"] == "small"
    assert options["language"] == "id"
    assert options["audio_format"] == "int16"
    assert options["use_vad"] is False
    assert options["vad_parameters"]["threshold"] == 0.55
    assert options["no_speech_thresh"] == 0.75
    assert options["no_speech_threshold"] == 0.75
    assert options["log_prob_threshold"] == -1.2
    assert options["local_agreement"] is True
    assert options["local_agreement_window_seconds"] == 20.0
    assert options["local_agreement_hop_seconds"] == 3.0
    assert options["dynamic_prompt"] is True
    assert options["speech_boundary_detection"] is True
    assert options["speech_boundary_silence_seconds"] == 0.8
    assert options["speech_boundary_max_wait_seconds"] == 5.0
    assert "Indonesian meeting transcript" in options["initial_prompt"]
    assert "do not translate to English" in options["initial_prompt"]
    assert "Do not add information" in options["initial_prompt"]
    assert "PLN" in options["hotwords"]
    assert "format" not in options
    assert fake.timeouts == [None]
    assert [message["status"] for _, message in statuses[:4]] == [
        "CLIENT_CONNECTING",
        "CLIENT_SOCKET_OPEN",
        "CLIENT_OPTIONS_SENT",
        "SERVER_READY",
    ]


def test_whisperlive_client_sends_int16_pcm_chunk_by_default() -> None:
    fake = FakeWebSocket()

    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile()),
        websocket_factory=lambda *args, **kwargs: fake,
        uid="mic-test",
    )
    client.connect()
    chunk = PreprocessedAudioChunk(
        source="mic",
        samples=np.array([0.0, 0.5, -0.5], dtype=np.float32),
        sample_rate=16_000,
        start_seconds=0.0,
        duration_seconds=3 / 16_000,
        rms_db=-12.0,
    )

    client.send_chunk(chunk)
    client.close()

    assert np.frombuffer(fake.sent_binary[0], dtype=np.int16).tolist() == [0, 16384, -16384]


def test_whisperlive_client_can_send_float32_pcm_chunk() -> None:
    fake = FakeWebSocket()

    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile(), audio_format="float32"),
        websocket_factory=lambda *args, **kwargs: fake,
        uid="mic-test",
    )
    client.connect()
    chunk = PreprocessedAudioChunk(
        source="mic",
        samples=np.array([0.0, 0.5, -0.5], dtype=np.float32),
        sample_rate=16_000,
        start_seconds=0.0,
        duration_seconds=3 / 16_000,
        rms_db=-12.0,
    )

    client.send_chunk(chunk)
    client.close()

    assert np.frombuffer(fake.sent_binary[0], dtype=np.float32).tolist() == [0.0, 0.5, -0.5]


def test_whisperlive_client_reports_remote_close() -> None:
    fake = FakeWebSocket(
        [
            json.dumps({"uid": "mic-test", "message": "SERVER_READY", "backend": "faster_whisper"}),
            "",
        ]
    )
    statuses: list[dict] = []

    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile()),
        websocket_factory=lambda *args, **kwargs: fake,
        on_status=lambda source, message: statuses.append(message),
        uid="mic-test",
    )

    client.connect()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not any(message["status"] == "CLIENT_REMOTE_CLOSED" for message in statuses):
        time.sleep(0.01)
    client.close()

    assert any(message["status"] == "CLIENT_REMOTE_CLOSED" for message in statuses)


def test_results_from_segments_preserves_source_label() -> None:
    results = _results_from_segments(
        "speaker",
        "large-v3-turbo",
        "id",
        [{"start": "1.000", "end": "2.500", "text": "halo dari meeting"}],
    )

    assert len(results) == 1
    assert results[0].source == "speaker"
    assert results[0].text == "halo dari meeting"
    assert results[0].duration_seconds == 1.5


def test_events_from_segments_preserves_completed_flag() -> None:
    events = _events_from_segments(
        "mic",
        "large-v3-turbo",
        "id",
        [{"start": "0.000", "end": "1.000", "text": "sedang bicara", "completed": False}],
    )

    assert len(events) == 1
    assert events[0].result.source == "mic"
    assert events[0].completed is False
