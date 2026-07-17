from __future__ import annotations

import json
import time

import numpy as np

from src.preprocessing.core import PreprocessedAudioChunk
from src.whisper.client_base import (
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
    WhisperLiveStreamClient,
)
from src.whisper.session_utils import _events_from_segments, _results_from_segments
from src.whisper.client_reconnect import _BufferedReconnectClient
from src.whisper.models import ReconnectPolicy


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


class StableFakeWebSocket:
    def __init__(self, *, fail_binary_once: bool = False) -> None:
        self.fail_binary_once = fail_binary_once
        self.sent_text: list[str] = []
        self.sent_binary: list[bytes] = []
        self.closed = False
        self._ready_sent = False

    def send(self, message: str) -> None:
        self.sent_text.append(message)

    def send_binary(self, message: bytes) -> None:
        if self.fail_binary_once:
            self.fail_binary_once = False
            raise RuntimeError("simulated send failure")
        self.sent_binary.append(message)

    def recv(self) -> str:
        if not self._ready_sent:
            self._ready_sent = True
            uid = json.loads(self.sent_text[0])["uid"]
            return json.dumps({"uid": uid, "message": "SERVER_READY", "backend": "faster_whisper"})
        while not self.closed:
            time.sleep(0.01)
        return ""

    def settimeout(self, timeout: float | None) -> None:
        return None

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
    assert options["model"] == "medium"
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
    assert options["initial_prompt"] == ""
    assert options["hotwords"] == ""
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


def test_forward_transcript_drops_stale_epoch_segments() -> None:
    forwarded: list[tuple[str, list[dict], dict]] = []
    wrapper = _BufferedReconnectClient(
        "mic",
        WhisperLiveConnectionConfig(profile=WhisperLiveProfile()),
        policy=ReconnectPolicy(enabled=True),
        on_transcript=lambda source, segments, message: forwarded.append((source, segments, message)),
        on_status=lambda source, message: None,
    )
    # Anggap epoch koneksi saat ini = 2.
    wrapper._generation = 2

    # Segmen dari epoch usang (generation 1) harus di-drop.
    wrapper._forward_transcript(1, "mic", [{"start": 0.0, "end": 1.0, "text": "lama"}], {})
    # Segmen dari epoch terkini (generation 2) harus diteruskan.
    wrapper._forward_transcript(2, "mic", [{"start": 0.0, "end": 1.0, "text": "baru"}], {})

    assert len(forwarded) == 1
    assert forwarded[0][1][0]["text"] == "baru"


def test_reconnect_resets_timeline_before_new_epoch() -> None:
    sockets = [StableFakeWebSocket(), StableFakeWebSocket()]

    def factory(url, timeout, header):
        return sockets.pop(0)

    connection = WhisperLiveConnectionConfig(profile=WhisperLiveProfile(), ready_timeout=1.0)
    forwarded: list[tuple[str, list[dict], dict]] = []
    wrapper = _BufferedReconnectClient(
        "mic",
        connection,
        policy=ReconnectPolicy(enabled=True, initial_backoff_seconds=0.1, max_backoff_seconds=0.1, buffer_seconds=5.0),
        on_transcript=lambda source, segments, message: forwarded.append((source, segments, message)),
        on_status=lambda source, message: None,
    )

    import src.whisper.client_reconnect as reconnect_module

    original_client = reconnect_module.WhisperLiveStreamClient
    checkpoints_at_construction: list[int] = []

    class RecordingClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["websocket_factory"] = factory
            # Rekam jumlah checkpoint timeline saat client baru dikonstruksi:
            # bila reset mendahului connect, timeline sudah kosong di titik ini.
            checkpoints_at_construction.append(len(wrapper.sent_timeline._checkpoints))
            super().__init__(*args, **kwargs)

    reconnect_module.WhisperLiveStreamClient = RecordingClient
    try:
        wrapper.connect_initial()  # epoch 1
        assert wrapper._generation == 1
        gen_epoch1 = wrapper._generation

        # Simulasikan drift epoch lama: offset +100 pada timeline.
        wrapper.sent_timeline.record_sent(sent_duration=0.0, real_start=100.0)
        wrapper._total_sent_seconds = 50.0
        assert wrapper.sent_timeline.map_to_real(5.0) == 105.0

        wrapper._connect(force=True)  # epoch 2
        assert wrapper._generation == 2

        # Timeline sudah bersih SAAT client epoch 2 dikonstruksi (reset sebelum connect).
        assert checkpoints_at_construction[-1] == 0
        # Counter epoch baru mulai dari 0; offset epoch lama hilang.
        assert wrapper._total_sent_seconds == 0.0
        assert wrapper.sent_timeline.map_to_real(5.0) == 5.0

        # Segmen epoch lama di-drop; epoch terkini diteruskan.
        wrapper._forward_transcript(gen_epoch1, "mic", [{"start": 0.0, "end": 1.0, "text": "lama"}], {})
        wrapper._forward_transcript(wrapper._generation, "mic", [{"start": 0.0, "end": 1.0, "text": "baru"}], {})
        assert len(forwarded) == 1
        assert forwarded[0][1][0]["text"] == "baru"
    finally:
        wrapper.close(send_end_of_audio=False)
        reconnect_module.WhisperLiveStreamClient = original_client


def test_buffered_chunks_during_connect_flush_fifo_with_correct_timeline() -> None:
    # Simulasi capture-and-buffer selama connect (startup latency): chunk terkumpul
    # saat belum connected, lalu dikirim FIFO begitu koneksi siap — tanpa duplikasi/
    # kehilangan, dan SentTimeline tetap benar (tanpa drift karena tak ada drop).
    socket = StableFakeWebSocket()

    def factory(url, timeout, header):
        return socket

    connection = WhisperLiveConnectionConfig(profile=WhisperLiveProfile(), ready_timeout=1.0)
    wrapper = _BufferedReconnectClient(
        "mic",
        connection,
        policy=ReconnectPolicy(enabled=False, initial_backoff_seconds=0.1, max_backoff_seconds=0.1, buffer_seconds=10.0),
        on_transcript=lambda source, segments, message: None,
        on_status=lambda source, message: None,
    )

    import src.whisper.client_reconnect as reconnect_module

    original_client = reconnect_module.WhisperLiveStreamClient

    class FactoryClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["websocket_factory"] = factory
            super().__init__(*args, **kwargs)

    reconnect_module.WhisperLiveStreamClient = FactoryClient
    try:
        def _chunk(value: float, start: float) -> PreprocessedAudioChunk:
            return PreprocessedAudioChunk(
                source="mic",
                samples=np.array([value], dtype=np.float32),
                sample_rate=16_000,
                start_seconds=start,
                duration_seconds=0.5,
                rms_db=-20.0,
            )

        chunks = [_chunk(0.1, 0.0), _chunk(0.2, 0.5), _chunk(0.3, 1.0)]

        # Belum connect (policy disabled) -> chunk hanya di-buffer, tidak ada yang terkirim.
        for chunk in chunks:
            outcome = wrapper.send(chunk)
            assert outcome.buffered == 1
            assert outcome.sent == 0
        assert wrapper.buffered_count == 3

        # Koneksi siap -> flush buffer secara FIFO.
        wrapper.connect_initial()
        flushed = wrapper.flush_buffer()
        assert flushed.sent == 3
        assert wrapper.buffered_count == 0

        # FIFO: urutan biner terkirim = urutan input (tanpa duplikasi/kehilangan).
        assert len(socket.sent_binary) == 3
        decoded = [int(np.frombuffer(b, dtype=np.int16)[0]) for b in socket.sent_binary]
        assert decoded == [round(0.1 * 32767), round(0.2 * 32767), round(0.3 * 32767)]

        # SentTimeline benar tanpa drift (tidak ada drop): pemetaan identitas.
        assert wrapper.sent_timeline.map_to_real(0.0) == 0.0
        assert wrapper.sent_timeline.map_to_real(0.5) == 0.5
        assert wrapper.sent_timeline.map_to_real(1.0) == 1.0
    finally:
        wrapper.close(send_end_of_audio=False)
        reconnect_module.WhisperLiveStreamClient = original_client


def test_buffered_reconnect_replays_local_audio_after_send_failure() -> None:
    sockets = [
        StableFakeWebSocket(fail_binary_once=True),
        StableFakeWebSocket(),
    ]
    created: list[StableFakeWebSocket] = []

    def factory(url, timeout, header):
        socket = sockets.pop(0)
        created.append(socket)
        return socket

    connection = WhisperLiveConnectionConfig(
        profile=WhisperLiveProfile(),
        ready_timeout=1.0,
    )
    wrapper = _BufferedReconnectClient(
        "mic",
        connection,
        policy=ReconnectPolicy(enabled=True, initial_backoff_seconds=0.1, max_backoff_seconds=0.1, buffer_seconds=5.0),
        on_transcript=lambda source, segments, message: None,
        on_status=lambda source, message: None,
    )
    # Inject factory through the connection client constructor by patching module default in this narrow test.
    import src.whisper.client_reconnect as reconnect_module

    original_client = reconnect_module.WhisperLiveStreamClient

    class FactoryClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["websocket_factory"] = factory
            super().__init__(*args, **kwargs)

    reconnect_module.WhisperLiveStreamClient = FactoryClient
    try:
        wrapper.connect_initial()
        chunk1 = PreprocessedAudioChunk(
            source="mic",
            samples=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=16_000,
            start_seconds=0.0,
            duration_seconds=0.5,
            rms_db=-20.0,
        )
        chunk2 = PreprocessedAudioChunk(
            source="mic",
            samples=np.array([0.3, 0.4], dtype=np.float32),
            sample_rate=16_000,
            start_seconds=0.5,
            duration_seconds=0.5,
            rms_db=-20.0,
        )

        first = wrapper.send(chunk1)
        assert first.buffered == 1
        assert wrapper.buffered_count == 1

        time.sleep(0.12)
        second = wrapper.send(chunk2)

        assert second.sent == 2
        assert second.reconnect_successes == 1
        assert wrapper.buffered_count == 0
        assert len(created) == 2
        assert len(created[1].sent_binary) == 2
    finally:
        wrapper.close(send_end_of_audio=False)
        reconnect_module.WhisperLiveStreamClient = original_client
