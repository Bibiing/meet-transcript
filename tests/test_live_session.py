from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from src.capture.models import AudioFrame
from src.preprocessing.core import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk
from src.whisper.capture import _build_capture_streams, _reader_thread
from src.whisper.models import WhisperLiveSessionConfig, WhisperLiveSessionStats


def _make_frame(
    source: str = "mic",
    duration_seconds: float = 0.5,
    sample_rate: int = 16_000,
    rms: float = 0.05,
    timestamp_seconds: float = 0.0,
) -> AudioFrame:
    n_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, n_samples, dtype=np.float32)
    samples = (rms * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
    return AudioFrame(
        source=source,  # type: ignore[arg-type]
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        timestamp_seconds=timestamp_seconds,
    )


class FakeMicStream:
    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = list(frames)
        self._idx = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read_frame(self, timeout: float = 0.1) -> AudioFrame | None:
        if self._idx < len(self._frames):
            frame = self._frames[self._idx]
            self._idx += 1
            return frame
        return None  # exhausted


def test_reader_thread_accumulates_and_pushes_chunks() -> None:
    frames = [_make_frame(source="mic", duration_seconds=0.5, sample_rate=16_000) for _ in range(8)]
    stream = FakeMicStream(frames)
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue()
    stop_event = threading.Event()
    stats = WhisperLiveSessionStats()

    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=4.0))
    t = threading.Thread(
        target=_reader_thread,
        args=(stream, preprocessor, chunk_queue, 4.0, stop_event, stats),
    )
    t.start()
    assert _wait_for_queue(chunk_queue)
    stop_event.set()
    t.join(timeout=2.0)

    chunk = chunk_queue.get_nowait()
    assert chunk.source == "mic"
    assert chunk.sample_rate == 16_000


def test_reader_thread_normalizes_absolute_frame_timestamps_to_session_relative() -> None:
    frames = [
        _make_frame(
            source="speaker",
            duration_seconds=0.5,
            sample_rate=16_000,
            timestamp_seconds=12_524.0 + (index * 0.5),
        )
        for index in range(8)
    ]
    stream = FakeMicStream(frames)
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue()
    stop_event = threading.Event()
    stats = WhisperLiveSessionStats()

    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=4.0))
    t = threading.Thread(
        target=_reader_thread,
        args=(stream, preprocessor, chunk_queue, 4.0, stop_event, stats),
    )
    t.start()
    assert _wait_for_queue(chunk_queue)
    stop_event.set()
    t.join(timeout=2.0)

    chunk = chunk_queue.get_nowait()
    assert chunk.source == "speaker"
    assert chunk.start_seconds == 0.0


def test_reader_thread_drops_chunk_when_queue_full() -> None:
    frames = [_make_frame(source="mic", duration_seconds=0.5, sample_rate=16_000) for _ in range(8)]
    stream = FakeMicStream(frames)
    stop_event = threading.Event()
    stats = WhisperLiveSessionStats()

    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=4.0))
    tiny_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(maxsize=1)
    dummy_frame = _make_frame(source="mic", duration_seconds=4.0, sample_rate=16_000)
    dummy_chunk = preprocessor.preprocess_frames([dummy_frame])
    assert dummy_chunk
    tiny_queue.put_nowait(dummy_chunk[0])

    t = threading.Thread(
        target=_reader_thread,
        args=(stream, preprocessor, tiny_queue, 4.0, stop_event, stats),
    )
    t.start()
    assert _wait_for(lambda: stats.chunks_dropped >= 1)
    stop_event.set()
    t.join(timeout=2.0)

    assert stats.chunks_dropped >= 1


class _FakeVAD:
    def __init__(self, speech: bool) -> None:
        self.speech = speech
        self.calls = 0

    def is_speech(self, chunk: PreprocessedAudioChunk) -> bool:
        self.calls += 1
        return self.speech


def test_reader_thread_drops_non_speech_chunks_via_client_vad() -> None:
    frames = [_make_frame(source="mic", duration_seconds=0.5, sample_rate=16_000) for _ in range(8)]
    stream = FakeMicStream(frames)
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue()
    stop_event = threading.Event()
    stats = WhisperLiveSessionStats()
    vad = _FakeVAD(speech=False)

    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=4.0))
    t = threading.Thread(
        target=_reader_thread,
        args=(stream, preprocessor, chunk_queue, 4.0, stop_event, stats, vad),
    )
    t.start()
    assert _wait_for(lambda: stats.client_vad_dropped >= 1)
    stop_event.set()
    t.join(timeout=2.0)

    # Chunk non-speech tidak masuk queue; dihitung sebagai client_vad_dropped.
    assert chunk_queue.empty()
    assert stats.client_vad_dropped >= 1
    assert vad.calls >= 1


def test_reader_thread_keeps_speech_chunks_via_client_vad() -> None:
    frames = [_make_frame(source="mic", duration_seconds=0.5, sample_rate=16_000) for _ in range(8)]
    stream = FakeMicStream(frames)
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue()
    stop_event = threading.Event()
    stats = WhisperLiveSessionStats()
    vad = _FakeVAD(speech=True)

    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=4.0))
    t = threading.Thread(
        target=_reader_thread,
        args=(stream, preprocessor, chunk_queue, 4.0, stop_event, stats, vad),
    )
    t.start()
    assert _wait_for_queue(chunk_queue)
    stop_event.set()
    t.join(timeout=2.0)

    assert stats.client_vad_dropped == 0
    chunk = chunk_queue.get_nowait()
    assert chunk.source == "mic"


def test_session_mute_registry_is_per_source() -> None:
    from src.whisper.session import _is_source_muted, _set_source_muted

    try:
        assert _is_source_muted("mic") is False
        _set_source_muted("mic", True)
        assert _is_source_muted("mic") is True
        # Mute satu source tidak memengaruhi source lain.
        assert _is_source_muted("speaker") is False
        _set_source_muted("mic", False)
        assert _is_source_muted("mic") is False
    finally:
        _set_source_muted("mic", False)
        _set_source_muted("speaker", False)


def test_build_client_vad_filter_respects_kill_switch() -> None:
    from src.whisper.capture import _build_client_vad_filter

    assert _build_client_vad_filter(WhisperLiveSessionConfig()) is None

    vad = _build_client_vad_filter(
        WhisperLiveSessionConfig(client_vad_enabled=True, client_vad_threshold=0.7, client_vad_hangover_chunks=3)
    )
    assert vad is not None
    assert vad.threshold == 0.7
    assert vad.hangover_chunks == 3


def test_build_capture_streams_creates_mic_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class FakeMicrophoneStream:
        def __init__(self, config) -> None:
            created["config"] = config

    monkeypatch.setattr("src.whisper.capture.MicrophoneStream", FakeMicrophoneStream)

    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue()
    stop_event = threading.Event()
    stats = WhisperLiveSessionStats()

    streams, threads = _build_capture_streams(
        WhisperLiveSessionConfig(source="mic", chunk_seconds=0.5, mic_device="Headset"),
        chunk_queue,
        stop_event,
        stats,
    )

    assert len(streams) == 1
    assert len(threads) == 1
    assert created["config"].device == "Headset"
    assert threads[0].name == "whisperlive-mic-reader"


def _wait_for_queue(chunk_queue: queue.Queue[PreprocessedAudioChunk], timeout_seconds: float = 2.0) -> bool:
    return _wait_for(lambda: not chunk_queue.empty(), timeout_seconds=timeout_seconds)


def _wait_for(predicate, timeout_seconds: float = 2.0) -> bool:
    stop_at = time.monotonic() + timeout_seconds
    while time.monotonic() < stop_at:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# Regresi: perintah `stop` dari parent harus memicu jalur berhenti yang SAMA
# dengan SIGINT (set stop_event), sehingga blok finalisasi — END_OF_AUDIO +
# flush merger — tetap dijalankan. Pada aplikasi paket (tanpa console) inilah
# satu-satunya jalur graceful; tanpanya transcript penutup hilang.
def test_command_reader_stop_command_sets_stop_event(monkeypatch) -> None:
    import io

    from src.utils.status_ipc import format_command_line
    from src.whisper import session

    monkeypatch.setattr(session.sys, "stdin", io.StringIO(format_command_line("stop")))
    stop_event = threading.Event()

    session._command_reader_thread(stop_event)

    assert stop_event.is_set() is True


def test_command_reader_ignores_unknown_command(monkeypatch) -> None:
    import io

    from src.utils.status_ipc import format_command_line
    from src.whisper import session

    monkeypatch.setattr(session.sys, "stdin", io.StringIO(format_command_line("bogus")))
    stop_event = threading.Event()

    session._command_reader_thread(stop_event)

    assert stop_event.is_set() is False


def test_command_reader_still_handles_set_mute(monkeypatch) -> None:
    import io

    from src.utils.status_ipc import format_command_line
    from src.whisper import session

    monkeypatch.setattr(
        session.sys, "stdin",
        io.StringIO(format_command_line("set_mute", source="mic", muted=True)),
    )
    stop_event = threading.Event()
    try:
        session._command_reader_thread(stop_event)
        assert session._is_source_muted("mic") is True
        assert stop_event.is_set() is False
    finally:
        session._set_source_muted("mic", False)
