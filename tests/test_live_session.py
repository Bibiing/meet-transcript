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
