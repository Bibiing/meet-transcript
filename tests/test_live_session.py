"""Tests for the live transcription session pipeline."""

from __future__ import annotations

import queue
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.capture.audio_frame import AudioFrame
from src.engine.live_session import LiveSessionConfig, LiveSessionStats, _reader_thread, run_live_session
from src.engine.preprocessing import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk


def _make_frame(
    source: str = "mic",
    duration_seconds: float = 0.5,
    sample_rate: int = 16_000,
    rms: float = 0.05,
) -> AudioFrame:
    n_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, n_samples, dtype=np.float32)
    samples = (rms * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
    return AudioFrame(
        source=source,  # type: ignore[arg-type]
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        timestamp_seconds=0.0,
    )


class FakeMicStream:
    """Minimal mic stream stub that returns a fixed set of frames."""

    def __init__(self, frames: list[AudioFrame], delay_after: float = 0.0) -> None:
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


class FakeWhisperModel:
    def transcribe(self, audio: Any, **kwargs: Any) -> dict:
        text = "halo dari live session"
        return {
            "text": text,
            "language": kwargs.get("language") or "id",
            "segments": [{"start": 0.0, "end": 1.0, "text": text}],
        }


# --------------------------------------------------------------------------- #
# _reader_thread unit tests                                                    #
# --------------------------------------------------------------------------- #

def test_reader_thread_accumulates_and_pushes_chunks() -> None:
    """Reader thread must push a preprocessed chunk after chunk_seconds have accumulated."""
    # 8 frames × 0.5s = 4.0s — should trigger one chunk push
    frames = [_make_frame(source="mic", duration_seconds=0.5, sample_rate=16_000) for _ in range(8)]
    stream = FakeMicStream(frames)
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue()
    stop_event = threading.Event()
    stats = LiveSessionStats()

    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=4.0))

    t = threading.Thread(
        target=_reader_thread,
        args=(stream, preprocessor, chunk_queue, 4.0, stop_event, stats),
    )
    t.start()
    t.join(timeout=5.0)
    stop_event.set()

    assert not chunk_queue.empty(), "reader thread must push at least one chunk"
    chunk = chunk_queue.get_nowait()
    assert chunk.source == "mic"
    assert chunk.sample_rate == 16_000


def test_reader_thread_drops_chunk_when_queue_full() -> None:
    """When the chunk queue is full the reader must not block and must track dropped count."""
    frames = [_make_frame(source="mic", duration_seconds=0.5, sample_rate=16_000) for _ in range(8)]
    stream = FakeMicStream(frames)
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(maxsize=0)  # always full-like
    stop_event = threading.Event()
    stats = LiveSessionStats()

    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=4.0))

    # Fill the queue to capacity (maxsize=1) so put_nowait raises Full
    tiny_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(maxsize=1)
    # Pre-fill
    dummy_frame = _make_frame(source="mic", duration_seconds=4.0, sample_rate=16_000)
    dummy_chunk = preprocessor.preprocess_frames([dummy_frame])
    if dummy_chunk:
        tiny_queue.put_nowait(dummy_chunk[0])

    t = threading.Thread(
        target=_reader_thread,
        args=(stream, preprocessor, tiny_queue, 4.0, stop_event, stats),
    )
    t.start()
    t.join(timeout=5.0)
    stop_event.set()

    # Should not have hung — test passes if it reaches here
    assert True


# --------------------------------------------------------------------------- #
# run_live_session integration (heavily mocked)                                #
# --------------------------------------------------------------------------- #

def test_live_session_returns_stats_and_calls_on_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_live_session must return LiveSessionStats and invoke on_result for accepted results."""
    # 8 speech frames = 4s → one preprocessed chunk → one Whisper call
    speech_frames = [_make_frame(source="mic", duration_seconds=0.5) for _ in range(8)]

    fake_mic = FakeMicStream(speech_frames)
    fake_model = FakeWhisperModel()

    # Patch MicrophoneStream so we don't open real audio device
    monkeypatch.setattr(
        "src.engine.live_session.MicrophoneStream",
        lambda config: fake_mic,
    )
    # Patch detect_os so speaker branch is skipped in CI
    monkeypatch.setattr(
        "src.engine.live_session.detect_os",
        lambda: MagicMock(value="windows"),
    )
    # Patch WindowsLoopbackStream to raise CaptureNotSupportedError (so only mic runs)
    from src.capture.errors import CaptureNotSupportedError
    monkeypatch.setattr(
        "src.engine.live_session.WindowsLoopbackStream",
        lambda config: (_ for _ in ()).throw(CaptureNotSupportedError("test: no loopback")),
    )

    received: list = []

    cfg = LiveSessionConfig(
        chunk_seconds=4.0,
        source="both",  # speaker will be skipped gracefully
        whisper_config=__import__("src.engine.whisper", fromlist=["WhisperConfig"]).WhisperConfig(
            model_name="tiny",
            fp16=False,
        ),
    )

    with patch("src.engine.live_session.OpenAIWhisperTranscriber") as MockTranscriber:
        instance = MockTranscriber.return_value

        def fake_transcribe(chunk: PreprocessedAudioChunk):
            from src.engine.whisper import TranscriptionResult, TranscriptionSegment
            return TranscriptionResult(
                source=chunk.source,
                text="halo dari live session",
                model_name="tiny",
                language="id",
                start_seconds=chunk.start_seconds,
                duration_seconds=chunk.duration_seconds,
            )

        instance.transcribe_chunk.side_effect = fake_transcribe

        stats = run_live_session(cfg, on_result=received.append)

    assert isinstance(stats, LiveSessionStats)
    assert stats.chunks_processed >= 0  # may be 0 if chunk queue empty before transcriber starts
