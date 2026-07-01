"""Replay a WAV file to WhisperLive for production smoke tests."""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from src.capture.audio_frame import AudioFrame
from src.engine.preprocessing import AudioPreprocessor, PreprocessConfig
from src.engine.whisperlive_client import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
    WhisperLiveStreamClient,
)
from src.engine.whisperlive_session import _events_from_segments
from src.engine.transcript_merger import TranscriptMerger


@dataclass(frozen=True, slots=True)
class WhisperLiveReplayConfig:
    wav_path: Path
    source: Literal["mic", "speaker"] = "mic"
    server_host: str = "localhost"
    server_port: int = 9090
    use_wss: bool = False
    api_key: str | None = None
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    chunk_seconds: float = 2.5
    realtime: bool = False
    profile: WhisperLiveProfile = field(default_factory=WhisperLiveProfile)


@dataclass(frozen=True, slots=True)
class WhisperLiveReplayResult:
    chunks_sent: int
    results_received: int


def replay_wav_to_whisperlive(config: WhisperLiveReplayConfig) -> WhisperLiveReplayResult:
    """Preprocess a WAV file and stream it as one source to WhisperLive."""

    frame = _read_wav_as_frame(config.wav_path, config.source)
    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=config.chunk_seconds))
    chunks = preprocessor.preprocess_frames([frame])
    if not chunks:
        raise ValueError(f"no speech chunks produced from {config.wav_path}")

    merger = TranscriptMerger(reorder_delay_seconds=0.0)
    results_received = 0

    def on_transcript(source: str, segments: list[dict], message: dict) -> None:
        nonlocal results_received
        for event in _events_from_segments(source, config.profile.model, config.profile.language, segments):
            results_received += 1
            for entry in merger.add_result(event.result, completed=event.completed):
                print(entry.display, flush=True)

    connection = WhisperLiveConnectionConfig(
        host=config.server_host,
        port=config.server_port,
        use_wss=config.use_wss,
        api_key=config.api_key,
        ready_timeout=config.ready_timeout,
        profile=config.profile,
    )
    client = WhisperLiveStreamClient(config.source, connection, on_transcript=on_transcript)
    client.connect()
    try:
        for chunk in chunks:
            client.send_chunk(chunk)
            if config.realtime:
                time.sleep(chunk.duration_seconds)

        if config.realtime:
            time.sleep(2.0)
        else:
            time.sleep(1.0)
    finally:
        client.close()

    for entry in merger.flush():
        print(entry.display, flush=True)

    return WhisperLiveReplayResult(chunks_sent=len(chunks), results_received=results_received)


def _read_wav_as_frame(path: Path, source: str) -> AudioFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels)
    else:
        samples = samples.reshape(-1, 1)

    return AudioFrame(
        source=source,  # type: ignore[arg-type]
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        timestamp_seconds=0.0,
    )
