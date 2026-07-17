"""Replay a WAV file to WhisperLive for production smoke tests."""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from src.capture.models import AudioFrame
from src.preprocessing.core import AudioPreprocessor
from src.preprocessing.models import PreprocessConfig
from src.whisper.capture import _build_mic_preprocess_config, _build_speaker_preprocess_config
from src.whisper.client_base import (
    WhisperLiveStreamClient,
)
from src.whisper.session_utils import _events_from_segments
from src.whisper.merger import TranscriptMerger


from src.whisper.models import (
    WhisperLiveConnectionConfig,
    WhisperLiveReplayConfig,
    WhisperLiveReplayResult,
    WhisperLiveSessionConfig,
)


def _preprocess_config_for_replay(config: WhisperLiveReplayConfig) -> PreprocessConfig:
    """Bangun konfigurasi DSP replay yang IDENTIK dengan mode live per-source.

    Memakai ulang builder live (`_build_mic/speaker_preprocess_config`) sehingga
    setiap perubahan profil DSP live otomatis mengalir ke replay — tanpa drift.
    Ini prasyarat agar baseline & A/B comparison merepresentasikan kondisi live
    (lihat 07-audit-findings: preprocess/replay lama memakai -20/24, live -23/18).
    """
    session_cfg = WhisperLiveSessionConfig(chunk_seconds=config.chunk_seconds)
    if config.source == "mic":
        return _build_mic_preprocess_config(session_cfg)
    return _build_speaker_preprocess_config(session_cfg)


def replay_wav_to_whisperlive(config: WhisperLiveReplayConfig) -> WhisperLiveReplayResult:
    """Preprocess a WAV file and stream it as one source to WhisperLive."""

    frame = _read_wav_as_frame(config.wav_path, config.source)
    preprocessor = AudioPreprocessor(_preprocess_config_for_replay(config))
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
        audio_format=config.audio_format,
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
