"""Phase 3 WAV preprocessing runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from src.capture.audio_frame import AudioFrame
from src.capture.wav_sink import write_frames_to_wav
from src.engine.preprocessing import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    source: str
    input_path: Path
    output_path: Path | None
    chunk_count: int
    duration_seconds: float
    warning: str = ""


# Alias kompatibilitas untuk test/dokumen lama yang masih memakai nama phase.
Phase3PreprocessResult = PreprocessResult


def preprocess_wav_file(
    input_path: Path,
    output_path: Path,
    *,
    source: str,
    preprocessor: AudioPreprocessor | None = None,
) -> PreprocessResult:
    if not input_path.exists():
        return PreprocessResult(
            source=source,
            input_path=input_path,
            output_path=None,
            chunk_count=0,
            duration_seconds=0.0,
            warning="input WAV does not exist",
        )

    frame = _read_wav_as_frame(input_path, source=source)
    processor = preprocessor or AudioPreprocessor(PreprocessConfig())
    chunks = processor.preprocess_frames([frame])
    if not chunks:
        return PreprocessResult(
            source=source,
            input_path=input_path,
            output_path=None,
            chunk_count=0,
            duration_seconds=0.0,
            warning="no speech chunk passed VAD",
        )

    output = _write_chunks(output_path, chunks)
    duration = sum(chunk.duration_seconds for chunk in chunks)
    return PreprocessResult(
        source=source,
        input_path=input_path,
        output_path=output,
        chunk_count=len(chunks),
        duration_seconds=duration,
    )


def preprocess_audio_dir(
    input_dir: Path = Path("audio"),
    output_dir: Path = Path("audio"),
    *,
    preprocessor: AudioPreprocessor | None = None,
) -> list[PreprocessResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = preprocessor or AudioPreprocessor(PreprocessConfig())
    return [
        preprocess_wav_file(
            input_dir / "mic.wav",
            output_dir / "mic.preprocessed.wav",
            source="mic",
            preprocessor=processor,
        ),
        preprocess_wav_file(
            input_dir / "speaker.wav",
            output_dir / "speaker.preprocessed.wav",
            source="speaker",
            preprocessor=processor,
        ),
    ]


def _read_wav_as_frame(path: Path, *, source: str) -> AudioFrame:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got sample width={sample_width}")

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    samples = pcm.reshape(-1, channels)
    return AudioFrame(
        source=source,  # type: ignore[arg-type]
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        timestamp_seconds=0.0,
    )


def _write_chunks(path: Path, chunks: list[PreprocessedAudioChunk]) -> Path:
    frames = [
        AudioFrame(
            source=chunk.source,  # type: ignore[arg-type]
            samples=chunk.samples.reshape(-1, 1),
            sample_rate=chunk.sample_rate,
            channels=1,
            timestamp_seconds=chunk.start_seconds,
        )
        for chunk in chunks
    ]
    return write_frames_to_wav(path, frames)
