from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from src.capture.models import AudioFrame
from src.capture.wav_sink import write_frames_to_wav
from src.preprocessing.core import AudioPreprocessor, PreprocessConfig


def _frame(samples: np.ndarray, sample_rate: int = 48_000, source: str = "speaker") -> AudioFrame:
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return AudioFrame(
        source=source,  # type: ignore[arg-type]
        samples=samples.astype(np.float32),
        sample_rate=sample_rate,
        channels=samples.shape[1],
        timestamp_seconds=10.0,
    )


def test_preprocessor_outputs_mono_16khz_normalized_chunks() -> None:
    sample_rate = 48_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    speech = 0.08 * np.sin(2 * np.pi * 440 * t)
    stereo = np.column_stack([speech, speech * 0.8]).astype(np.float32)
    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=1.0))

    chunks = preprocessor.preprocess_frames([_frame(stereo, sample_rate=sample_rate)])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source == "speaker"
    assert chunk.sample_rate == 16_000
    assert chunk.samples.ndim == 1
    assert 15_900 <= chunk.frame_count <= 16_100
    assert -21.0 <= chunk.rms_db <= -19.0
    assert np.max(np.abs(chunk.samples)) <= 0.95


def test_preprocessor_drops_silent_audio() -> None:
    silence = np.zeros((48_000, 2), dtype=np.float32)
    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=1.0))

    assert preprocessor.preprocess_frames([_frame(silence)]) == []


def test_highpass_filter_reduces_low_frequency_energy() -> None:
    sample_rate = 48_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    low = 0.2 * np.sin(2 * np.pi * 30 * t)
    speech = 0.05 * np.sin(2 * np.pi * 440 * t)
    mixed = (low + speech).astype(np.float32)
    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=1.0))

    chunk = preprocessor.preprocess_frames([_frame(mixed, sample_rate=sample_rate)])[0]

    fft = np.fft.rfft(chunk.samples)
    freqs = np.fft.rfftfreq(chunk.samples.size, d=1 / chunk.sample_rate)
    low_bin = int(np.argmin(np.abs(freqs - 30)))
    speech_bin = int(np.argmin(np.abs(freqs - 440)))
    assert np.abs(fft[speech_bin]) > np.abs(fft[low_bin]) * 5


def test_preprocessor_can_process_mic_source() -> None:
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    speech = 0.1 * np.sin(2 * np.pi * 440 * t)
    preprocessor = AudioPreprocessor(PreprocessConfig(chunk_seconds=1.0))

    chunks = preprocessor.preprocess_frames([_frame(speech, sample_rate=sample_rate, source="mic")])

    assert len(chunks) == 1
    assert chunks[0].source == "mic"
    assert chunks[0].sample_rate == 16_000


def test_preprocessor_marks_vad_disabled_acceptance() -> None:
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    low_speech = 0.01 * np.sin(2 * np.pi * 440 * t)
    preprocessor = AudioPreprocessor(
        PreprocessConfig(
            chunk_seconds=1.0,
            client_vad_enabled=False,
            min_input_rms_db=-60.0,
        )
    )

    chunks = preprocessor.preprocess_frames([_frame(low_speech, sample_rate=sample_rate, source="speaker")])

    assert len(chunks) == 1
    assert preprocessor.last_decisions[0]["reason"] == "vad_disabled_accepted"
    assert preprocessor.last_decisions[0]["client_vad_enabled"] is False


def test_preprocessor_with_vad_disabled_still_drops_silence_by_noise_floor() -> None:
    silence = np.zeros((16_000, 1), dtype=np.float32)
    preprocessor = AudioPreprocessor(
        PreprocessConfig(
            chunk_seconds=1.0,
            client_vad_enabled=False,
            min_input_rms_db=-48.0,
        )
    )

    assert preprocessor.preprocess_frames([_frame(silence, sample_rate=16_000, source="speaker")]) == []
    assert preprocessor.last_decisions[0]["reason"] == "below_min_rms"
    assert preprocessor.last_decisions[0]["client_vad_enabled"] is False
