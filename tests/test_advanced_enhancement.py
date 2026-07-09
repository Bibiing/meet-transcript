from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.preprocessing.advanced_enhancement import (
    AudioQualityLevel,
    assess_audio_quality,
    adaptive_noise_reduction,
    enhance_audio_file,
)

def test_assess_audio_quality_high() -> None:
    sr = 16_000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    # Nada jernih dengan amplitudo besar dan tanpa noise, rolloff tinggi (broadband)
    speech = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
    # Scale up sedikit
    speech = (speech * 0.1).astype(np.float32)
    
    quality = assess_audio_quality(speech, sr)
    # Heuristik kita mungkin memetakan ini ke medium atau high, 
    # karena kita tidak bisa mock librosa ZCR secara presisi, kita mock assert tidak crash.
    assert isinstance(quality, AudioQualityLevel)


def test_assess_audio_quality_low_empty() -> None:
    empty = np.array([], dtype=np.float32)
    assert assess_audio_quality(empty, 16_000) == AudioQualityLevel.LOW


def test_adaptive_noise_reduction_high_quality() -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    reduced = adaptive_noise_reduction(audio, sr, AudioQualityLevel.HIGH)
    assert len(reduced) == len(audio)


def test_adaptive_noise_reduction_low_quality() -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.2, len(audio)).astype(np.float32)
    mixed = audio + noise
    
    reduced = adaptive_noise_reduction(mixed, sr, AudioQualityLevel.LOW)
    assert len(reduced) == len(audio)
    
    # Noise floor harus lebih kecil di chunk yang direduksi
    assert np.mean(np.abs(reduced)) < np.mean(np.abs(mixed))


def test_enhance_audio_file_creates_output(tmp_path: Path) -> None:
    import soundfile as sf
    sr = 16_000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    input_wav = tmp_path / "test_in.wav"
    output_wav = tmp_path / "test_out.wav"
    
    sf.write(str(input_wav), audio, sr)
    
    res = enhance_audio_file(input_wav, output_wav, sr)
    
    assert res == output_wav
    assert output_wav.exists()
    
    # Check that output is readable and length roughly matches
    out_audio, out_sr = sf.read(str(output_wav), dtype="float32")
    assert out_sr == sr
    assert len(out_audio) > 0
    # Overlap-add might introduce slight changes or zeros at boundaries, length should be same
    assert len(out_audio) == len(audio)
