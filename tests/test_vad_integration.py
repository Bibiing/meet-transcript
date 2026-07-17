"""Integrasi VAD dengan model Silero NYATA.

Berbeda dari tests/test_vad.py (murni, memock inferensi), berkas ini memuat model
ONNX asli dan menjalankan inferensi sungguhan. Di-skip otomatis bila onnxruntime
atau asset model tidak tersedia, sehingga tetap deterministik di CI tanpa dependensi
berat namun mengunci kebenaran signature IO (_infer_512) di lingkungan terverifikasi.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from src.preprocessing.models import PreprocessedAudioChunk
from src.preprocessing.vad import _VAD_SAMPLE_RATE, SileroVADFilter, default_model_path

SR = _VAD_SAMPLE_RATE

pytestmark = pytest.mark.skipif(
    not default_model_path().exists(), reason="Silero VAD model asset tidak tersedia"
)


def _chunk(samples: np.ndarray) -> PreprocessedAudioChunk:
    return PreprocessedAudioChunk(
        source="mic",
        samples=samples.astype(np.float32),
        sample_rate=SR,
        start_seconds=0.0,
        duration_seconds=len(samples) / SR,
        rms_db=-20.0,
    )


def _voiced(dur: float = 0.5, f0: float = 130.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 8))
    sig = sig * 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    return (0.3 * sig / np.max(np.abs(sig))).astype(np.float32)


def test_real_model_loads_and_runs_without_disabling() -> None:
    # Risiko utama M4: signature IO salah -> inferensi throw -> filter disable senyap.
    vad = SileroVADFilter(model_path=default_model_path())
    vad._init_session()
    assert vad._disabled is False
    vad.is_speech(_chunk(np.zeros(SR // 2, dtype=np.float32)))
    assert vad._disabled is False  # inferensi nyata berhasil, tidak jatuh ke passthrough


def test_real_model_drops_silence() -> None:
    vad = SileroVADFilter(model_path=default_model_path(), threshold=0.5, hangover_chunks=0)
    assert vad.is_speech(_chunk(np.zeros(SR // 2, dtype=np.float32))) is False


def test_real_model_keeps_voiced_onset() -> None:
    vad = SileroVADFilter(model_path=default_model_path(), threshold=0.5, hangover_chunks=0)
    assert vad.is_speech(_chunk(_voiced())) is True
