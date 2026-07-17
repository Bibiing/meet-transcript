from __future__ import annotations

from pathlib import Path

import numpy as np

from src.preprocessing.models import PreprocessedAudioChunk
from src.preprocessing.vad import (
    _VAD_WINDOW_SAMPLES,
    _VAD_SAMPLE_RATE,
    SileroVADFilter,
    default_model_path,
)


def test_default_model_path_points_to_bundled_asset() -> None:
    path = default_model_path()
    assert path.is_absolute()
    assert path.parts[-2:] == ("assets", "silero_vad.onnx")


def _chunk(samples: np.ndarray, *, sample_rate: int = _VAD_SAMPLE_RATE) -> PreprocessedAudioChunk:
    return PreprocessedAudioChunk(
        source="mic",
        samples=samples.astype(np.float32),
        sample_rate=sample_rate,
        start_seconds=0.0,
        duration_seconds=len(samples) / sample_rate,
        rms_db=-20.0,
    )


def test_passthrough_when_model_missing(tmp_path: Path) -> None:
    # Model tidak ada -> filter nonaktif, semua audio diloloskan (graceful degradation).
    vad = SileroVADFilter(model_path=tmp_path / "missing.onnx")
    assert vad.is_speech(_chunk(np.zeros(_VAD_WINDOW_SAMPLES))) is True
    assert vad._disabled is True


def test_passthrough_when_model_cannot_be_loaded(tmp_path: Path) -> None:
    # File ada tetapi bukan model ONNX valid ATAU onnxruntime tak terpasang.
    # Kedua kondisi harus berujung pada passthrough, bukan exception yang naik.
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"not-a-real-onnx-model")
    vad = SileroVADFilter(model_path=model)
    assert vad.is_speech(_chunk(np.zeros(_VAD_WINDOW_SAMPLES))) is True
    assert vad._disabled is True


def test_passthrough_and_disable_on_non_16khz(tmp_path: Path) -> None:
    vad = SileroVADFilter(model_path=tmp_path / "m.onnx")
    # Paksa state "loaded" agar is_speech mencapai pengecekan sample_rate.
    vad._session = object()  # type: ignore[assignment]
    vad.reset_state()

    assert vad.is_speech(_chunk(np.zeros(_VAD_WINDOW_SAMPLES), sample_rate=8_000)) is True
    assert vad._disabled is True


def test_threshold_and_hangover_logic(tmp_path: Path) -> None:
    vad = SileroVADFilter(model_path=tmp_path / "m.onnx", threshold=0.5, hangover_chunks=2)
    vad._session = object()  # type: ignore[assignment]
    vad.reset_state()

    probs = iter([0.9, 0.1, 0.1, 0.1])
    vad._infer_512 = lambda x_512: next(probs)  # type: ignore[assignment]

    window = np.zeros(_VAD_WINDOW_SAMPLES, dtype=np.float32)
    assert vad.is_speech(_chunk(window)) is True   # 0.9 >= 0.5 -> aktif, hangover=2
    assert vad.is_speech(_chunk(window)) is True   # hening, hangover 2->1 -> True
    assert vad.is_speech(_chunk(window)) is True   # hening, hangover 1->0 -> True
    assert vad.is_speech(_chunk(window)) is False  # hening, hangover habis -> False


def test_speech_active_when_any_window_exceeds_threshold(tmp_path: Path) -> None:
    vad = SileroVADFilter(model_path=tmp_path / "m.onnx", threshold=0.5, hangover_chunks=0)
    vad._session = object()  # type: ignore[assignment]
    vad.reset_state()

    probs = iter([0.1, 0.8])  # window kedua aktif
    vad._infer_512 = lambda x_512: next(probs)  # type: ignore[assignment]

    two_windows = np.zeros(_VAD_WINDOW_SAMPLES * 2, dtype=np.float32)
    # max_prob antar-window dipakai: satu window aktif -> chunk dianggap speech.
    assert vad.is_speech(_chunk(two_windows)) is True


def test_disabled_filter_is_passthrough_without_touching_model(tmp_path: Path) -> None:
    vad = SileroVADFilter(model_path=tmp_path / "does-not-matter.onnx")
    vad._disabled = True
    # Tidak boleh mencoba load / infer saat sudah disabled.
    assert vad.is_speech(_chunk(np.zeros(_VAD_WINDOW_SAMPLES))) is True
