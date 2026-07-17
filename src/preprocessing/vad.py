from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.preprocessing.core import PreprocessedAudioChunk

if TYPE_CHECKING:
    import onnxruntime as ort

_log = logging.getLogger(__name__)

# Constants for Silero VAD v5 (16kHz)
_VAD_SAMPLE_RATE = 16000
_VAD_WINDOW_SAMPLES = 512
_VAD_CONTEXT_SAMPLES = 64


def default_model_path() -> Path:
    """Path default model Silero VAD, di-resolve relatif terhadap file ini.

    Memakai `__file__` (bukan cwd) agar tetap benar saat dijalankan dari folder
    lain maupun setelah dibundel Nuitka (asset diekstrak ke dir sementara di
    samping modul). Bila hasilnya tidak ada, SileroVADFilter degradasi ke
    passthrough — bukan error.
    """
    return Path(__file__).resolve().parents[1] / "assets" / "silero_vad.onnx"


class SileroVADFilter:
    """
    Filter pre-processing lokal menggunakan model ONNX Silero VAD.
    Digunakan untuk menentukan apakah sebuah chunk audio mengandung ucapan
    (berdasarkan threshold dan hangover), sehingga klien dapat men-drop chunk
    hening dan mengurangi beban decoding server (admission control, ADR-001).
    """

    def __init__(self, model_path: Path, threshold: float = 0.5, hangover_chunks: int = 2) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.hangover_chunks = hangover_chunks
        self._disabled = False
        self._session: ort.InferenceSession | None = None
        self._state: np.ndarray | None = None
        self._context: np.ndarray | None = None
        
        self._hangover_counter = 0

    def _init_session(self) -> None:
        if self._disabled:
            return
            
        if self._session is not None:
            return

        if not self.model_path.exists():
            _log.warning(
                "Silero VAD model not found at %s. VAD pre-filter will be disabled.",
                self.model_path
            )
            self._disabled = True
            return

        # Import onnxruntime secara lazy: dependensi native yang absen harus
        # menyebabkan degradasi ke passthrough (kontrak graceful degradation),
        # bukan ModuleNotFoundError saat modul di-import.
        try:
            import onnxruntime as ort
        except ImportError as exc:
            _log.warning(
                "onnxruntime tidak tersedia (%s). VAD pre-filter dinonaktifkan (passthrough).",
                exc,
            )
            self._disabled = True
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            # Prevent excessive warning logs
            opts.log_severity_level = 3
            
            self._session = ort.InferenceSession(
                str(self.model_path), 
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
            self.reset_state()
            _log.info("Loaded Silero VAD from %s", self.model_path)
        except Exception as exc:
            _log.warning("Failed to load Silero VAD model (%s). VAD will be disabled.", exc)
            self._disabled = True

    def reset_state(self) -> None:
        """Reset internal RNN state and context buffer."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _VAD_CONTEXT_SAMPLES), dtype=np.float32)
        self._hangover_counter = 0

    def _infer_512(self, x_512: np.ndarray) -> float:
        """Menjalankan inferensi untuk 1 frame (512 sample)."""
        if self._session is None or self._state is None or self._context is None:
            return 1.0  # Fallback to speech if uninitialized

        # Gabungkan dengan context sebelumnya (64 samples)
        # Bentuk akhir: (1, 576)
        x_in = np.concatenate([self._context, x_512], axis=1)
        
        ort_inputs = {
            "input": x_in,
            "state": self._state,
            "sr": np.array(_VAD_SAMPLE_RATE, dtype=np.int64)
        }
        
        try:
            out, next_state = self._session.run(None, ort_inputs)
            self._state = next_state
            # Simpan 64 sample terakhir sebagai context untuk frame berikutnya
            self._context = x_in[:, -_VAD_CONTEXT_SAMPLES:]
            return float(out[0, 0])
        except Exception as exc:
            _log.warning("Silero VAD inference failed: %s. Disabling VAD.", exc)
            self._disabled = True
            return 1.0

    def is_speech(self, chunk: PreprocessedAudioChunk) -> bool:
        """
        Menentukan apakah `chunk` memiliki suara ucapan (prob > threshold).
        Juga mengaplikasikan logika hangover.
        Jika VAD disabled atau error, selalu return True (graceful degradation).
        """
        if self._disabled:
            return True
            
        self._init_session()
        if self._disabled:
            return True

        if chunk.sample_rate != _VAD_SAMPLE_RATE:
            # Silero VAD v5 strictly requires 16kHz for 512-window logic
            # For simplicity, if not 16kHz, fallback to True
            _log.warning("VAD requires 16kHz audio, got %s. Disabling VAD.", chunk.sample_rate)
            self._disabled = True
            return True

        # Data sudah float32 [-1, 1] dari preprocessing
        audio_float = chunk.samples
        
        # Tambahkan batch dimension (1, N)
        audio_float = np.expand_dims(audio_float, 0)
        
        n_samples = audio_float.shape[1]
        
        # Zero-pad jika frame tidak merupakan kelipatan persis 512
        if n_samples % _VAD_WINDOW_SAMPLES != 0:
            pad_len = _VAD_WINDOW_SAMPLES - (n_samples % _VAD_WINDOW_SAMPLES)
            audio_float = np.pad(audio_float, ((0, 0), (0, pad_len)), mode='constant')
            
        max_prob = 0.0
        
        for i in range(0, audio_float.shape[1], _VAD_WINDOW_SAMPLES):
            x_512 = audio_float[:, i:i+_VAD_WINDOW_SAMPLES]
            prob = self._infer_512(x_512)
            max_prob = max(max_prob, prob)
            
        is_active = max_prob >= self.threshold
        
        if is_active:
            self._hangover_counter = self.hangover_chunks
            return True
        else:
            if self._hangover_counter > 0:
                self._hangover_counter -= 1
                return True
            return False
