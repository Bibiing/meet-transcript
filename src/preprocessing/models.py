from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

# konfigurasi default untuk preprocessing audio.
@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    target_sample_rate: int = 16_000            # 16000 Hz untuk Whisper
    chunk_seconds: float = 2.5                  # durasi chunk audio untuk dikirim ke WhisperLive
    highpass_cutoff_hz: float = 80.0            # frekuensi potong high-pass filter
    highpass_order: int = 2                     # order filter untuk high-pass filter
    target_rms_db: float = -20.0                # target RMS level untuk normalisasi audio
    clip_limit: float = 0.95                    # batas clipping untuk normalisasi audio
    vad_rms_threshold: float = 0.015            # threshold RMS untuk VAD
    vad_peak_threshold: float = 0.05            # threshold puncak untuk VAD
    vad_speech_fraction: float = 0.30           # fraksi minimum frame yang harus terdeteksi
    client_vad_enabled: bool = True             # aktifkan energy VAD lokal
    min_input_rms_db: float = -42.0             # minimum RMS level untuk input audio agar diterima
    min_chunk_seconds: float = 1.0              # durasi minimum chunk audio agar diterima
    max_normalization_gain_db: float = 24.0     # gain maksimum untuk normalisasi audio


# menyimpan chunk audio yang telah diproses beserta metadata diagnostik.
@dataclass(frozen=True, slots=True)
class PreprocessedAudioChunk:
    source: str                 # sumber audio
    samples: np.ndarray         # array 1D float32 audio mono
    sample_rate: int            # sample rate audio
    start_seconds: float        # waktu mulai chunk audio relatif terhadap awal sumber
    duration_seconds: float     # durasi chunk audio 
    rms_db: float               # RMS level dari chunk audio setelah normalisasi
    input_rms_db: float = 0.0   # RMS level dari chunk audio sebelum normalisasi

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    source: str
    input_path: Path
    output_path: Path | None
    chunk_count: int
    duration_seconds: float
    warning: str = ""
