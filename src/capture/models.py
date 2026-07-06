from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path

import numpy as np

AudioSource = Literal["mic", "speaker"]
CaptureSource = Literal["mic", "speaker", "both"]

# frame audio yang ditangkap dari sumber audio
@dataclass(frozen=True, slots=True)
class AudioFrame:
    source: AudioSource
    samples: np.ndarray         # data audio dalam bentuk np
    sample_rate: int            # kecepatan hz per detik
    channels: int               # jumlah saluran audio (1 mono, 2 stereo)
    timestamp_seconds: float    # waktu dalam detik saat frame audio
    status: str = ""            # status tambahan untuk frame audio 

    # jumlah frame audio dalam frame
    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    # durasi frame audio dalam detik
    # durasi = jumlah frame / kecepatan titik data audio per detik 
    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate 

# normalisasi sampel audio ke float32 dan bentuk (frame, saluran). saluran: (1 untuk mono, 2 untuk stereo)
def normalize_samples(samples: np.ndarray, channels: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32) # convert ke float32
    if audio.ndim == 1: 
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError("audio samples must be a 1D or 2D array")
    if audio.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, got {audio.shape[1]}")
    return audio.copy()

@dataclass(frozen=True, slots=True)
class CaptureOptions:
    seconds: float = 3.0                            # durasi perekaman dalam detik
    output_dir: Path = Path("audio")                # direktori output untuk menyimpan file WAV
    source: CaptureSource = "both"                  # sumber audio yang akan direkam: "mic", "speaker", atau "both"
    sample_rate: int | None = None                  # sample rate dalam Hz
    block_size: int = 1_024                         # ukuran blok audio dalam sampel
    queue_size: int = 64                            # ukuran antrian untuk buffer audio
    mic_channels: int | None = None                 # jumlah saluran untuk mikrofon
    speaker_channels: int | None = None             # jumlah saluran untuk speaker
    mic_device: int | str | None = None             # perangkat mikrofon
    speaker_device: int | str | None = None         # perangkat speaker

@dataclass(frozen=True, slots=True)
class CaptureResult:
    source: str                                     # sumber audio yang direkam: "mic" atau "speaker"
    path: Path | None = None                        # path ke file WAV yang dihasilkan
    frame_count: int = 0                            # jumlah frame yang direkam
    duration_seconds: float = 0.0                   # durasi perekaman dalam detik
    sample_rate: int = 0                            # sample rate dalam Hz
    channels: int = 0                               # jumlah saluran audio
    warning: str = ""                               # peringatan jika ada masalah selama perekaman

    @property
    def ok(self) -> bool: return self.path is not None and not self.warning # mengembalikan True jika perekaman berhasil dan tidak ada peringatan

@dataclass(frozen=True, slots=True)
class MicrophoneConfig:
    """Konfigurasi capture mikrofon yang dipakai oleh MicrophoneStream."""
    sample_rate: int | None = None  
    channels: int | None = None     
    block_size: int = 1_024
    dtype: str = "float32"
    latency: str | float | None = "low"
    device: int | str | None = None
    queue_size: int = 64

@dataclass(frozen=True, slots=True)
class MacSystemAudioConfig:
    sample_rate: int = 48_000
    channels: int = 2
    block_size: int = 1_024
    dtype: str = "float32"
    queue_size: int = 64
    excludes_current_process_audio: bool = True

@dataclass(frozen=True, slots=True)
class WasapiDevice:
    index: int
    name: str
    sample_rate: int
    channels: int

# Konfigurasi capture loopback speaker Windows
@dataclass(frozen=True, slots=True)
class WindowsLoopbackConfig:
    sample_rate: int | None = None
    channels: int | None = None
    block_size: int = 1_024
    device: int | str | None = None
    queue_size: int = 64

# Konfigurasi capture speaker macOS
@dataclass(frozen=True, slots=True)
class SoundcardLoopbackDevice:
    index: int
    name: str
    channels: int
    raw_device: Any
