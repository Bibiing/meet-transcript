from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np


AudioSource = Literal["mic", "speaker"]

@dataclass(frozen=True, slots=True)
class AudioFrame:
    """Block raw audio captured dari satu sumber."""
    source: AudioSource
    samples: np.ndarray         # data audio dalam bentuk np
    sample_rate: int            # kecepatan hz per detik
    channels: int               # jumlah saluran audio (1 mono, 2 stereo)
    timestamp_seconds: float    # waktu dalam detik saat frame audio
    status: str = ""            # status tambahan untuk frame audio 

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate


def normalize_samples(samples: np.ndarray, channels: int) -> np.ndarray:
    """Mengembalikan sampel audio sebagai array float32 yang disalin dengan bentuk (frame, saluran)."""

    audio = np.asarray(samples, dtype=np.float32) # convert ke float32
    if audio.ndim == 1: 
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError("audio samples must be a 1D or 2D array")
    if audio.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, got {audio.shape[1]}")
    return audio.copy()
