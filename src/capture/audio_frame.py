"""Shared audio frame model for capture modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


AudioSource = Literal["mic", "speaker"]


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A raw audio block captured from one source."""

    source: AudioSource
    samples: np.ndarray
    sample_rate: int
    channels: int
    timestamp_seconds: float
    status: str = ""

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate


def normalize_samples(samples: np.ndarray, channels: int) -> np.ndarray:
    """Return samples as copied float32 array with shape (frames, channels)."""

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError("audio samples must be a 1D or 2D array")
    if audio.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, got {audio.shape[1]}")
    return audio.copy()
