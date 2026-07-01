"""Voice activity detection filters for preprocessed audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class VoiceActivityDetector(Protocol):
    """Small VAD protocol used by the preprocessing pipeline."""

    def is_speech(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Return True when samples contain speech-like energy."""


@dataclass(frozen=True, slots=True)
class EnergyVad:
    """Fast local VAD based on RMS energy with speech-fraction gating.

    Checks both overall energy (rms + peak) **and** the fraction of
    short sub-frames that exceed the RMS threshold.  This prevents a
    single noise spike from passing a chunk that is otherwise silent,
    and stops monotone background noise (fan, AC, electronics) from
    triggering Whisper's looping-hallucination pattern.

    Default thresholds are calibrated for typical indoor environments
    where background noise measures around 0.005-0.008 RMS.  Set
    *speech_fraction* to 0.0 to disable the fraction check.

    Changed from original:
      - rms_threshold  : 0.003 -> 0.01  (filters AC/fan hum)
      - peak_threshold : 0.01  -> 0.04  (filters low noise bursts)
      - speech_fraction: NEW   = 0.30   (30% of 20-ms sub-frames must be active)
    """

    rms_threshold: float = 0.01     # overall RMS gate
    peak_threshold: float = 0.04    # peak amplitude gate
    speech_fraction: float = 0.30   # min active sub-frame fraction (0.0 = disabled)

    def is_speech(self, samples: np.ndarray, sample_rate: int) -> bool:
        if sample_rate <= 0 or samples.size == 0:
            return False
        audio = np.asarray(samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))

        # Gate 1: overall energy must clear both thresholds
        if rms < self.rms_threshold or peak < self.peak_threshold:
            return False

        # Gate 2 (speech-fraction): majority-of-sub-frames check.
        # Split the chunk into ~20 ms windows; require that at least
        # *speech_fraction* of those windows exceed rms_threshold.
        # This rejects monotone noise that has a high overall RMS but
        # no dynamic speech-like variation.
        if self.speech_fraction > 0.0:
            frame_size = max(1, int(round(0.02 * sample_rate)))  # 20 ms
            n_frames = audio.shape[0] // frame_size
            if n_frames >= 3:
                active = sum(
                    1
                    for i in range(n_frames)
                    if float(np.sqrt(np.mean(np.square(
                        audio[i * frame_size:(i + 1) * frame_size]
                    )))) >= self.rms_threshold
                )
                if active / n_frames < self.speech_fraction:
                    return False

        return True


class SileroVad:
    """Lazy Silero VAD adapter.

    The heavy torch/Silero dependency is deliberately not loaded at import
    time. It can be used later by installing torch and passing this detector
    into AudioPreprocessor.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None
        self._get_speech_timestamps = None

    def is_speech(self, samples: np.ndarray, sample_rate: int) -> bool:
        if samples.size == 0:
            return False
        self._ensure_loaded()

        import torch

        audio = np.asarray(samples, dtype=np.float32)
        tensor = torch.from_numpy(audio)
        timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
        )
        return bool(timestamps)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._get_speech_timestamps is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("SileroVad requires torch to be installed") from exc

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]
