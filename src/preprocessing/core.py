from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Iterable

import numpy as np
from scipy import signal

from src.capture.models import AudioFrame

# yang perlu diketahui:
# RMS merupakan ukuran energi rata-rata dari sinyal audio. RMS yang lebih tinggi berarti audio lebih keras, sedangkan RMS yang lebih rendah berarti audio lebih lembut.
# ASR (Automatic Speech Recognition) adalah proses mengubah sinyal audio menjadi teks. Untuk ASR, kualitas audio sangat penting agar model dapat mengenali ucapan dengan akurat. (Whisper)

from src.preprocessing.models import PreprocessConfig, PreprocessedAudioChunk
# mengubah audio yang ditangkap menjadi chunk mono 16 kHz untuk ASR.
class AudioPreprocessor:

    def __init__(
        self,
        config: PreprocessConfig | None = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self._validate_config()
        self.last_decisions: list[dict[str, Any]] = []

    # menggabungkan frame audio
    # jadi intinya fungsi ini menggabungkan frame audio yang ditangkap menjadi satu chunk audio untuk diproses lebih lanjut, agar chunk audio yang dikirim ke ars tidak terlalu pendek, sehingga bisa diproses dengan baik oleh ars. 
    def preprocess_frames(self, frames: Iterable[AudioFrame]) -> list[PreprocessedAudioChunk]:
        frame_list = [frame for frame in frames if frame.frame_count > 0] 
        if not frame_list:
            return []

        first = frame_list[0] 
        sample_rate = first.sample_rate 
        channels = first.channels
        source = first.source
        start_seconds = first.timestamp_seconds

        for frame in frame_list:
            if frame.sample_rate != sample_rate:
                raise ValueError("all frames for one preprocessing pass must share sample_rate")
            if frame.channels != channels:
                raise ValueError("all frames for one preprocessing pass must share channel count")
            if frame.source != source:
                raise ValueError("all frames for one preprocessing pass must share source")

        samples = np.concatenate([frame.samples for frame in frame_list], axis=0) # menggabungkan semua frame menjadi satu array samples
        # mengembalikan hasil preprocessing audio yang telah digabungkan menjadi chunk mono 16 kHz untuk ASR.
    
        return self.preprocess_samples(
            samples,
            sample_rate=sample_rate,
            channels=channels,
            source=source,
            start_seconds=start_seconds,
        )

    def preprocess_samples(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        channels: int,
        source: str,
        start_seconds: float = 0.0,
    ) -> list[PreprocessedAudioChunk]:
        """Run cast, mono, resample, high-pass, normalization, and clip."""

        self.last_decisions = []
        audio = _as_float32_matrix(samples, channels)
        mono = _stereo_to_mono(audio)
        resampled = _resample(mono, sample_rate, self.config.target_sample_rate)
        filtered = _highpass_filter(
            resampled,
            sample_rate=self.config.target_sample_rate,
            cutoff_hz=self.config.highpass_cutoff_hz,
            order=self.config.highpass_order,
        )

        chunks: list[PreprocessedAudioChunk] = []
        chunk_size = int(round(self.config.chunk_seconds * self.config.target_sample_rate))
        for index, start in enumerate(range(0, filtered.shape[0], chunk_size)):
            # Client hanya melakukan transform format audio. Keputusan
            # speech/no-speech adalah tanggung jawab server VAD.
            chunk = filtered[start : start + chunk_size]
            if chunk.size == 0:
                continue
            duration_seconds = chunk.shape[0] / self.config.target_sample_rate
            chunk_start = start_seconds + (index * self.config.chunk_seconds)
            if duration_seconds < self.config.min_chunk_seconds:
                self._record_decision(
                    source=source,
                    start_seconds=chunk_start,
                    duration_seconds=duration_seconds,
                    passed=False,
                    reason="too_short",
                )
                continue
            input_rms_db = _rms_db(chunk)
            if self._should_drop_legacy_silence(chunk, input_rms_db):
                self._record_decision(
                    source=source,
                    start_seconds=chunk_start,
                    duration_seconds=duration_seconds,
                    passed=False,
                    reason="below_min_rms",
                    input_rms_db=input_rms_db,
                    noise_reduction_enabled=self.config.noise_reduction_enabled,
                    client_vad_enabled=self.config.client_vad_enabled,
                )
                continue
            normalized = _normalize_rms(
                chunk,
                target_db=self.config.target_rms_db,
                max_gain_db=self.config.max_normalization_gain_db,
            )
            denoised = _apply_noise_reduction(
                normalized,
                enabled=self.config.noise_reduction_enabled,
                strength=self.config.noise_reduction_strength,
            )
            clipped = np.clip(denoised, -self.config.clip_limit, self.config.clip_limit).astype(np.float32)
            rms_db = _rms_db(clipped)
            decision_reason = "accepted"
            if not self.config.client_vad_enabled:
                decision_reason = "vad_disabled_accepted"
            self._record_decision(
                source=source,
                start_seconds=chunk_start,
                duration_seconds=duration_seconds,
                passed=True,
                reason=decision_reason,
                input_rms_db=input_rms_db,
                output_rms_db=rms_db,
                noise_reduction_enabled=self.config.noise_reduction_enabled,
                client_vad_enabled=self.config.client_vad_enabled,
            )
            chunks.append(
                PreprocessedAudioChunk(
                    source=source,
                    samples=clipped,
                    sample_rate=self.config.target_sample_rate,
                    start_seconds=chunk_start,
                    duration_seconds=duration_seconds,
                    rms_db=rms_db,
                    input_rms_db=input_rms_db,
                )
            )

        return chunks

    def _record_decision(
        self,
        *,
        source: str,
        start_seconds: float,
        duration_seconds: float,
        passed: bool,
        reason: str,
        input_rms_db: float | None = None,
        output_rms_db: float | None = None,
        noise_reduction_enabled: bool | None = None,
        client_vad_enabled: bool | None = None,
    ) -> None:
        decision: dict[str, Any] = {
            "source": source,
            "start_seconds": round(start_seconds, 3),
            "duration_seconds": round(duration_seconds, 3),
            "passed": passed,
            "reason": reason,
        }
        if input_rms_db is not None:
            decision["input_rms_db"] = round(input_rms_db, 2)
        if output_rms_db is not None:
            decision["output_rms_db"] = round(output_rms_db, 2)
        if noise_reduction_enabled is not None:
            decision["noise_reduction_enabled"] = noise_reduction_enabled
        if client_vad_enabled is not None:
            decision["client_vad_enabled"] = client_vad_enabled
        self.last_decisions.append(decision)

    def _should_drop_legacy_silence(self, chunk: np.ndarray, input_rms_db: float) -> bool:
        if chunk.size == 0:
            return True
        if self.config.min_input_rms_db is None:
            return bool(np.allclose(chunk, 0.0, atol=1e-8))
        return input_rms_db < self.config.min_input_rms_db

    def _validate_config(self) -> None:
        if self.config.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.config.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        if self.config.min_chunk_seconds <= 0:
            raise ValueError("min_chunk_seconds must be positive")
        if self.config.highpass_cutoff_hz <= 0:
            raise ValueError("highpass_cutoff_hz must be positive")
        nyquist = self.config.target_sample_rate / 2
        if self.config.highpass_cutoff_hz >= nyquist:
            raise ValueError("highpass_cutoff_hz must be below Nyquist")
        if self.config.highpass_order <= 0:
            raise ValueError("highpass_order must be positive")
        if not 0 < self.config.clip_limit <= 1:
            raise ValueError("clip_limit must be within (0, 1]")
        if self.config.max_normalization_gain_db < 0:
            raise ValueError("max_normalization_gain_db must be non-negative")


def _as_float32_matrix(samples: np.ndarray, channels: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError("samples must be a 1D or 2D array")
    if audio.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, got {audio.shape[1]}")
    return audio


def _stereo_to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] == 1:
        return samples[:, 0].astype(np.float32, copy=True)
    return np.mean(samples, axis=1, dtype=np.float32)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError("source sample_rate must be positive")
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=True)

    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return signal.resample_poly(samples, up=up, down=down).astype(np.float32)


def _highpass_filter(samples: np.ndarray, *, sample_rate: int, cutoff_hz: float, order: int) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32)

    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    # sosfiltfilt gives better phase behavior, but needs enough samples.
    min_len_for_filtfilt = 3 * (2 * len(sos) + 1)
    if samples.shape[0] > min_len_for_filtfilt:
        return signal.sosfiltfilt(sos, samples).astype(np.float32)
    return signal.sosfilt(sos, samples).astype(np.float32)


def _normalize_rms(samples: np.ndarray, *, target_db: float, max_gain_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms <= 1e-8:
        return samples.astype(np.float32, copy=True)
    target = 10 ** (target_db / 20)
    gain = min(target / rms, 10 ** (max_gain_db / 20))
    return (samples * gain).astype(np.float32)


def _apply_noise_reduction(samples: np.ndarray, *, enabled: bool, strength: float) -> np.ndarray:
    if not enabled or samples.size == 0:
        return samples.astype(np.float32, copy=True)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return samples.astype(np.float32, copy=True)
    noise_floor = np.median(np.abs(samples)) * 0.5
    adjusted = samples.copy()
    adjusted[np.abs(adjusted) < noise_floor] *= 1.0 - strength
    return adjusted.astype(np.float32)


def _rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 1e-8:
        return float("-inf")
    return float(20 * np.log10(rms))
