from __future__ import annotations


import logging
from math import gcd
from typing import Any, Iterable
import collections

import numpy as np
from scipy import signal

from src.capture.models import AudioFrame

# yang perlu diketahui:
# RMS merupakan ukuran energi rata-rata dari sinyal audio. RMS yang lebih tinggi berarti audio lebih keras, sedangkan RMS yang lebih rendah berarti audio lebih lembut.
# ASR (Automatic Speech Recognition) adalah proses mengubah sinyal audio menjadi teks. Untuk ASR, kualitas audio sangat penting agar model dapat mengenali ucapan dengan akurat. (Whisper)

from src.preprocessing.models import PreprocessConfig, PreprocessedAudioChunk

_log = logging.getLogger(__name__)


# mengubah audio yang ditangkap menjadi chunk mono 16 kHz untuk ASR.
class AudioPreprocessor:

    def __init__(
        self,
        config: PreprocessConfig | None = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self._validate_config()
        self.last_decisions: list[dict[str, Any]] = []
        self._prev_gate_gain: collections.defaultdict[str, float] = collections.defaultdict(lambda: 1.0)
        self._filter_zi: dict[str, np.ndarray] = {}
        self._sos = signal.butter(
            self.config.highpass_order,
            self.config.highpass_cutoff_hz,
            btype="highpass",
            fs=self.config.target_sample_rate,
            output="sos",
        )

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
        """Run cast, mono, resample, noise gate, high-pass, normalization, and clip.
        
        Untuk respons gating yang dinamis, fungsi ini sebaiknya dipanggil
        dengan buffer pendek secara kontinu (misal 500ms - 2.5s), bukan blok
        durasi panjang, karena noise gate dihitung rata-rata per-pemanggilan.
        """

        self.last_decisions = []
        audio = _as_float32_matrix(samples, channels)
        mono = _stereo_to_mono(audio)
        resampled = _resample(mono, sample_rate, self.config.target_sample_rate)
        denoised = self._apply_noise_gate(resampled, source=source)
        filtered = self._apply_highpass_filter(
            denoised,
            source=source,
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
            normalized = _normalize_rms(
                chunk,
                target_db=self.config.target_rms_db,
                max_gain_db=self.config.max_normalization_gain_db,
            )
            clipped = np.clip(normalized, -self.config.clip_limit, self.config.clip_limit).astype(np.float32)
            rms_db = _rms_db(clipped)
            decision_reason = "accepted"
            self._record_decision(
                source=source,
                start_seconds=chunk_start,
                duration_seconds=duration_seconds,
                passed=True,
                reason=decision_reason,
                input_rms_db=input_rms_db,
                output_rms_db=rms_db,
                noise_reduction_enabled=self.config.noise_reduction_enabled,
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

    def _apply_noise_gate(self, samples: np.ndarray, source: str) -> np.ndarray:
        if not self.config.noise_reduction_enabled or samples.size == 0:
            self._prev_gate_gain[source] = 1.0
            return samples.astype(np.float32, copy=True)
            
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Noise reduction expects mono 1D samples")
            
        # Asumsi: input audio berada dalam rentang float32 [-1.0, 1.0]
        # Menghitung RMS energi dari chunk saat ini
        rms = float(np.sqrt(np.mean(np.square(audio))))
        
        # Menentukan target gain (Downward Expander dengan floor)
        threshold = self.config.noise_gate_threshold_rms
        if rms >= threshold:
            target_gain = 1.0
        elif rms < threshold / 10:
            # Silence mutlak jika sangat jauh di bawah threshold
            target_gain = 0.0
        else:
            # Proportional attenuation (Downward Expander)
            target_gain = rms / threshold

        # Mencegah artifact pumping/klik di boundary chunk dengan membuat envelope gain
        # yang bertransisi secara linier dari gain chunk sebelumnya ke target gain
        gain_envelope = np.linspace(self._prev_gate_gain[source], target_gain, audio.size, dtype=np.float32)
        
        # Simpan state untuk chunk berikutnya
        self._prev_gate_gain[source] = target_gain
        
        return audio * gain_envelope

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
        self.last_decisions.append(decision)

    def _validate_config(self) -> None:
        if self.config.noise_gate_threshold_rms <= 0:
            raise ValueError("noise_gate_threshold_rms must be positive")
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

    def _apply_highpass_filter(self, samples: np.ndarray, *, source: str) -> np.ndarray:
        if samples.size == 0:
            return samples.astype(np.float32)
        
        # Gunakan state awal jika ada
        zi = self._filter_zi.get(source)
        if zi is None:
            zi = signal.sosfilt_zi(self._sos) * samples[0]
            
        filtered, zf = signal.sosfilt(self._sos, samples, zi=zi)
        self._filter_zi[source] = zf
        return filtered.astype(np.float32)


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




def _normalize_rms(samples: np.ndarray, *, target_db: float, max_gain_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms <= 1e-8:
        return samples.astype(np.float32, copy=True)
    target = 10 ** (target_db / 20)
    gain = min(target / rms, 10 ** (max_gain_db / 20))
    return (samples * gain).astype(np.float32)



def _rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 1e-8:
        return float("-inf")
    return float(20 * np.log10(rms))
