from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Iterable

import numpy as np
from scipy import signal

from src.capture.audio_frame import AudioFrame
from src.engine.vad_filter import EnergyVad, VoiceActivityDetector

# yang perlu diketahui:
# RMS merupakan ukuran energi rata-rata dari sinyal audio. RMS yang lebih tinggi berarti audio lebih keras, sedangkan RMS yang lebih rendah berarti audio lebih lembut.
# ASR (Automatic Speech Recognition) adalah proses mengubah sinyal audio menjadi teks. Untuk ASR, kualitas audio sangat penting agar model dapat mengenali ucapan dengan akurat. (Whisper)

# konfigurasi default untuk preprocessing audio.
@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    target_sample_rate: int = 16_000            # 16000 Hz untuk Whisper
    chunk_seconds: float = 2.5                  # durasi chunk audio untuk dikirim ke WhisperLive
    highpass_cutoff_hz: float = 80.0            # frekuensi potong high-pass filter, > mengurangi noise low-frequency, < mengurangi kualitas suara. 
    highpass_order: int = 2                     # order filter untuk high-pass filter. 2 karena order 2 memberikan roll-off yang cukup untuk mengurangi noise low-frequency, tapi tidak terlalu tinggi sehingga tidak merusak kualitas suara.             
    target_rms_db: float = -20.0                # target RMS level untuk normalisasi audio. -20 dBFS adalah level yang baik untuk audio manusia, karena memberikan headroom yang cukup untuk puncak suara tanpa distorsi.
    clip_limit: float = 0.95                    # batas clipping untuk normalisasi audio. 0.95 berarti bahwa amplitudo audio akan dibatasi pada 95% dari nilai maksimum, untuk menghindari distorsi.
    vad_rms_threshold: float = 0.015            # threshold RMS untuk VAD. 0.015 adalah nilai yang baik untuk mendeteksi suara manusia, karena lebih sensitif terhadap suara yang lebih lembut.
    vad_peak_threshold: float = 0.05            # threshold puncak untuk VAD. 0.05 adalah nilai yang baik untuk mendeteksi suara manusia, karena lebih sensitif terhadap puncak suara yang lebih tinggi.
    vad_speech_fraction: float = 0.30           # fraksi minimum frame yang harus terdeteksi sebagai suara untuk dianggap sebagai chunk suara. 0.30 berarti bahwa setidaknya 30% dari frame dalam chunk harus terdeteksi sebagai suara agar chunk tersebut dianggap sebagai chunk suara. 
    min_input_rms_db: float = -42.0             # minimum RMS level untuk input audio agar diterima. -42 dBFS adalah level yang baik untuk audio manusia, karena lebih sensitif terhadap suara yang lebih lembut.
    min_chunk_seconds: float = 1.0              # durasi minimum chunk audio agar diterima. 1.0 detik adalah durasi yang baik untuk audio manusia, karena memberikan cukup waktu untuk mendeteksi suara manusia.
    max_normalization_gain_db: float = 24.0     # gain maksimum untuk normalisasi audio. 24 dB adalah gain yang baik untuk audio manusia, karena memberikan headroom yang cukup untuk puncak suara tanpa distorsi.


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

# mengubah audio yang ditangkap menjadi chunk mono 16 kHz untuk ASR.
class AudioPreprocessor:

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        *,
        vad: VoiceActivityDetector | None = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self._validate_config()
        self.vad = vad or EnergyVad(
            rms_threshold=self.config.vad_rms_threshold,
            peak_threshold=self.config.vad_peak_threshold,
            speech_fraction=self.config.vad_speech_fraction,
        )
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
        """Run cast, mono, resample, high-pass, normalization, clip, and VAD."""

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
            # Setiap chunk dievaluasi sendiri. Chunk pendek, silence, atau RMS
            # terlalu rendah tidak dikirim agar Whisper tidak halusinasi.
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
            if not self.vad.is_speech(chunk, self.config.target_sample_rate):
                self._record_decision(
                    source=source,
                    start_seconds=chunk_start,
                    duration_seconds=duration_seconds,
                    passed=False,
                    reason="vad_silence",
                    input_rms_db=_rms_db(chunk),
                )
                continue
            input_rms_db = _rms_db(chunk)
            if input_rms_db < self.config.min_input_rms_db:
                self._record_decision(
                    source=source,
                    start_seconds=chunk_start,
                    duration_seconds=duration_seconds,
                    passed=False,
                    reason="below_min_rms",
                    input_rms_db=input_rms_db,
                )
                continue

            normalized = _normalize_rms(
                chunk,
                target_db=self.config.target_rms_db,
                max_gain_db=self.config.max_normalization_gain_db,
            )
            clipped = np.clip(normalized, -self.config.clip_limit, self.config.clip_limit).astype(np.float32)
            rms_db = _rms_db(clipped)
            self._record_decision(
                source=source,
                start_seconds=chunk_start,
                duration_seconds=duration_seconds,
                passed=True,
                reason="accepted",
                input_rms_db=input_rms_db,
                output_rms_db=rms_db,
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
        self.last_decisions.append(decision)

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


def _rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 1e-8:
        return float("-inf")
    return float(20 * np.log10(rms))
