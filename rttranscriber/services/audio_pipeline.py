from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import floor

from rttranscriber.model.audio import AudioChunk, AudioFormat, AudioFrame


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_BITS_PER_SAMPLE = 16


def _clamp_int16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


class AudioFrameNormalizer:
    """Normalisasi frame ke PCM mono 16 kHz 16-bit."""

    def normalize(self, frame: AudioFrame) -> AudioFrame:
        if frame.audio_format.channels <= 0 or frame.audio_format.sample_rate <= 0:
            raise ValueError("audio format sumber tidak valid")

        mono_samples = self._downmix_to_mono(frame.samples, frame.audio_format.channels)
        normalized_samples = self._resample_linear(mono_samples, frame.audio_format.sample_rate)
        normalized_frame_index = int(frame.frame_index * TARGET_SAMPLE_RATE / frame.audio_format.sample_rate)

        return AudioFrame(
            timestamp_seconds=frame.timestamp_seconds,
            frame_index=normalized_frame_index,
            audio_format=AudioFormat(
                sample_rate=TARGET_SAMPLE_RATE,
                channels=TARGET_CHANNELS,
                bits_per_sample=TARGET_BITS_PER_SAMPLE,
            ),
            samples=normalized_samples,
        )

    def _downmix_to_mono(self, samples: list[int], channels: int) -> list[float]:
        if channels == 1:
            return [float(sample) for sample in samples]

        mono: list[float] = []
        for offset in range(0, len(samples), channels):
            # Downmix dirata-rata agar seluruh channel tetap terwakili.
            chunk = samples[offset : offset + channels]
            mono.append(sum(chunk) / float(channels))
        return mono

    def _resample_linear(self, mono_samples: list[float], source_rate: int) -> list[int]:
        if not mono_samples:
            return []

        if source_rate == TARGET_SAMPLE_RATE:
            return [_clamp_int16(sample) for sample in mono_samples]

        ratio = TARGET_SAMPLE_RATE / float(source_rate)
        output_count = max(1, int(len(mono_samples) * ratio))
        output: list[int] = []
        for out_index in range(output_count):
            source_position = out_index / ratio
            left_index = int(floor(source_position))
            right_index = min(left_index + 1, len(mono_samples) - 1)
            fraction = source_position - left_index
            interpolated = mono_samples[left_index] + (
                (mono_samples[right_index] - mono_samples[left_index]) * fraction
            )
            output.append(_clamp_int16(interpolated))
        return output


@dataclass(slots=True)
class _SamplePoint:
    value: int
    frame_index: int


class TimestampedPcmRingBuffer:
    """Buffer PCM ringan untuk scheduling chunk dengan metadata frame index."""

    def __init__(self, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("kapasitas buffer harus positif")
        self._max_samples = max_samples
        self._samples: deque[_SamplePoint] = deque()
        self._base_seconds: float | None = None
        self._audio_format = AudioFormat()

    def clear(self) -> None:
        self._samples.clear()
        self._base_seconds = None

    def push(self, frame: AudioFrame) -> None:
        if frame.audio_format.sample_rate != TARGET_SAMPLE_RATE or frame.audio_format.channels != 1:
            raise ValueError("buffer hanya menerima frame mono 16 kHz")

        if self._base_seconds is None:
            self._base_seconds = frame.timestamp_seconds
            self._audio_format = frame.audio_format

        for index, sample in enumerate(frame.samples):
            self._samples.append(_SamplePoint(value=sample, frame_index=frame.frame_index + index))

        while len(self._samples) > self._max_samples:
            self._samples.popleft()

    def sample_count(self) -> int:
        return len(self._samples)

    def build_chunk(self, window_samples: int) -> AudioChunk | None:
        if not self._samples:
            return None
        return self.build_chunk_at(self._samples[0].frame_index, window_samples)

    def build_chunk_at(self, start_frame_index: int, window_samples: int) -> AudioChunk | None:
        if self._base_seconds is None or window_samples <= 0 or not self._samples:
            return None

        sample_list = list(self._samples)
        start_offset = None
        for index, point in enumerate(sample_list):
            if point.frame_index == start_frame_index:
                start_offset = index
                break
            if point.frame_index > start_frame_index:
                return None

        if start_offset is None:
            return None

        end_offset = start_offset + window_samples
        if end_offset > len(sample_list):
            return None

        first = sample_list[start_offset]
        last = sample_list[end_offset - 1]
        start_seconds = self._base_seconds + (first.frame_index / self._audio_format.sample_rate)
        end_frame_index = last.frame_index + 1
        end_seconds = self._base_seconds + (end_frame_index / self._audio_format.sample_rate)

        return AudioChunk(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            start_frame_index=first.frame_index,
            end_frame_index=end_frame_index,
            audio_format=self._audio_format,
            samples=[point.value for point in sample_list[start_offset:end_offset]],
        )

    def discard_before(self, start_frame_index: int) -> None:
        while self._samples and self._samples[0].frame_index < start_frame_index:
            self._samples.popleft()

    def consume_until(self, next_start_frame_index: int) -> None:
        self.discard_before(next_start_frame_index)


class SlidingWindowChunkScheduler:
    """Membangun chunk sliding window dari frame audio yang sudah dinormalisasi."""

    def __init__(self, window_seconds: int = 6, hop_seconds: int = 2, buffer_seconds: int = 30) -> None:
        if window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("window dan hop harus positif")
        if hop_seconds > window_seconds:
            raise ValueError("hop tidak boleh lebih besar dari window")

        self.window_samples = window_seconds * TARGET_SAMPLE_RATE
        self.hop_samples = hop_seconds * TARGET_SAMPLE_RATE
        self._buffer = TimestampedPcmRingBuffer(TARGET_SAMPLE_RATE * buffer_seconds)
        self._next_chunk_start_frame_index: int | None = None

    def push(self, frame: AudioFrame) -> list[AudioChunk]:
        self._buffer.push(frame)
        if self._next_chunk_start_frame_index is None:
            self._next_chunk_start_frame_index = frame.frame_index

        produced: list[AudioChunk] = []
        while self._next_chunk_start_frame_index is not None:
            chunk = self._buffer.build_chunk_at(self._next_chunk_start_frame_index, self.window_samples)
            if chunk is None:
                break
            produced.append(chunk)
            self._next_chunk_start_frame_index += self.hop_samples
            self._buffer.discard_before(self._next_chunk_start_frame_index)
        return produced
