from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AudioFormat:
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16


@dataclass(slots=True)
class AudioFrame:
    timestamp_seconds: float
    frame_index: int
    audio_format: AudioFormat
    samples: list[int] = field(default_factory=list)


@dataclass(slots=True)
class AudioChunk:
    start_seconds: float
    end_seconds: float
    start_frame_index: int
    end_frame_index: int
    audio_format: AudioFormat
    samples: list[int] = field(default_factory=list)
