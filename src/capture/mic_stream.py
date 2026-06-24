"""Universal microphone capture using sounddevice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sounddevice as sd

from src.capture.queued_stream import QueuedAudioStream, StreamFactory


@dataclass(frozen=True, slots=True)
class MicrophoneConfig:
    sample_rate: int | None = None  
    channels: int | None = None     
    block_size: int = 1_024
    dtype: str = "float32"
    latency: str | float | None = "low"
    device: int | str | None = None
    queue_size: int = 64


class MicrophoneStream(QueuedAudioStream):
    """Capture microphone audio into a queue of AudioFrame objects."""

    def __init__(
        self,
        config: MicrophoneConfig | None = None,
        *,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self.config = config or MicrophoneConfig()
        
        default_device_info = sd.query_devices(kind='input')
        
        actual_sample_rate = self.config.sample_rate or int(default_device_info['default_samplerate'])
        actual_channels = self.config.channels or int(default_device_info['max_input_channels'])
        
        stream_kwargs: dict[str, Any] = {
            "device": self.config.device,
            "latency": self.config.latency,
        }

        super().__init__(
            source="mic",
            sample_rate=actual_sample_rate,  
            channels=actual_channels,        
            block_size=self.config.block_size,
            dtype=self.config.dtype,
            queue_size=self.config.queue_size,
            stream_factory=stream_factory or sd.InputStream,
            stream_kwargs=stream_kwargs,
        )
def list_input_devices() -> list[dict[str, Any]]:
    """Return input-capable sounddevice devices."""

    devices = sd.query_devices()
    return [
        {"index": index, **dict(device)}
        for index, device in enumerate(devices)
        if int(device.get("max_input_channels", 0)) > 0
    ]
