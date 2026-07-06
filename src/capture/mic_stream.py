"""Capture mikrofon lintas platform memakai sounddevice."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re                # regular expression untuk normalisasi nama device
from typing import Any
import sounddevice as sd # capture audio lintas platform

from src.capture.queued_stream import QueuedAudioStream, StreamFactory

_log = logging.getLogger(__name__)

from src.capture.models import MicrophoneConfig


class MicrophoneStream(QueuedAudioStream):
    """Capture audio mikrofon ke queue berisi AudioFrame."""

    def __init__(
        self,
        config: MicrophoneConfig | None = None,
        *,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self.config = config or MicrophoneConfig()
        
        selected_device_info = sd.query_devices(self.config.device, kind="input")
        
        actual_sample_rate = self.config.sample_rate or int(selected_device_info['default_samplerate'])
        # RC#2: Force mono capture for mic -- mic should always be mono.
        # Capturing multi-channel and then averaging in preprocessing wastes
        # bandwidth; take mono directly from the device.
        actual_channels = self.config.channels or 1
        
        stream_kwargs: dict[str, Any] = {
            "device": self.config.device,
            "latency": self.config.latency,
        }
        _log.info(
            "microphone device selected: name=%r device=%r channels=%s sample_rate=%s block_size=%s",
            selected_device_info.get("name", "default"),
            self.config.device,
            actual_channels,
            actual_sample_rate,
            self.config.block_size,
        )

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
def list_input_devices(*, concise: bool = False, include_system_aliases: bool = False) -> list[dict[str, Any]]:
    """Kembalikan daftar device input sounddevice.

    Windows mengekspos mikrofon fisik yang sama melalui beberapa host API
    (MME, DirectSound, WASAPI, WDM-KS). Mode concise menjaga UI tetap ringkas
    dengan memilih endpoint yang paling masuk akal.
    """

    hostapis = list(sd.query_hostapis())
    try:
        default_input = int(sd.default.device[0])
    except Exception:
        default_input = -1

    results: list[dict[str, Any]] = []
    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        if not include_system_aliases and _looks_like_non_microphone_input(str(device.get("name", ""))):
            continue
        hostapi_index = int(device.get("hostapi", -1))
        hostapi_name = _hostapi_name(hostapis, hostapi_index)
        item = {"index": index, **dict(device)}
        item["hostapi_name"] = hostapi_name
        item["is_default"] = index == default_input
        item["label"] = _device_label(item)
        item["quality_rank"] = _hostapi_rank(hostapi_name)
        results.append(item)

    if not concise:
        return results
    return _dedupe_input_devices(results)


def _hostapi_name(hostapis: list[dict[str, Any]], index: int) -> str:
    if 0 <= index < len(hostapis):
        return str(hostapis[index].get("name", f"hostapi-{index}"))
    return f"hostapi-{index}"


def _hostapi_rank(hostapi_name: str) -> int:
    name = hostapi_name.upper()
    if "WASAPI" in name:
        return 0
    if "DIRECTSOUND" in name:
        return 1
    if "MME" in name:
        return 2
    if "WDM-KS" in name:
        return 3
    return 4


def _normalize_device_name(name: str) -> str:
    normalized = name.lower()
    normalized = normalized.replace("realtek(r) audio", "")
    normalized = normalized.replace("realtek hd audio", "")
    normalized = normalized.replace("mic array input", "")
    normalized = normalized.replace("mic input", "")
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    if "headset" in normalized and ("microphone" in normalized or "mic" in normalized):
        return "headset microphone"
    if "microphone array" in normalized:
        return "microphone array"
    if "microphone" in normalized:
        return "microphone"
    return normalized


def _looks_like_non_microphone_input(name: str) -> bool:
    lowered = name.lower()
    blocked_fragments = (
        "microsoft sound mapper",
        "primary sound capture driver",
        "stereo mix",
        "what u hear",
        "loopback",
        "monitor of",
    )
    if any(fragment in lowered for fragment in blocked_fragments):
        return True
    if "speaker" in lowered or "output" in lowered:
        return "microphone" not in lowered and "mic" not in lowered
    return False


def _device_label(device: dict[str, Any]) -> str:
    name = str(device.get("name", f"device-{device.get('index')}"))
    hostapi_name = str(device.get("hostapi_name", ""))
    default = " default" if device.get("is_default") else ""
    host = f" [{hostapi_name}]" if hostapi_name else ""
    return f"{name}{host}{default}"


def _dedupe_input_devices(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pilih satu representasi terbaik untuk tiap mikrofon fisik."""
    def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
        return (
            0 if item.get("is_default") else 1,
            int(item.get("quality_rank", 9)),
            str(item.get("name", "")),
        )

    selected: dict[str, dict[str, Any]] = {}
    for device in sorted(devices, key=sort_key):
        name = str(device.get("name", ""))
        key = _normalize_device_name(name)
        if not key:
            key = str(device.get("index"))
        if key not in selected:
            selected[key] = device

    concise = sorted(selected.values(), key=sort_key)
    return concise
