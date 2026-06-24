"""Windows system-audio capture via WASAPI loopback recorder."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Any

import numpy as np
import soundcard as sc
import sounddevice as sd

from src.capture.audio_frame import AudioFrame, normalize_samples
from src.capture.errors import CaptureNotSupportedError


@dataclass(frozen=True, slots=True)
class WasapiDevice:
    index: int
    name: str
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class WindowsLoopbackConfig:
    sample_rate: int | None = None
    channels: int | None = None
    block_size: int = 1_024
    device: int | str | None = None
    queue_size: int = 64


@dataclass(frozen=True, slots=True)
class SoundcardLoopbackDevice:
    index: int
    name: str
    channels: int
    raw_device: Any


class WindowsLoopbackStream:
    """Capture speaker/system audio from a real WASAPI loopback microphone."""

    def __init__(
        self,
        config: WindowsLoopbackConfig | None = None,
        *,
        soundcard_module: Any = sc,
        system_name: str | None = None,
    ) -> None:
        self.config = config or WindowsLoopbackConfig()
        self._soundcard = soundcard_module
        self._system_name = system_name
        self.device = select_soundcard_loopback_device(self.config.device, soundcard_module=soundcard_module)
        self.sample_rate = self.config.sample_rate or 48_000
        self.channels = self.config.channels or self.device.channels
        self.block_size = self.config.block_size
        self.frames_received = 0
        self.frames_dropped = 0
        self._queue: Queue[AudioFrame] = Queue(maxsize=self.config.queue_size)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._recorder: Any | None = None

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.config.queue_size <= 0:
            raise ValueError("queue_size must be positive")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        system_name = self._system_name or platform.system()
        if system_name != "Windows":
            raise CaptureNotSupportedError("WASAPI loopback capture is only supported on Windows")
        if self._thread is not None:
            return

        try:
            self._recorder = self.device.raw_device.recorder(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
            )
            self._stop_event.clear()
            self._thread = Thread(target=self._record_loop, name="windows-loopback-capture", daemon=True)
            self._thread.start()
        except Exception as exc:
            self._thread = None
            self._recorder = None
            raise CaptureNotSupportedError(
                "failed to start WASAPI loopback capture with soundcard; "
                f"selected device={self.device.index} ({self.device.name}). Original error: {exc}"
            ) from exc

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        self._recorder = None

    def read_frame(self, timeout: float | None = None) -> AudioFrame | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def drain(self) -> list[AudioFrame]:
        frames: list[AudioFrame] = []
        while True:
            try:
                frames.append(self._queue.get_nowait())
            except Empty:
                return frames

    def __enter__(self) -> WindowsLoopbackStream:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def _record_loop(self) -> None:
        assert self._recorder is not None
        try:
            with self._recorder as recorder:
                while not self._stop_event.is_set():
                    samples = recorder.record(numframes=self.block_size)
                    if samples is None:
                        continue
                    self._push_samples(samples)
        except Exception as exc:
            self._put_latest(
                AudioFrame(
                    source="speaker",
                    samples=np.zeros((0, self.channels), dtype=np.float32),
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    timestamp_seconds=perf_counter(),
                    status=f"loopback recorder failed: {exc}",
                )
            )

    def _push_samples(self, samples: Any) -> None:
        audio = normalize_samples(samples, self.channels)
        if audio.shape[0] == 0:
            return
        frame = AudioFrame(
            source="speaker",
            samples=audio,
            sample_rate=self.sample_rate,
            channels=self.channels,
            timestamp_seconds=perf_counter(),
        )
        self.frames_received += frame.frame_count
        self._put_latest(frame)

    def _put_latest(self, frame: AudioFrame) -> None:
        try:
            self._queue.put_nowait(frame)
            return
        except Full:
            self.frames_dropped += frame.frame_count

        try:
            self._queue.get_nowait()
        except Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except Full:
            self.frames_dropped += frame.frame_count


def select_soundcard_loopback_device(
    preferred_device: int | str | None = None,
    *,
    soundcard_module: Any = sc,
) -> SoundcardLoopbackDevice:
    """Select a soundcard loopback microphone for Windows speaker capture."""

    microphones = list(soundcard_module.all_microphones(include_loopback=True))
    loopbacks = [
        (index, device)
        for index, device in enumerate(microphones)
        if bool(getattr(device, "isloopback", False))
    ]
    if not loopbacks:
        raise CaptureNotSupportedError("no Windows loopback recording device is available")

    if isinstance(preferred_device, int):
        for index, device in loopbacks:
            if index == preferred_device:
                return _to_soundcard_loopback_device(index, device)
        raise CaptureNotSupportedError(f"loopback device index {preferred_device} was not found")

    if isinstance(preferred_device, str):
        needle = preferred_device.lower()
        for index, device in loopbacks:
            if needle in str(getattr(device, "name", "")).lower():
                return _to_soundcard_loopback_device(index, device)
        raise CaptureNotSupportedError(f"loopback device matching '{preferred_device}' was not found")

    default_speaker_name = str(getattr(soundcard_module.default_speaker(), "name", ""))
    for index, device in loopbacks:
        if default_speaker_name and default_speaker_name.lower() in str(getattr(device, "name", "")).lower():
            return _to_soundcard_loopback_device(index, device)

    index, device = loopbacks[0]
    return _to_soundcard_loopback_device(index, device)


def _to_soundcard_loopback_device(index: int, device: Any) -> SoundcardLoopbackDevice:
    return SoundcardLoopbackDevice(
        index=index,
        name=str(getattr(device, "name", f"loopback-{index}")),
        channels=max(1, int(getattr(device, "channels", 2))),
        raw_device=device,
    )


def select_wasapi_output_device(
    preferred_device: int | str | None = None,
    *,
    sd_module: Any = sd,
) -> WasapiDevice:
    """Select a Windows WASAPI output device for speaker capture."""

    hostapis = list(sd_module.query_hostapis())
    wasapi_index = next(
        (index for index, api in enumerate(hostapis) if "WASAPI" in str(api.get("name", "")).upper()),
        None,
    )
    if wasapi_index is None:
        raise CaptureNotSupportedError("Windows WASAPI host API is not available")

    devices = list(sd_module.query_devices())
    wasapi_api = hostapis[wasapi_index]

    if preferred_device is None:
        candidate_indices = [int(wasapi_api.get("default_output_device", -1))]
    elif isinstance(preferred_device, int):
        candidate_indices = [preferred_device]
    else:
        needle = preferred_device.lower()
        candidate_indices = [
            index
            for index, device in enumerate(devices)
            if int(device.get("hostapi", -1)) == wasapi_index and needle in str(device.get("name", "")).lower()
        ]

    for index in candidate_indices:
        if index < 0 or index >= len(devices):
            continue
        device = devices[index]
        if int(device.get("hostapi", -1)) != wasapi_index:
            continue
        output_channels = int(device.get("max_output_channels", 0))
        if output_channels <= 0:
            continue
        return WasapiDevice(
            index=index,
            name=str(device.get("name", f"device-{index}")),
            sample_rate=int(float(device.get("default_samplerate", 48_000))),
            channels=output_channels,
        )

    raise CaptureNotSupportedError("no WASAPI output device matched the requested selection")


def list_wasapi_output_devices(sd_module: Any = sd) -> list[WasapiDevice]:
    """Return WASAPI output devices that can be candidates for loopback capture."""

    hostapis = list(sd_module.query_hostapis())
    wasapi_index = next(
        (index for index, api in enumerate(hostapis) if "WASAPI" in str(api.get("name", "")).upper()),
        None,
    )
    if wasapi_index is None:
        return []

    devices = list(sd_module.query_devices())
    results: list[WasapiDevice] = []
    for index, device in enumerate(devices):
        if int(device.get("hostapi", -1)) != wasapi_index:
            continue
        output_channels = int(device.get("max_output_channels", 0))
        if output_channels <= 0:
            continue
        results.append(
            WasapiDevice(
                index=index,
                name=str(device.get("name", f"device-{index}")),
                sample_rate=int(float(device.get("default_samplerate", 48_000))),
                channels=output_channels,
            )
        )
    return results
