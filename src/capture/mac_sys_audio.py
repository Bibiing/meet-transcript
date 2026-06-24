"""macOS system-audio capture adapter for ScreenCaptureKit."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any, Protocol

from src.capture.audio_frame import AudioFrame, normalize_samples
from src.capture.errors import CaptureNotSupportedError
from src.capture.queued_stream import Clock, QueuedAudioStream


@dataclass(frozen=True, slots=True)
class MacSystemAudioConfig:
    sample_rate: int = 48_000
    channels: int = 2
    block_size: int = 1_024
    dtype: str = "float32"
    queue_size: int = 64
    excludes_current_process_audio: bool = True


class ScreenCaptureKitProvider(Protocol):
    """Small provider boundary around pyobjc ScreenCaptureKit integration."""

    def start(self, callback: Any) -> Any:
        """Start native capture and call callback(samples, status)."""


class PyObjCScreenCaptureKitProvider:
    """Lazy loader for ScreenCaptureKit.

    A concrete native delegate can be added behind this provider without
    changing the queue contract used by the rest of the application.
    """

    def __init__(self, config: MacSystemAudioConfig) -> None:
        self.config = config

    def start(self, callback: Any) -> Any:
        try:
            __import__("ScreenCaptureKit")
        except ImportError as exc:
            raise CaptureNotSupportedError(
                "pyobjc-framework-ScreenCaptureKit is required for macOS system audio capture"
            ) from exc
        raise CaptureNotSupportedError(
            "ScreenCaptureKit provider is available but native SCStream delegate wiring "
            "has not been enabled in this runtime"
        )


class MacSystemAudioStream(QueuedAudioStream):
    """Queue-backed macOS system audio stream boundary."""

    def __init__(
        self,
        config: MacSystemAudioConfig | None = None,
        *,
        provider: ScreenCaptureKitProvider | None = None,
        system_name: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or MacSystemAudioConfig()
        self._provider = provider or PyObjCScreenCaptureKitProvider(self.config)
        self._system_name = system_name
        self._native_stream: Any | None = None
        super().__init__(
            source="speaker",
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            block_size=self.config.block_size,
            dtype=self.config.dtype,
            queue_size=self.config.queue_size,
            stream_factory=self._start_provider,
            clock=clock or __import__("time").perf_counter,
        )

    def start(self) -> None:
        system_name = self._system_name or platform.system()
        if system_name != "Darwin":
            raise CaptureNotSupportedError("ScreenCaptureKit system audio capture is only supported on macOS")
        super().start()

    def _start_provider(self, **_: Any) -> Any:
        self._native_stream = self._provider.start(self._on_provider_audio)
        return self._NativeStreamHandle(self._native_stream)

    def _on_provider_audio(self, samples: Any, status: str = "") -> None:
        audio = normalize_samples(samples, self.channels)
        frame = AudioFrame(
            source="speaker",
            samples=audio,
            sample_rate=self.sample_rate,
            channels=self.channels,
            timestamp_seconds=self._clock(),
            status=status,
        )
        self.frames_received += frame.frame_count
        self._put_latest(frame)

    class _NativeStreamHandle:
        def __init__(self, native_stream: Any) -> None:
            self._native_stream = native_stream

        def start(self) -> None:
            start = getattr(self._native_stream, "start", None)
            if callable(start):
                start()

        def stop(self) -> None:
            stop = getattr(self._native_stream, "stop", None)
            if callable(stop):
                stop()

        def close(self) -> None:
            close = getattr(self._native_stream, "close", None)
            if callable(close):
                close()
