"""Helper stream berbasis queue untuk callback capture realtime."""

from __future__ import annotations

from queue import Empty, Full, Queue
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.capture.audio_frame import AudioFrame, AudioSource, normalize_samples
from src.capture.errors import CaptureError


StreamFactory = Callable[..., Any]
Clock = Callable[[], float]


class QueuedAudioStream:
    """Bungkus audio callback stream dan expose pembacaan frame non-blocking."""

    def __init__(
        self,
        *,
        source: AudioSource,
        sample_rate: int,
        channels: int,
        block_size: int,
        dtype: str,
        queue_size: int,
        stream_factory: StreamFactory,
        stream_kwargs: dict[str, Any] | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")

        self.source = source
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.dtype = dtype
        self.frames_received = 0
        self.frames_dropped = 0
        self._queue: Queue[AudioFrame] = Queue(maxsize=queue_size)
        self._stream_factory = stream_factory
        self._stream_kwargs = stream_kwargs or {}
        self._clock = clock
        self._stream: Any | None = None

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Create and start the underlying audio stream."""

        if self._stream is not None:
            return

        try:
            self._stream = self._stream_factory(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._on_audio,
                **self._stream_kwargs,
            )
            self._stream.start()
        except CaptureError:
            self._stream = None
            raise
        except Exception as exc:  # pragma: no cover - concrete message tested via callers
            self._stream = None
            raise CaptureError(f"failed to start {self.source} capture: {exc}") from exc

    def stop(self) -> None:
        """Stop and close the underlying audio stream."""

        stream = self._stream
        self._stream = None
        if stream is None:
            return

        stop = getattr(stream, "stop", None)
        close = getattr(stream, "close", None)
        if callable(stop):
            stop()
        if callable(close):
            close()

    def read_frame(self, timeout: float | None = None) -> AudioFrame | None:
        """Read the next captured frame, or None on timeout."""

        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def drain(self) -> list[AudioFrame]:
        """Return all currently queued frames."""

        frames: list[AudioFrame] = []
        while True:
            try:
                frames.append(self._queue.get_nowait())
            except Empty:
                return frames

    def __enter__(self) -> QueuedAudioStream:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def _on_audio(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        samples = normalize_samples(indata, self.channels)
        timestamp = _timestamp_from_time_info(time_info, self._clock)
        frame = AudioFrame(
            source=self.source,
            samples=samples,
            sample_rate=self.sample_rate,
            channels=self.channels,
            timestamp_seconds=timestamp,
            status=str(status) if status else "",
        )

        self.frames_received += int(frames)
        self._put_latest(frame)

    def _put_latest(self, frame: AudioFrame) -> None:
        self.frames_dropped += put_latest_frame(self._queue, frame)


def _timestamp_from_time_info(time_info: Any, clock: Clock) -> float:
    timestamp = getattr(time_info, "inputBufferAdcTime", None)
    if timestamp is not None:
        try:
            return float(timestamp)
        except (TypeError, ValueError):
            pass
    return float(clock())


def put_latest_frame(frame_queue: Queue[AudioFrame], frame: AudioFrame) -> int:
    """Masukkan frame terbaru; jika queue penuh, buang frame terlama.

    Return value adalah jumlah frame audio yang dianggap dropped. Helper ini
    menjaga policy queue konsisten antara mic, speaker loopback, dan stream lain:
    sistem lebih memilih audio terbaru daripada menahan data lama.
    """

    try:
        frame_queue.put_nowait(frame)
        return 0
    except Full:
        dropped = frame.frame_count

    try:
        frame_queue.get_nowait()
    except Empty:
        pass

    try:
        frame_queue.put_nowait(frame)
    except Full:
        dropped += frame.frame_count
    return dropped
