from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import time

from .audio_models import AudioFormat, AudioFrame
from .pysoundio_runtime import ffi, lib


PREFERRED_SAMPLE_RATES = (48000, 44100, 32000, 24000, 16000)


class PySoundIoError(RuntimeError):
    pass


def _check(code: int, message: str) -> None:
    if code != lib.SoundIoErrorNone:
        raise PySoundIoError(f"{message}: {ffi.string(lib.soundio_strerror(code)).decode()}")


@dataclass(slots=True)
class DeviceSummary:
    index: int
    name: str
    is_default: bool
    is_raw: bool


class PySoundIoCapture:
    """Wrapper runtime yang mengikuti pola PySoundIo untuk Phase 2."""

    def __init__(self) -> None:
        self._soundio = lib.soundio_create()
        if self._soundio == ffi.NULL:
            raise PySoundIoError("gagal membuat context soundio")
        _check(lib.soundio_connect(self._soundio), "gagal connect backend")
        lib.soundio_flush_events(self._soundio)

        self._device = ffi.NULL
        self._instream = ffi.NULL
        self._ring_buffer = ffi.NULL
        self._read_callback_handle = None
        self._overflow_callback_handle = None
        self._error_callback_handle = None
        self._event_thread: Thread | None = None
        self._drain_thread: Thread | None = None
        self._stop_event = Event()
        self._frames: Queue[AudioFrame] = Queue()
        self._frame_index = 0
        self._start_time = time.perf_counter()
        self._source_format = AudioFormat(sample_rate=48000, channels=2, bits_per_sample=16)

    def diagnostics(self) -> str:
        backend_name = "WASAPI" if self._soundio.current_backend == lib.SoundIoBackendWasapi else str(
            self._soundio.current_backend
        )
        input_count = lib.soundio_input_device_count(self._soundio)
        default_input = lib.soundio_default_input_device_index(self._soundio)
        lines = [
            f"backend={backend_name}",
            f"default_input_index={default_input}",
            f"input_devices={input_count}",
        ]
        for index in range(input_count):
            device = lib.soundio_get_input_device(self._soundio, index)
            if device == ffi.NULL:
                continue
            name = ffi.string(device.name).decode() if device.name != ffi.NULL else "<unknown>"
            lines.append(
                f"  - [{index}] {name}"
                f"{' [default]' if index == default_input else ''}"
                f"{' [raw]' if device.is_raw else ''}"
            )
            lib.soundio_device_unref(device)
        return "\n".join(lines)

    def list_input_devices(self) -> list[DeviceSummary]:
        result: list[DeviceSummary] = []
        input_count = lib.soundio_input_device_count(self._soundio)
        default_input = lib.soundio_default_input_device_index(self._soundio)
        for index in range(input_count):
            device = lib.soundio_get_input_device(self._soundio, index)
            if device == ffi.NULL:
                continue
            result.append(
                DeviceSummary(
                    index=index,
                    name=ffi.string(device.name).decode() if device.name != ffi.NULL else "<unknown>",
                    is_default=index == default_input,
                    is_raw=bool(device.is_raw),
                )
            )
            lib.soundio_device_unref(device)
        return result

    def start(self) -> None:
        default_index = lib.soundio_default_input_device_index(self._soundio)
        if default_index < 0:
            raise PySoundIoError("tidak ada input device default")

        self._device = lib.soundio_get_input_device(self._soundio, default_index)
        if self._device == ffi.NULL:
            raise PySoundIoError("gagal mengambil device input default")
        if self._device.probe_error != 0:
            raise PySoundIoError(f"device probe error: {ffi.string(lib.soundio_strerror(self._device.probe_error)).decode()}")

        lib.soundio_device_sort_channel_layouts(self._device)
        self._instream = lib.soundio_instream_create(self._device)
        if self._instream == ffi.NULL:
            raise PySoundIoError("gagal membuat input stream")

        selected_format = lib.SoundIoFormatS16LE
        if not lib.soundio_device_supports_format(self._device, selected_format):
            raise PySoundIoError("format S16LE tidak didukung device")

        sample_rate = next(
            (rate for rate in PREFERRED_SAMPLE_RATES if lib.soundio_device_supports_sample_rate(self._device, rate)),
            lib.soundio_device_nearest_sample_rate(self._device, 48000),
        )
        if sample_rate <= 0:
            sample_rate = 48000

        layout = lib.soundio_channel_layout_get_default(2)
        if layout == ffi.NULL:
            raise PySoundIoError("layout stereo default tidak tersedia")

        self._instream.format = selected_format
        self._instream.sample_rate = sample_rate
        self._instream.layout = layout[0]
        self._instream.software_latency = 0.1
        self._instream.userdata = ffi.new_handle(self)

        @ffi.callback("void(struct SoundIoInStream *, int, int)")
        def _read_callback(instream, frame_count_min, frame_count_max):
            self._on_read(instream, frame_count_min, frame_count_max)

        @ffi.callback("void(struct SoundIoInStream *)")
        def _overflow_callback(instream):
            return None

        @ffi.callback("void(struct SoundIoInStream *, int)")
        def _error_callback(instream, err):
            return None

        self._read_callback_handle = _read_callback
        self._overflow_callback_handle = _overflow_callback
        self._error_callback_handle = _error_callback
        self._instream.read_callback = _read_callback
        self._instream.overflow_callback = _overflow_callback
        self._instream.error_callback = _error_callback

        _check(lib.soundio_instream_open(self._instream), "gagal open input stream")

        self._source_format = AudioFormat(
            sample_rate=int(self._instream.sample_rate),
            channels=int(self._instream.layout.channel_count),
            bits_per_sample=int(self._instream.bytes_per_sample * 8),
        )
        capacity = int(8 * self._instream.sample_rate * self._instream.bytes_per_frame)
        self._ring_buffer = lib.soundio_ring_buffer_create(self._soundio, capacity)
        if self._ring_buffer == ffi.NULL:
            raise PySoundIoError("gagal membuat ring buffer")

        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._frame_index = 0
        _check(lib.soundio_instream_start(self._instream), "gagal start input stream")
        self._event_thread = Thread(target=self._run_event_loop, daemon=True)
        self._drain_thread = Thread(target=self._run_drain_loop, daemon=True)
        self._event_thread.start()
        self._drain_thread.start()

    def read_frame(self, timeout: float = 1.0) -> AudioFrame | None:
        try:
            return self._frames.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._soundio != ffi.NULL:
            lib.soundio_wakeup(self._soundio)
        if self._event_thread is not None:
            self._event_thread.join(timeout=2.0)
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=2.0)
        if self._ring_buffer != ffi.NULL:
            lib.soundio_ring_buffer_destroy(self._ring_buffer)
            self._ring_buffer = ffi.NULL
        if self._instream != ffi.NULL:
            lib.soundio_instream_destroy(self._instream)
            self._instream = ffi.NULL
        if self._device != ffi.NULL:
            lib.soundio_device_unref(self._device)
            self._device = ffi.NULL
        if self._soundio != ffi.NULL:
            lib.soundio_disconnect(self._soundio)
            lib.soundio_destroy(self._soundio)
            self._soundio = ffi.NULL

    def _run_event_loop(self) -> None:
        while not self._stop_event.is_set():
            lib.soundio_wait_events(self._soundio)

    def _run_drain_loop(self) -> None:
        while not self._stop_event.is_set():
            self._drain_once()
            time.sleep(0.02)
        self._drain_once()

    def _on_read(self, instream, frame_count_min: int, frame_count_max: int) -> None:
        write_ptr = lib.soundio_ring_buffer_write_ptr(self._ring_buffer)
        free_bytes = lib.soundio_ring_buffer_free_count(self._ring_buffer)
        free_frames = int(free_bytes / instream.bytes_per_frame)
        if free_frames < frame_count_min:
            return

        write_frames = min(free_frames, frame_count_max)
        frames_left = write_frames
        cursor = write_ptr

        # Callback hanya memindahkan PCM mentah ke ring buffer agar aman untuk
        # thread audio realtime.
        while frames_left > 0:
            frame_count_ptr = ffi.new("int *", frames_left)
            areas_ptr = ffi.new("struct SoundIoChannelArea **")
            _check(lib.soundio_instream_begin_read(instream, areas_ptr, frame_count_ptr), "begin_read gagal")
            frame_count = int(frame_count_ptr[0])
            if frame_count == 0:
                break

            if areas_ptr[0] == ffi.NULL:
                ffi.memmove(cursor, b"\x00" * (frame_count * instream.bytes_per_frame), frame_count * instream.bytes_per_frame)
                cursor += frame_count * instream.bytes_per_frame
            else:
                for frame in range(frame_count):
                    for channel in range(instream.layout.channel_count):
                        area = areas_ptr[0][channel]
                        src_ptr = area.ptr + (area.step * frame)
                        ffi.memmove(cursor, src_ptr, instream.bytes_per_sample)
                        cursor += instream.bytes_per_sample

            _check(lib.soundio_instream_end_read(instream), "end_read gagal")
            frames_left -= frame_count

        lib.soundio_ring_buffer_advance_write_ptr(self._ring_buffer, write_frames * instream.bytes_per_frame)

    def _drain_once(self) -> None:
        fill_bytes = lib.soundio_ring_buffer_fill_count(self._ring_buffer)
        if fill_bytes <= 0:
            return

        bytes_per_frame = self._source_format.channels * (self._source_format.bits_per_sample // 8)
        frame_count = fill_bytes // bytes_per_frame
        read_ptr = lib.soundio_ring_buffer_read_ptr(self._ring_buffer)
        data = ffi.unpack(ffi.cast("int16_t *", read_ptr), frame_count * self._source_format.channels)
        samples = [int(value) for value in data]
        lib.soundio_ring_buffer_advance_read_ptr(self._ring_buffer, frame_count * bytes_per_frame)

        timestamp_seconds = time.perf_counter() - self._start_time
        frame = AudioFrame(
            timestamp_seconds=timestamp_seconds,
            frame_index=self._frame_index,
            audio_format=self._source_format,
            samples=samples,
        )
        self._frame_index += frame_count
        self._frames.put(frame)
