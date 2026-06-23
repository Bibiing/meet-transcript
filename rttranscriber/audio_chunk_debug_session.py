from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .services.audio_pipeline import AudioFrameNormalizer, TimestampedPcmRingBuffer
from .pysoundio_microphone_source import PySoundIoCapture
from .wav_chunk_writer import WavChunkSink


@dataclass(slots=True)
class AudioChunkDebugConfig:
    capture_seconds: int = 7
    window_seconds: int = 6
    overlap_seconds: int = 4


class AudioChunkDebugRunner:
    def __init__(self, config: AudioChunkDebugConfig, output_directory: Path) -> None:
        self._config = config
        self._normalizer = AudioFrameNormalizer()
        self._buffer = TimestampedPcmRingBuffer(16000 * 30)
        self._capture = PySoundIoCapture()
        self._wav_sink = WavChunkSink(output_directory)

    def run(self) -> tuple[str, list[Path]]:
        diagnostics = self._capture.diagnostics()
        self._capture.start()
        created_files: list[Path] = []
        start = perf_counter()
        window_samples = self._config.window_seconds * 16000
        overlap_samples = self._config.overlap_seconds * 16000

        try:
            while perf_counter() - start < self._config.capture_seconds:
                frame = self._capture.read_frame(timeout=0.5)
                if frame is None:
                    continue
                normalized = self._normalizer.normalize(frame)
                self._buffer.push(normalized)
                chunk = self._buffer.build_chunk(window_samples)
                if chunk is None:
                    continue
                created_files.append(self._wav_sink.publish(chunk))
                self._buffer.consume_until(chunk.end_frame_index - overlap_samples)
                break
        finally:
            self._capture.stop()

        return diagnostics, created_files
