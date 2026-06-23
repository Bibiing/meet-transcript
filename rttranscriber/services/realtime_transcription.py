from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter, sleep

from rttranscriber.model.audio import AudioChunk
from rttranscriber.model.transcription import (
    RealtimeSessionResult,
    TranscriptSnapshot,
    TranscriptToken,
    TranscriptionChunkResult,
)
from rttranscriber.services.audio_pipeline import AudioFrameNormalizer, SlidingWindowChunkScheduler
from rttranscriber.services.capture_sources import AudioSource
from rttranscriber.services.transcription_engine import TranscriptEngine
from rttranscriber.wav_chunk_writer import WavChunkSink


class AsyncTranscriptionWorker:
    """Worker inference async agar callback/capture tidak tertahan engine transcript."""

    def __init__(self, engine: TranscriptEngine) -> None:
        self._engine = engine
        self._input_queue: Queue[AudioChunk | None] = Queue()
        self._output_queue: Queue[TranscriptionChunkResult] = Queue()
        self._pending_count = 0
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._closed = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, chunk: AudioChunk) -> None:
        if self._closed:
            raise RuntimeError("worker sudah ditutup")
        self._pending_count += 1
        self._input_queue.put(chunk)

    def drain_ready(self) -> list[TranscriptionChunkResult]:
        ready: list[TranscriptionChunkResult] = []
        while True:
            try:
                ready.append(self._output_queue.get_nowait())
            except Empty:
                return ready

    def close_input(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._input_queue.put(None)

    def is_idle(self) -> bool:
        return self._pending_count == 0

    def stop(self) -> None:
        self.close_input()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            chunk = self._input_queue.get()
            if chunk is None:
                break
            result = self._engine.transcribe(chunk)
            self._pending_count -= 1
            self._output_queue.put(result)


class TranscriptAssembler:
    """Menggabungkan hasil overlap menjadi final dan partial transcript."""

    def __init__(self, hop_samples: int) -> None:
        self._hop_samples = hop_samples
        self._processed_chunk_count = 0
        self._committed_tokens: list[TranscriptToken] = []
        self._pending_tokens: dict[int, TranscriptToken] = {}
        self._last_partial_text = ""

    def apply(self, result: TranscriptionChunkResult) -> TranscriptSnapshot:
        self._processed_chunk_count += 1
        for token in result.tokens:
            self._pending_tokens[token.start_frame_index] = token

        commit_threshold = result.chunk_start_frame_index + self._hop_samples
        committable_keys = sorted(
            key for key, token in self._pending_tokens.items() if token.end_frame_index <= commit_threshold
        )
        for key in committable_keys:
            self._committed_tokens.append(self._pending_tokens.pop(key))

        self._last_partial_text = result.partial_text
        return self.snapshot()

    def finalize(self) -> TranscriptSnapshot:
        for key in sorted(self._pending_tokens):
            self._committed_tokens.append(self._pending_tokens[key])
        self._pending_tokens.clear()
        self._last_partial_text = ""
        return self.snapshot()

    def snapshot(self) -> TranscriptSnapshot:
        committed_tokens = sorted(self._committed_tokens, key=lambda token: token.start_frame_index)
        pending_tokens = sorted(self._pending_tokens.values(), key=lambda token: token.start_frame_index)
        final_text = " ".join(token.text for token in committed_tokens)
        pending_text = " ".join(token.text for token in pending_tokens)
        partial_parts = [part for part in (final_text, pending_text or self._last_partial_text) if part]
        return TranscriptSnapshot(
            final_text=final_text,
            partial_text=" ".join(partial_parts).strip(),
            committed_tokens=committed_tokens,
            pending_tokens=pending_tokens,
            processed_chunk_count=self._processed_chunk_count,
        )


@dataclass(slots=True)
class RealtimeTranscriptionConfig:
    capture_seconds: int = 8
    window_seconds: int = 6
    hop_seconds: int = 2
    buffer_seconds: int = 30
    write_debug_wav: bool = True


class RealtimeTranscriptionCoordinator:
    """Use case utama Phase 3 yang menjahit source, worker, merger, dan sink."""

    def __init__(
        self,
        audio_source: AudioSource,
        transcript_engine: TranscriptEngine,
        output_directory: Path,
    ) -> None:
        self._audio_source = audio_source
        self._normalizer = AudioFrameNormalizer()
        self._transcript_engine = transcript_engine
        self._output_directory = output_directory

    def run(self, config: RealtimeTranscriptionConfig) -> RealtimeSessionResult:
        scheduler = SlidingWindowChunkScheduler(
            window_seconds=config.window_seconds,
            hop_seconds=config.hop_seconds,
            buffer_seconds=config.buffer_seconds,
        )
        worker = AsyncTranscriptionWorker(self._transcript_engine)
        assembler = TranscriptAssembler(hop_samples=scheduler.hop_samples)
        wav_sink = WavChunkSink(self._output_directory) if config.write_debug_wav else None

        diagnostics = self._audio_source.diagnostics()
        created_files: list[Path] = []
        snapshots: list[TranscriptSnapshot] = []

        self._audio_source.start()
        worker.start()
        start = perf_counter()

        try:
            while perf_counter() - start < config.capture_seconds:
                frame = self._audio_source.read_frame(timeout=0.25)
                if frame is not None:
                    normalized = self._normalizer.normalize(frame)
                    for chunk in scheduler.push(normalized):
                        if wav_sink is not None:
                            created_files.append(wav_sink.publish(chunk))
                        worker.submit(chunk)

                snapshots.extend(self._consume_ready_results(worker, assembler))
                sleep(0.01)
        finally:
            self._audio_source.stop()

        worker.close_input()
        while not worker.is_idle():
            snapshots.extend(self._consume_ready_results(worker, assembler))
            sleep(0.01)
        snapshots.extend(self._consume_ready_results(worker, assembler))
        worker.stop()

        final_snapshot = assembler.finalize()
        snapshots.append(final_snapshot)
        return RealtimeSessionResult(
            diagnostics=diagnostics,
            created_files=created_files,
            snapshots=snapshots,
            final_snapshot=final_snapshot,
        )

    def _consume_ready_results(
        self,
        worker: AsyncTranscriptionWorker,
        assembler: TranscriptAssembler,
    ) -> list[TranscriptSnapshot]:
        snapshots: list[TranscriptSnapshot] = []
        for result in worker.drain_ready():
            snapshots.append(assembler.apply(result))
        return snapshots
