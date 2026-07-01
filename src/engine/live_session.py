"""Real-time live transcription session — concurrent streaming pipeline.

Architecture
------------
Two capture threads (mic + speaker) accumulate raw AudioFrame objects
until *chunk_seconds* worth of audio have been collected, then push the
preprocessed chunks into a shared queue.  A single transcription loop on
the main thread drains that queue, calls Whisper, and prints results as
they arrive.

Using a single transcription thread is intentional: Whisper is CPU-bound
and not thread-safe, so serialising calls avoids race conditions and gives
predictable memory usage.  The capture threads are non-blocking and will
drop chunks if the transcription queue overflows (prefer fresh audio over
stale buffered audio in a live-meeting scenario).
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, perf_counter
from typing import Callable, Literal

import numpy as np

from src.capture.audio_frame import AudioFrame
from src.capture.errors import CaptureError, CaptureNotSupportedError
from src.capture.mic_stream import MicrophoneConfig, MicrophoneStream
from src.capture.win_loopback import WindowsLoopbackConfig, WindowsLoopbackStream
from src.engine.preprocessing import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk
from src.engine.whisper import OpenAIWhisperTranscriber, TranscriptionResult, WhisperConfig
from src.utils.formatter import format_transcript_line, result_to_line
from src.utils.os_detector import OperatingSystem, detect_os
from src.utils.transcript_log import TranscriptLog

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LiveSessionConfig:
    """Runtime configuration for a live transcription session.

    chunk_seconds
        How many seconds of audio to accumulate before sending to Whisper.
        Lower = less latency, more risk of cut-off words.
        Higher = more latency, better context for Whisper.
        Recommended: 4.0 for CPU, 2.0 for GPU.

    source
        Which audio sources to capture: "mic", "speaker", or "both".
        For online meetings, "both" captures your voice (mic) and
        remote participants (speaker / WASAPI loopback).

    max_chunk_queue_size
        How many unprocessed chunks to buffer.  When full, the oldest
        chunk is dropped to prefer fresh audio.
    """

    chunk_seconds: float = 4.0
    source: Literal["mic", "speaker", "both"] = "both"
    sample_rate: int | None = None
    block_size: int = 1_024
    queue_size: int = 128
    max_chunk_queue_size: int = 16
    transcript_log_path: Path | None = None
    whisper_config: WhisperConfig | None = None


@dataclass
class LiveSessionStats:
    """Runtime statistics collected during a live session."""

    chunks_processed: int = 0
    chunks_dropped: int = 0
    results_accepted: int = 0
    results_rejected: int = 0
    start_time: float = field(default_factory=perf_counter)

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.start_time


def run_live_session(
    config: LiveSessionConfig | None = None,
    *,
    on_result: Callable[[TranscriptionResult], None] | None = None,
) -> LiveSessionStats:
    """Run a real-time transcription session until Ctrl+C or KeyboardInterrupt.

    Parameters
    ----------
    config:
        Session configuration.  Uses safe defaults when omitted.
    on_result:
        Optional callback called for every accepted transcription result.
        Useful for integrations (e.g., write to file, send to UI).

    Returns
    -------
    LiveSessionStats
        Runtime statistics for the completed session.
    """
    cfg = config or LiveSessionConfig()
    whisper_cfg = cfg.whisper_config or WhisperConfig()
    stats = LiveSessionStats()

    # ------------------------------------------------------------------ #
    # Transcript log (crash-resistant real-time backup)                   #
    # ------------------------------------------------------------------ #
    transcript_log: TranscriptLog | None = None
    if cfg.transcript_log_path:
        transcript_log = TranscriptLog(cfg.transcript_log_path)
        transcript_log.save()

    # ------------------------------------------------------------------ #
    # Shared state                                                        #
    # ------------------------------------------------------------------ #
    stop_event = threading.Event()
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(
        maxsize=cfg.max_chunk_queue_size
    )

    # ------------------------------------------------------------------ #
    # Capture streams & reader threads                                    #
    # ------------------------------------------------------------------ #
    capture_streams: list = []
    reader_threads: list[threading.Thread] = []
    preprocess_config = PreprocessConfig(chunk_seconds=cfg.chunk_seconds)

    if cfg.source in ("mic", "both"):
        try:
            mic_stream = MicrophoneStream(
                MicrophoneConfig(
                    sample_rate=cfg.sample_rate,
                    block_size=cfg.block_size,
                    queue_size=cfg.queue_size,
                )
            )
            capture_streams.append(mic_stream)
            reader_threads.append(
                threading.Thread(
                    target=_reader_thread,
                    args=(
                        mic_stream,
                        AudioPreprocessor(preprocess_config),
                        chunk_queue,
                        cfg.chunk_seconds,
                        stop_event,
                        stats,
                    ),
                    name="live-mic-reader",
                    daemon=True,
                )
            )
            _log.info("live session: mic stream created")
        except Exception as exc:  # pragma: no cover
            _log.warning("live session: mic stream unavailable — %s", exc)

    if cfg.source in ("speaker", "both"):
        os_ = detect_os()
        if os_ is OperatingSystem.WINDOWS:
            try:
                spk_stream = WindowsLoopbackStream(
                    WindowsLoopbackConfig(
                        sample_rate=cfg.sample_rate,
                        block_size=cfg.block_size,
                        queue_size=cfg.queue_size,
                    )
                )
                capture_streams.append(spk_stream)
                reader_threads.append(
                    threading.Thread(
                        target=_reader_thread,
                        args=(
                            spk_stream,
                            AudioPreprocessor(preprocess_config),
                            chunk_queue,
                            cfg.chunk_seconds,
                            stop_event,
                            stats,
                        ),
                        name="live-speaker-reader",
                        daemon=True,
                    )
                )
                _log.info("live session: speaker loopback stream created")
            except CaptureNotSupportedError as exc:
                _log.warning(
                    "live session: speaker loopback unavailable — %s\n"
                    "  Tip: pastikan 'Stereo Mix' atau loopback device aktif di Sound Settings.",
                    exc,
                )
        else:
            _log.warning(
                "live session: speaker capture not supported on %s — mic only",
                os_.value,
            )

    if not capture_streams:
        _log.error("live session: no capture streams available — aborting")
        print("[ERROR] Tidak ada audio stream yang tersedia. Periksa perangkat audio Anda.")
        return stats

    # ------------------------------------------------------------------ #
    # Graceful Ctrl+C handler                                             #
    # ------------------------------------------------------------------ #
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigbreak = signal.getsignal(signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else None

    def _sigint_handler(signum: int, frame: object) -> None:
        _log.info("live session: stop signal received signal=%s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sigint_handler)

    # ------------------------------------------------------------------ #
    # Start streams and reader threads                                    #
    # ------------------------------------------------------------------ #
    active_streams: list = []
    for stream in capture_streams:
        try:
            stream.start()
            active_streams.append(stream)
            _log.info("live session: started stream %s", getattr(stream, "device", stream))
        except CaptureError as exc:
            _log.warning("live session: failed to start stream: %s", exc)

    if not active_streams:
        print("[ERROR] Gagal memulai audio stream. Periksa izin mikrofon / audio device.")
        signal.signal(signal.SIGINT, original_sigint)
        return stats

    for thread in reader_threads:
        thread.start()

    # ------------------------------------------------------------------ #
    # Transcription loop (main thread — serialised, thread-safe)         #
    # ------------------------------------------------------------------ #
    transcriber = OpenAIWhisperTranscriber(whisper_cfg)

    source_labels = {"mic": "MIC", "speaker": "SPEAKER"}
    session_start_wall = monotonic()

    print("=" * 60)
    print("Live transcription aktif. Tekan Ctrl+C untuk berhenti.")
    source_list = ", ".join(
        source_labels.get(s, s.upper())
        for s in (["mic", "speaker"] if cfg.source == "both" else [cfg.source])
    )
    print(f"Sumber: {source_list}  |  Chunk: {cfg.chunk_seconds:.0f}s  |  Model: {whisper_cfg.model_name}")
    print("=" * 60)

    try:
        while not stop_event.is_set():
            try:
                chunk = chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            _log.debug(
                "live session: transcribing chunk source=%s duration=%.2fs",
                chunk.source,
                chunk.duration_seconds,
            )

            result = transcriber.transcribe_chunk(chunk)
            stats.chunks_processed += 1

            if result.text:
                stats.results_accepted += 1
                line = format_transcript_line(result_to_line(result))
                print(line, flush=True)
                if transcript_log is not None:
                    transcript_log.append_result(result)
                if on_result is not None:
                    on_result(result)
                _log.info("live session: accepted source=%s text=%r", result.source, result.text[:80])
            else:
                stats.results_rejected += 1
                _log.debug(
                    "live session: rejected source=%s warning=%s",
                    result.source,
                    result.warning,
                )

    finally:
        # ---------------------------------------------------------------- #
        # Shutdown — stop streams, join threads, restore signal handler    #
        # ---------------------------------------------------------------- #
        stop_event.set()
        for stream in active_streams:
            try:
                stream.stop()
            except Exception as exc:  # pragma: no cover
                _log.debug("live session: error stopping stream: %s", exc)
        for thread in reader_threads:
            thread.join(timeout=3.0)
        signal.signal(signal.SIGINT, original_sigint)
        if hasattr(signal, "SIGBREAK") and original_sigbreak is not None:
            signal.signal(signal.SIGBREAK, original_sigbreak)

        elapsed = monotonic() - session_start_wall
        print("\n" + "=" * 60)
        print(f"Sesi selesai. Durasi: {elapsed:.0f}s")
        print(
            f"Chunks diproses: {stats.chunks_processed}  |  "
            f"Diterima: {stats.results_accepted}  |  "
            f"Ditolak: {stats.results_rejected}  |  "
            f"Dropped: {stats.chunks_dropped}"
        )
        if transcript_log is not None:
            print(f"Transcript log: {cfg.transcript_log_path}")
        print("=" * 60)

    return stats


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _reader_thread(
    stream: object,
    preprocessor: AudioPreprocessor,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    chunk_seconds: float,
    stop_event: threading.Event,
    stats: LiveSessionStats,
) -> None:
    """Accumulate frames from *stream*, preprocess, push to *chunk_queue*.

    This function runs in its own daemon thread.  It intentionally does not
    reference the Whisper model — only audio capture and preprocessing happen
    here.  Thread-safety is maintained by:
      - Only writing to *chunk_queue* (thread-safe Queue).
      - Only reading *stop_event.is_set()* (atomic bool-like).
      - *preprocessor* is NOT shared between threads.
    """
    buffer: list[AudioFrame] = []
    source_name = "unknown"
    frames_seen = 0
    chunks_enqueued = 0
    first_frame_logged = False
    dropped_by_preprocess = 0

    while not stop_event.is_set():
        frame = stream.read_frame(timeout=0.1)  # type: ignore[attr-defined]
        if frame is None:
            continue

        source_name = frame.source
        frames_seen += 1
        if not first_frame_logged:
            _log.info(
                "live session: first frame source=%s sample_rate=%s channels=%s frames=%s rms=%.6f status=%s",
                frame.source,
                frame.sample_rate,
                frame.channels,
                frame.frame_count,
                _frame_rms(frame),
                frame.status or "",
            )
            first_frame_logged = True
        buffer.append(frame)

        # Check accumulated duration
        total_duration = sum(f.duration_seconds for f in buffer)
        if total_duration < chunk_seconds:
            continue

        input_rms = _buffer_rms(buffer)
        input_frames = sum(frame.frame_count for frame in buffer)
        # Preprocess the accumulated buffer
        chunks = preprocessor.preprocess_frames(buffer)
        buffer = []
        if not chunks:
            dropped_by_preprocess += 1
            _log.info(
                "live session: preprocess produced no chunks source=%s duration=%.3f input_frames=%s input_rms=%.6f dropped_windows=%s",
                source_name,
                total_duration,
                input_frames,
                input_rms,
                dropped_by_preprocess,
            )
            continue

        for chunk in chunks:
            try:
                chunk_queue.put_nowait(chunk)
                chunks_enqueued += 1
                _log.debug(
                    "live session: queued chunk source=%s duration=%.3f input_rms_db=%.2f output_rms_db=%.2f queued=%s",
                    chunk.source,
                    chunk.duration_seconds,
                    chunk.input_rms_db,
                    chunk.rms_db,
                    chunks_enqueued,
                )
            except queue.Full:
                # Drop the oldest item and insert the new one (prefer fresh audio)
                try:
                    chunk_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    chunk_queue.put_nowait(chunk)
                    chunks_enqueued += 1
                    stats.chunks_dropped += 1
                    _log.warning(
                        "live session: chunk queue full — dropped oldest %s chunk", source_name
                    )
                except queue.Full:
                    pass

    if buffer:
        partial_duration = sum(f.duration_seconds for f in buffer)
        partial_frames = sum(frame.frame_count for frame in buffer)
        partial_rms = _buffer_rms(buffer)
        _log.info(
            "live session: reader exiting with partial buffer source=%s duration=%.3f frames=%s rms=%.6f",
            source_name,
            partial_duration,
            partial_frames,
            partial_rms,
        )
        if partial_duration >= preprocessor.config.min_chunk_seconds:
            for chunk in preprocessor.preprocess_frames(buffer):
                try:
                    chunk_queue.put_nowait(chunk)
                    chunks_enqueued += 1
                    _log.info(
                        "live session: queued final partial chunk source=%s duration=%.3f input_rms_db=%.2f output_rms_db=%.2f",
                        chunk.source,
                        chunk.duration_seconds,
                        chunk.input_rms_db,
                        chunk.rms_db,
                    )
                except queue.Full:
                    stats.chunks_dropped += 1
                    _log.warning("live session: final partial chunk dropped because queue is full source=%s", source_name)
    _log.debug(
        "live session: reader thread exiting (source=%s frames_seen=%s chunks_enqueued=%s dropped_windows=%s)",
        source_name,
        frames_seen,
        chunks_enqueued,
        dropped_by_preprocess,
    )


def _frame_rms(frame: AudioFrame) -> float:
    if frame.samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.samples.astype(np.float32)))))


def _buffer_rms(frames: list[AudioFrame]) -> float:
    samples = [frame.samples.astype(np.float32).reshape(-1) for frame in frames if frame.samples.size]
    if not samples:
        return 0.0
    joined = np.concatenate(samples)
    if joined.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(joined))))
