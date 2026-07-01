"""Live dual-stream session that sends audio to a WhisperLive server."""

from __future__ import annotations

import logging
import queue
import signal
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic, perf_counter
from typing import Callable, Literal

from src.capture.audio_frame import AudioFrame
from src.capture.errors import CaptureError, CaptureNotSupportedError
from src.capture.wav_sink import write_frames_to_wav
from src.capture.mac_sys_audio import MacSystemAudioConfig, MacSystemAudioStream
from src.capture.mic_stream import MicrophoneConfig, MicrophoneStream
from src.capture.win_loopback import WindowsLoopbackConfig, WindowsLoopbackStream
from src.engine.live_session import _reader_thread
from src.engine.preprocessing import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk
from src.engine.transcript_merger import MEETING_SOURCE_LABELS, TranscriptMerger
from src.engine.whisper import TranscriptionResult, TranscriptionSegment
from src.engine.whisperlive_client import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
    WhisperLiveStreamClient,
)
from src.utils.os_detector import OperatingSystem, detect_os
from src.utils.transcript_log import TranscriptLog

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WhisperLiveSessionConfig:
    """Runtime config for WhisperLive-backed live transcription."""

    server_host: str = "localhost"
    server_port: int = 9090
    use_wss: bool = False
    api_key: str | None = None
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    chunk_seconds: float = 0.5
    audio_format: Literal["float32", "int16", "uint8"] = "int16"
    source: Literal["mic", "speaker", "both"] = "both"
    sample_rate: int | None = None
    block_size: int = 1_024
    queue_size: int = 128
    max_chunk_queue_size: int = 32
    final_drain_seconds: float = 10.0
    show_partials: bool = True
    transcript_log_path: Path | None = None
    chunk_archive_dir: Path | None = None
    profile: WhisperLiveProfile = field(default_factory=WhisperLiveProfile)


@dataclass
class WhisperLiveSessionStats:
    chunks_sent: int = 0
    chunks_dropped: int = 0
    results_received: int = 0
    start_time: float = field(default_factory=perf_counter)

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.start_time


TranscriptCallback = Callable[[TranscriptionResult], None]


def run_whisperlive_session(
    config: WhisperLiveSessionConfig | None = None,
    *,
    on_result: TranscriptCallback | None = None,
) -> WhisperLiveSessionStats:
    """Capture mic/speaker audio and stream each source to WhisperLive."""

    cfg = config or WhisperLiveSessionConfig()
    stats = WhisperLiveSessionStats()
    stop_event = threading.Event()
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(maxsize=cfg.max_chunk_queue_size)
    reader_threads: list[threading.Thread] = []
    capture_streams: list[object] = []

    transcript_log: TranscriptLog | None = None
    if cfg.transcript_log_path:
        transcript_log = TranscriptLog(cfg.transcript_log_path)
        transcript_log.save()
    transcript_merger = TranscriptMerger()
    transcript_lock = threading.Lock()
    partial_preview = _PartialTranscriptPreview() if cfg.show_partials else None
    chunk_archive = _ChunkArchive(cfg.chunk_archive_dir) if cfg.chunk_archive_dir is not None else None

    def handle_status(source: str, message: dict) -> None:
        status = str(message.get("status") or message.get("message") or "UNKNOWN")
        label = source.upper()
        if status == "CLIENT_CONNECTING":
            print(
                f"[{label}] connecting to {message.get('url')} "
                f"model={message.get('model')} ready_timeout={message.get('ready_timeout')}s",
                flush=True,
            )
            _log.info("whisperlive connecting source=%s details=%s", source, message)
            return
        if status == "CLIENT_SOCKET_OPEN":
            print(f"[{label}] websocket connected", flush=True)
            _log.info("whisperlive socket open source=%s details=%s", source, message)
            return
        if status == "CLIENT_OPTIONS_SENT":
            print(f"[{label}] options sent audio_format={message.get('audio_format')}", flush=True)
            _log.info("whisperlive options sent source=%s details=%s", source, message)
            return
        if status == "CLIENT_WAITING_SERVER_READY":
            print(
                f"[{label}] waiting server ready "
                f"waited={message.get('waited_seconds')}s remaining={message.get('remaining_seconds')}s",
                flush=True,
            )
            _log.info("whisperlive waiting server ready source=%s details=%s", source, message)
            return
        if status == "SERVER_READY":
            print(f"[{label}] server ready backend={message.get('backend')}", flush=True)
            _log.info("whisperlive stream ready source=%s details=%s", source, message)
            return
        if status == "CLIENT_RECV_ERROR":
            print(
                f"[{label}] receive error: {message.get('message')} "
                f"(chunks={message.get('chunks_sent')} messages={message.get('messages_received')} "
                f"segments={message.get('segments_received')})",
                flush=True,
            )
            _log.warning("whisperlive receive error source=%s details=%s", source, message)
            stop_event.set()
            return
        if status == "CLIENT_REMOTE_CLOSED":
            print(
                f"[{label}] remote closed websocket "
                f"(chunks={message.get('chunks_sent')} messages={message.get('messages_received')} "
                f"segments={message.get('segments_received')} elapsed={message.get('elapsed_seconds')}s)",
                flush=True,
            )
            _log.warning("whisperlive remote closed websocket source=%s details=%s", source, message)
            stop_event.set()
            return
        if status in {"DISCONNECT", "CLIENT_READY_TIMEOUT"}:
            print(f"[{label}] disconnected: {status} {message.get('message') or ''}".rstrip(), flush=True)
            _log.warning("whisperlive disconnected source=%s details=%s", source, message)
            stop_event.set()
            return
        if status == "CLIENT_CLOSED":
            print(
                f"[{label}] client closed "
                f"chunks={message.get('chunks_sent')} messages={message.get('messages_received')} "
                f"segments={message.get('segments_received')}",
                flush=True,
            )
            _log.info("whisperlive client closed source=%s details=%s", source, message)
            return
        if status == "CLIENT_CLOSING":
            _log.info("whisperlive client closing source=%s details=%s", source, message)
            return
        if status in {"WAIT", "ERROR"}:
            print(f"[{label}] server status {status}: {message.get('message')}", flush=True)
            _log.warning("whisperlive server status source=%s details=%s", source, message)
            if status == "ERROR":
                stop_event.set()
            return
        if "segments" in message:
            _log.info("whisperlive transcript status source=%s details=%s", source, message)
            return
        print(f"[{label}] status {status}: {message}", flush=True)
        _log.warning("whisperlive status source=%s details=%s", source, message)

    def handle_transcript(source: str, segments: list[dict], message: dict) -> None:
        _log.info(
            "whisperlive transcript received source=%s segment_count=%s completed=%s",
            source,
            len(segments),
            sum(1 for segment in segments if segment.get("completed")),
        )
        if partial_preview is not None:
            partial_preview.show(source, segments)
        for event in _events_from_segments(source, cfg.profile.model, cfg.profile.language, segments):
            with transcript_lock:
                entries = transcript_merger.add_result(event.result, completed=event.completed)
            for entry in entries:
                _emit_merged_entry(entry, transcript_log, on_result)
                stats.results_received += 1

    connection_config = WhisperLiveConnectionConfig(
        host=cfg.server_host,
        port=cfg.server_port,
        use_wss=cfg.use_wss,
        api_key=cfg.api_key,
        ready_timeout=cfg.ready_timeout,
        audio_format=cfg.audio_format,
        profile=cfg.profile,
    )

    active_sources = _requested_sources(cfg.source)
    clients = {
        source: WhisperLiveStreamClient(
            source,
            connection_config,
            on_transcript=handle_transcript,
            on_status=handle_status,
        )
        for source in active_sources
    }

    try:
        _connect_clients_parallel(clients)

        capture_streams, reader_threads = _build_capture_streams(cfg, chunk_queue, stop_event, stats)
        if not capture_streams:
            print("[ERROR] Tidak ada audio stream yang tersedia. Periksa perangkat audio Anda.")
            return stats

        active_streams: list[object] = []
        for stream in capture_streams:
            try:
                stream.start()  # type: ignore[attr-defined]
                active_streams.append(stream)
            except CaptureError as exc:
                _log.warning("whisperlive session: failed to start stream: %s", exc)

        if not active_streams:
            print("[ERROR] Gagal memulai audio stream. Periksa izin mikrofon / audio device.")
            return stats

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigbreak = signal.getsignal(signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else None

        def _sigint_handler(signum: int, frame: object) -> None:
            _log.info("whisperlive session: stop signal received signal=%s", signum)
            stop_event.set()

        signal.signal(signal.SIGINT, _sigint_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _sigint_handler)
        for thread in reader_threads:
            thread.start()

        session_start = monotonic()
        print("=" * 60)
        print("Live transcription aktif via WhisperLive. Tekan Ctrl+C untuk berhenti.")
        print(
            f"Server: {cfg.server_host}:{cfg.server_port}  |  "
            f"Model: {cfg.profile.model}  |  VAD: {cfg.profile.vad_threshold:.2f}"
        )
        print(f"Sumber: {', '.join(source.upper() for source in active_sources)}")
        print("=" * 60)

        try:
            while not stop_event.is_set():
                try:
                    chunk = chunk_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                client = clients.get(chunk.source)
                if client is None:
                    stats.chunks_dropped += 1
                    continue
                try:
                    if chunk_archive is not None:
                        chunk_archive.save(chunk)
                    client.send_chunk(chunk)
                    stats.chunks_sent += 1
                    _log.debug(
                        "whisperlive chunk sent source=%s duration=%.2f rms=%.2f",
                        chunk.source,
                        chunk.duration_seconds,
                        chunk.rms_db,
                    )
                except Exception as exc:
                    stats.chunks_dropped += 1
                    _log.error("failed to send %s chunk to WhisperLive: %s", chunk.source, exc)
        finally:
            stop_event.set()
            for stream in active_streams:
                try:
                    stream.stop()  # type: ignore[attr-defined]
                except Exception as exc:
                    _log.debug("error stopping stream: %s", exc)
            for thread in reader_threads:
                thread.join(timeout=3.0)
            signal.signal(signal.SIGINT, original_sigint)
            if hasattr(signal, "SIGBREAK") and original_sigbreak is not None:
                signal.signal(signal.SIGBREAK, original_sigbreak)

            drained = _drain_pending_chunks(chunk_queue, clients, stats, chunk_archive)
            if drained:
                print(f"Finalisasi: mengirim {drained} chunk tersisa...", flush=True)
            for client in clients.values():
                client.finish_audio()
            if cfg.final_drain_seconds > 0:
                print(f"Finalisasi: menunggu hasil server {cfg.final_drain_seconds:g}s...", flush=True)
                _wait_for_final_results(stats, cfg.final_drain_seconds)

            elapsed = monotonic() - session_start
            with transcript_lock:
                flushed_entries = transcript_merger.flush()
            for entry in flushed_entries:
                _emit_merged_entry(entry, transcript_log, on_result)
                stats.results_received += 1

            print("\n" + "=" * 60)
            print(f"Sesi selesai. Durasi: {elapsed:.0f}s")
            print(
                f"Chunks terkirim: {stats.chunks_sent}  |  "
                f"Dropped: {stats.chunks_dropped}  |  "
                f"Results: {stats.results_received}"
            )
            if transcript_log is not None:
                print(f"Transcript log: {cfg.transcript_log_path}")
            if chunk_archive is not None:
                print(f"Chunk archive: {chunk_archive.root}")
            print("=" * 60)
    finally:
        for client in clients.values():
            client.close()

    return stats


def _build_capture_streams(
    cfg: WhisperLiveSessionConfig,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    stop_event: threading.Event,
    stats: WhisperLiveSessionStats,
) -> tuple[list[object], list[threading.Thread]]:
    capture_streams: list[object] = []
    reader_threads: list[threading.Thread] = []
    mic_preprocess_config = PreprocessConfig(
        chunk_seconds=cfg.chunk_seconds,
        min_chunk_seconds=cfg.chunk_seconds,
    )
    speaker_preprocess_config = PreprocessConfig(
        chunk_seconds=cfg.chunk_seconds,
        min_chunk_seconds=cfg.chunk_seconds,
        vad_rms_threshold=0.008,
        vad_peak_threshold=0.03,
        vad_speech_fraction=0.20,
        min_input_rms_db=-48.0,
    )

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
            reader_threads.append(_reader("mic", mic_stream, mic_preprocess_config, chunk_queue, stop_event, stats))
        except Exception as exc:  # pragma: no cover
            _log.warning("whisperlive session: mic unavailable: %s", exc)

    if cfg.source in ("speaker", "both"):
        os_ = detect_os()
        try:
            if os_ is OperatingSystem.WINDOWS:
                speaker_stream = WindowsLoopbackStream(
                    WindowsLoopbackConfig(
                        sample_rate=cfg.sample_rate,
                        block_size=cfg.block_size,
                        queue_size=cfg.queue_size,
                    )
                )
            elif os_ is OperatingSystem.MACOS:
                speaker_stream = MacSystemAudioStream(
                    MacSystemAudioConfig(
                        sample_rate=cfg.sample_rate or 48_000,
                        block_size=cfg.block_size,
                        queue_size=cfg.queue_size,
                    )
                )
            else:
                raise CaptureNotSupportedError(f"speaker capture is unsupported on {os_.value}")
            capture_streams.append(speaker_stream)
            reader_threads.append(_reader("speaker", speaker_stream, speaker_preprocess_config, chunk_queue, stop_event, stats))
        except CaptureNotSupportedError as exc:
            _log.warning("whisperlive session: speaker unavailable: %s", exc)

    return capture_streams, reader_threads


def _reader(
    source: str,
    stream: object,
    preprocess_config: PreprocessConfig,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    stop_event: threading.Event,
    stats: WhisperLiveSessionStats,
) -> threading.Thread:
    return threading.Thread(
        target=_reader_thread,
        args=(
            stream,
            AudioPreprocessor(preprocess_config),
            chunk_queue,
            preprocess_config.chunk_seconds,
            stop_event,
            stats,
        ),
        name=f"whisperlive-{source}-reader",
        daemon=True,
    )


def _connect_clients_parallel(clients: dict[str, WhisperLiveStreamClient]) -> None:
    errors: list[tuple[str, BaseException]] = []
    error_lock = threading.Lock()

    def connect_one(source: str, client: WhisperLiveStreamClient) -> None:
        try:
            client.connect()
        except BaseException as exc:
            with error_lock:
                errors.append((source, exc))

    threads = [
        threading.Thread(
            target=connect_one,
            args=(source, client),
            name=f"whisperlive-{source}-connect",
            daemon=True,
        )
        for source, client in clients.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        for client in clients.values():
            client.close()
        source, exc = errors[0]
        raise RuntimeError(f"failed to connect WhisperLive stream '{source}': {exc}") from exc


def _drain_pending_chunks(
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    clients: dict[str, WhisperLiveStreamClient],
    stats: WhisperLiveSessionStats,
    chunk_archive: "_ChunkArchive | None" = None,
) -> int:
    drained = 0
    while True:
        try:
            chunk = chunk_queue.get_nowait()
        except queue.Empty:
            return drained

        client = clients.get(chunk.source)
        if client is None:
            stats.chunks_dropped += 1
            continue
        try:
            if chunk_archive is not None:
                chunk_archive.save(chunk)
            client.send_chunk(chunk)
            stats.chunks_sent += 1
            drained += 1
            _log.debug(
                "whisperlive final chunk sent source=%s duration=%.2f rms=%.2f",
                chunk.source,
                chunk.duration_seconds,
                chunk.rms_db,
            )
        except Exception as exc:
            stats.chunks_dropped += 1
            _log.error("failed to send final %s chunk to WhisperLive: %s", chunk.source, exc)


class _ChunkArchive:
    def __init__(self, root_dir: Path) -> None:
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = root_dir / session_id
        self._counters = {"mic": 0, "speaker": 0}
        self._lock = threading.Lock()

    def save(self, chunk: PreprocessedAudioChunk) -> Path:
        with self._lock:
            index = self._counters.get(chunk.source, 0) + 1
            self._counters[chunk.source] = index
        output_path = self.root / chunk.source / f"{index:06d}.wav"
        frame = AudioFrame(
            source=chunk.source,  # type: ignore[arg-type]
            samples=chunk.samples.reshape(-1, 1),
            sample_rate=chunk.sample_rate,
            channels=1,
            timestamp_seconds=chunk.start_seconds,
        )
        return write_frames_to_wav(output_path, [frame])


def _wait_for_final_results(stats: WhisperLiveSessionStats, timeout_seconds: float) -> None:
    deadline = perf_counter() + timeout_seconds
    last_results = stats.results_received
    quiet_since = perf_counter()
    while perf_counter() < deadline:
        threading.Event().wait(0.25)
        if stats.results_received != last_results:
            last_results = stats.results_received
            quiet_since = perf_counter()
        if perf_counter() - quiet_since >= 1.5:
            return


class _PartialTranscriptPreview:
    def __init__(self, *, min_interval_seconds: float = 0.75) -> None:
        self._last_text_by_source: dict[str, str] = {}
        self._last_print_by_source: dict[str, float] = {}
        self._min_interval_seconds = min_interval_seconds

    def show(self, source: str, segments: list[dict]) -> None:
        partials = [segment for segment in segments if not segment.get("completed", True)]
        if not partials:
            return

        segment = partials[-1]
        text = str(segment.get("text", "")).strip()
        if len(text) < 3:
            return

        normalized = " ".join(text.split())
        if normalized == self._last_text_by_source.get(source):
            return

        now = perf_counter()
        last_print = self._last_print_by_source.get(source, 0.0)
        if now - last_print < self._min_interval_seconds:
            return

        self._last_text_by_source[source] = normalized
        self._last_print_by_source[source] = now
        label = MEETING_SOURCE_LABELS.get(source, source.upper())
        start = _float_or_zero(segment.get("start"))
        end = _float_or_zero(segment.get("end"))
        print(f"[live {_format_timestamp(start)} - {_format_timestamp(end)}] [{label}] {normalized}", flush=True)


def _requested_sources(source: Literal["mic", "speaker", "both"]) -> list[Literal["mic", "speaker"]]:
    return ["mic", "speaker"] if source == "both" else [source]


def _results_from_segments(
    source: str,
    model_name: str,
    language: str | None,
    segments: list[dict],
) -> list[TranscriptionResult]:
    return [event.result for event in _events_from_segments(source, model_name, language, segments)]


@dataclass(frozen=True, slots=True)
class WhisperLiveTranscriptEvent:
    result: TranscriptionResult
    completed: bool


def _events_from_segments(
    source: str,
    model_name: str,
    language: str | None,
    segments: list[dict],
) -> list[WhisperLiveTranscriptEvent]:
    events: list[WhisperLiveTranscriptEvent] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = _float_or_zero(segment.get("start"))
        end = _float_or_zero(segment.get("end"))
        duration = max(0.0, end - start)
        result = (
            TranscriptionResult(
                source=source,
                text=text,
                model_name=model_name,
                language=language,
                start_seconds=start,
                duration_seconds=duration,
                segments=[
                    TranscriptionSegment(
                        start=start,
                        end=end,
                        text=text,
                    )
                ],
            )
        )
        events.append(WhisperLiveTranscriptEvent(result=result, completed=bool(segment.get("completed", True))))
    return events


def _emit_merged_entry(entry, transcript_log: TranscriptLog | None, on_result: TranscriptCallback | None) -> None:
    print(entry.display, flush=True)
    if transcript_log is not None:
        transcript_log.append_result(entry.result, label=entry.label, display=entry.display)
    if on_result is not None:
        on_result(entry.result)


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
