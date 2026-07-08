from __future__ import annotations

import logging
import queue
import signal
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic, perf_counter
from typing import Literal

from src.preprocessing.core import PreprocessedAudioChunk
from src.whisper.merger import MEETING_SOURCE_LABELS, TranscriptMerger
from src.whisper.client_base import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
)
from src.utils.logging import (
    TranscriptLog,
    clear_process_log_context,
    configure_process_logging,
    flush_process_log_summaries,
    log_process_event,
)

from src.whisper.client_reconnect import (
    ReconnectPolicy,
    _BufferedReconnectClient,
)
from src.whisper.capture import _build_capture_streams
from src.whisper.session_utils import (
    TranscriptCallback,
    LogCallback,
    MergedEntryCallback,
    _ChunkArchive,
    _PartialTranscriptPreview,
    _RollingAudioArchive,
    TranscriptQualityTracker,
    _wait_for_final_results,
    _requested_sources,
    _events_from_segments,
    _emit_merged_entry,
    _format_timestamp,
)

_log = logging.getLogger(__name__)


from src.whisper.models import WhisperLiveSessionConfig, WhisperLiveSessionStats


def _server_vad_for_source(cfg: WhisperLiveSessionConfig, source: str) -> bool:
    if source == "mic":
        return cfg.mic_server_vad
    if source == "speaker":
        return cfg.speaker_server_vad
    return False


def _connection_config_for_source(cfg: WhisperLiveSessionConfig, source: str) -> WhisperLiveConnectionConfig:
    profile = replace(cfg.profile, use_vad=_server_vad_for_source(cfg, source))
    return WhisperLiveConnectionConfig(
        host=cfg.server_host,
        port=cfg.server_port,
        use_wss=cfg.use_wss,
        sample_rate=cfg.sample_rate or 16_000,
        channels=1,
        audio_format=cfg.audio_format,
        connect_timeout=10.0,
        ready_timeout=cfg.ready_timeout,
        api_key=cfg.api_key,
        profile=profile,
    )


# live mode
# capture mic/speaker audio dan stream setiap source ke WhisperLive
# Args:
#     config: konfigurasi sesi.
#     on_result: callback setiap ada hasil transcript yang sudah di-emit.
#     stop_event: event eksternal untuk menghentikan sesi.
#     on_log: callback log/status untuk UI; None berarti print ke stdout.
#     on_merged_entry: callback entry hasil merge untuk GUI.
#     install_signal_handlers: False saat berjalan di thread GUI/QThread.
# Returns:
#     WhisperLiveSessionStats: statistik runtime untuk sesi yang telah selesai. 

def run_whisperlive_session(
    config: WhisperLiveSessionConfig,
    *,
    on_result: TranscriptCallback | None = None, # hasil yang telah di merge
    stop_event: threading.Event | None = None, # event eksternal untuk menghentikan sesi
    on_log: LogCallback | None = None, # callback log/status untuk UI; None berarti print ke stdout
    on_merged_entry: MergedEntryCallback | None = None, # callback entry hasil merge untuk GUI
    install_signal_handlers: bool = True, # False saat berjalan di thread GUI/QThread
) -> WhisperLiveSessionStats:
    cfg = config
    stats = WhisperLiveSessionStats()
    configure_process_logging(
        session_id=cfg.session_id,
        include_hot_path=cfg.process_log_include_hot_path,
        summary_interval_seconds=cfg.process_log_summary_interval_seconds,
    )
    _stop = stop_event or threading.Event()
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(maxsize=cfg.max_chunk_queue_size)
    reader_threads: list[threading.Thread] = []
    capture_streams: list[object] = []

    def _print(text: str) -> None:
        if on_log is not None:
            on_log(text)
        else:
            print(text, flush=True)

    # buat log transcript
    transcript_log: TranscriptLog | None = None
    if cfg.log_transcript:
        transcript_log = TranscriptLog.load(cfg.log_transcript) if cfg.resume_transcript_log else TranscriptLog(cfg.log_transcript)
        transcript_log.save()

    transcript_merger = TranscriptMerger(max_emitted_keys=cfg.merger_emitted_cache_max_entries)
    transcript_lock = threading.Lock()
    
    candidate_transcript_keys: set[tuple[str, int, str]] = set()
    candidate_transcript_order: deque[tuple[str, int, str]] = deque()
    active_sources = _requested_sources(cfg.source) # mengambil source yang aktif (mic, speaker, atau keduanya)
    partial_preview = _PartialTranscriptPreview() if cfg.show_partials else None
    chunk_archive = _ChunkArchive(cfg.chunk_archive_dir) if cfg.chunk_archive_dir is not None else None
    rolling_archive = (
        _RollingAudioArchive(
            cfg.rolling_audio_archive_dir,
            session_id=cfg.session_id,
            segment_seconds=cfg.rolling_audio_segment_seconds,
            metadata={
                "session_id": cfg.session_id,
                "model": cfg.profile.model,
                "language": cfg.profile.language,
                "source": cfg.source,
                "chunk_seconds": cfg.chunk_seconds,
                "audio_format": cfg.audio_format,
                "server_vad": {
                    "mic": cfg.mic_server_vad,
                    "speaker": cfg.speaker_server_vad,
                    "vad_threshold": cfg.profile.vad_threshold,
                    "vad_parameters": cfg.profile.vad_parameters(),
                },
                "mic_preprocess": {
                    "target_rms_db": cfg.mic_target_rms_db,
                    "max_normalization_gain_db": cfg.mic_max_normalization_gain_db,
                },
                "speaker_preprocess": {
                    "target_rms_db": cfg.speaker_target_rms_db,
                    "max_normalization_gain_db": cfg.speaker_max_normalization_gain_db,
                },
            },
        )
        if cfg.rolling_audio_archive_dir is not None
        else None
    )
    quality_tracker = TranscriptQualityTracker()
    if rolling_archive is not None:
        stats.rolling_audio_dir = rolling_archive.root
    
    log_process_event(
        "client.session_start",
        session_id=cfg.session_id,
        server_host=cfg.server_host,
        server_port=cfg.server_port,
        use_wss=cfg.use_wss,
        model=cfg.profile.model,
        language=cfg.profile.language,
        source=cfg.source,
        sample_rate=cfg.sample_rate,
        chunk_seconds=cfg.chunk_seconds,
        audio_format=cfg.audio_format,
        ready_timeout=cfg.ready_timeout,
        mic_device=cfg.mic_device,
        speaker_device=cfg.speaker_device,
        mic_server_vad=cfg.mic_server_vad,
        speaker_server_vad=cfg.speaker_server_vad,
        server_vad_threshold=cfg.profile.vad_threshold,
        server_vad_parameters=cfg.profile.vad_parameters(),
        auto_reconnect=cfg.auto_reconnect,
        reconnect_initial_backoff_seconds=cfg.reconnect_initial_backoff_seconds,
        reconnect_max_backoff_seconds=cfg.reconnect_max_backoff_seconds,
        reconnect_buffer_seconds=cfg.reconnect_buffer_seconds,
        process_log_include_hot_path=cfg.process_log_include_hot_path,
        process_log_summary_interval_seconds=cfg.process_log_summary_interval_seconds,
        candidate_cache_max_entries=cfg.candidate_cache_max_entries,
        merger_emitted_cache_max_entries=cfg.merger_emitted_cache_max_entries,
        mic_preprocess={
            "target_rms_db": cfg.mic_target_rms_db,
            "max_normalization_gain_db": cfg.mic_max_normalization_gain_db,
        },
        speaker_preprocess={
            "target_rms_db": cfg.speaker_target_rms_db,
            "max_normalization_gain_db": cfg.speaker_max_normalization_gain_db,
        },
        rolling_audio_archive_dir=cfg.rolling_audio_archive_dir,
        rolling_audio_segment_seconds=cfg.rolling_audio_segment_seconds,
        resume_transcript_log=cfg.resume_transcript_log,
    )

    def _remember_candidate_key(key: tuple[str, int, str]) -> bool:
        if key in candidate_transcript_keys:
            return False
        candidate_transcript_keys.add(key)
        candidate_transcript_order.append(key)
        while len(candidate_transcript_order) > max(1, cfg.candidate_cache_max_entries):
            oldest = candidate_transcript_order.popleft()
            candidate_transcript_keys.discard(oldest)
        return True

    def handle_status(source: str, message: dict) -> None:
        # Status WebSocket dipisah dari transcript supaya UI bisa menampilkan
        # koneksi, SERVER_READY, END_OF_AUDIO, dan error secara jelas.
        status = str(message.get("status") or message.get("message") or "UNKNOWN")
        label = source.upper()
        if status == "CLIENT_CONNECTING":
            log_process_event("client.ws_connect", source=source, details=message)
            _print(
                f"[{label}] connecting to {message.get('url')} "
                f"model={message.get('model')} ready_timeout={message.get('ready_timeout')}s"
            )
            _log.info("whisperlive connecting source=%s details=%s", source, message)
            return
        if status == "CLIENT_SOCKET_OPEN":
            log_process_event("client.ws_open", source=source, details=message)
            _print(f"[{label}] websocket connected")
            _log.info("whisperlive socket open source=%s details=%s", source, message)
            return
        if status == "CLIENT_OPTIONS_SENT":
            log_process_event("client.options_sent", source=source, details=message)
            _print(
                f"[{label}] options sent "
                f"audio_format={message.get('audio_format')} server_vad={message.get('server_vad')}"
            )
            _log.info("whisperlive options sent source=%s details=%s", source, message)
            return
        if status == "CLIENT_WAITING_SERVER_READY":
            _print(
                f"[{label}] waiting server ready "
                f"waited={message.get('waited_seconds')}s remaining={message.get('remaining_seconds')}s"
            )
            _log.info("whisperlive waiting server ready source=%s details=%s", source, message)
            return
        if status == "SERVER_READY":
            log_process_event("client.server_ready", source=source, details=message)
            _print(f"[{label}] server ready backend={message.get('backend')}")
            _log.info("whisperlive stream ready source=%s details=%s", source, message)
            return
        if status == "CLIENT_AUDIO_FINISHED":
            log_process_event("client.end_of_audio", source=source, details=message)
            _print(f"[{label}] status CLIENT_AUDIO_FINISHED: {message}")
            _log.info("whisperlive audio finished source=%s details=%s", source, message)
            return
        if status == "CLIENT_RECV_ERROR":
            log_process_event("client.ws_receive_error", source=source, details=message)
            _print(
                f"[{label}] receive error: {message.get('message')} "
                f"(chunks={message.get('chunks_sent')} messages={message.get('messages_received')} "
                f"segments={message.get('segments_received')})"
            )
            _log.warning("whisperlive receive error source=%s details=%s", source, message)
            if not cfg.auto_reconnect:
                _stop.set()
            return
        if status == "CLIENT_REMOTE_CLOSED":
            log_process_event("client.ws_remote_closed", source=source, details=message)
            _print(
                f"[{label}] remote closed websocket "
                f"(chunks={message.get('chunks_sent')} messages={message.get('messages_received')} "
                f"segments={message.get('segments_received')} elapsed={message.get('elapsed_seconds')}s)"
            )
            _log.warning("whisperlive remote closed websocket source=%s details=%s", source, message)
            if not cfg.auto_reconnect:
                _stop.set()
            return
        if status in {"DISCONNECT", "CLIENT_READY_TIMEOUT"}:
            log_process_event("client.ws_disconnected", source=source, details=message)
            _print(f"[{label}] disconnected: {status} {message.get('message') or ''}".rstrip())
            _log.warning("whisperlive disconnected source=%s details=%s", source, message)
            if not cfg.auto_reconnect:
                _stop.set()
            return
        if status == "CLIENT_CLOSED":
            log_process_event("client.ws_closed", source=source, details=message)
            _print(
                f"[{label}] client closed "
                f"chunks={message.get('chunks_sent')} messages={message.get('messages_received')} "
                f"segments={message.get('segments_received')}"
            )
            _log.info("whisperlive client closed source=%s details=%s", source, message)
            return
        if status == "CLIENT_CLOSING":
            log_process_event("client.ws_closing", source=source, details=message)
            _log.info("whisperlive client closing source=%s details=%s", source, message)
            return
        if status in {"WAIT", "ERROR"}:
            log_process_event("client.server_status", source=source, details=message)
            _print(f"[{label}] server status {status}: {message.get('message')}")
            _log.warning("whisperlive server status source=%s details=%s", source, message)
            if status == "ERROR":
                _stop.set()
            return
        if "segments" in message:
            _log.info("whisperlive transcript status source=%s details=%s", source, message)
            return
        _print(f"[{label}] status {status}: {message}")
        _log.warning("whisperlive status source=%s details=%s", source, message)

    def handle_transcript(source: str, segments: list[dict], message: dict) -> None:
        # Server dapat mengirim candidate dan stable segment. Candidate tetap
        # disimpan untuk audit, tetapi output utama user memakai hasil merger.
        log_process_event(
            "client.transcript_received",
            source=source,
            segment_count=len(segments),
            completed_count=sum(1 for segment in segments if segment.get("completed")),
            reliability=[
                {
                    "score": segment.get("reliability_score"),
                    "action": segment.get("reliability_action"),
                    "completed": segment.get("completed"),
                    "text": str(segment.get("text", ""))[:160],
                }
                for segment in segments
            ],
            uid=message.get("uid"),
        )
        _log.info(
            "whisperlive transcript received source=%s segment_count=%s completed=%s",
            source,
            len(segments),
            sum(1 for segment in segments if segment.get("completed")),
        )
        if partial_preview is not None:
            partial_preview.show(source, segments)
        for event in _events_from_segments(source, cfg.profile.model, cfg.profile.language, segments):
            quality_tracker.observe_event(event)
            if not event.completed and transcript_log is not None and event.result.text.strip():
                # Candidate disimpan sekali per window agar transcript audit
                # tidak kosong ketika TVE menahan hasil sebagai belum final.
                candidate_key = (
                    event.result.source,
                    int(event.result.start_seconds // 3),
                    " ".join(event.result.text.lower().split())[:180],
                )
                if _remember_candidate_key(candidate_key):
                    label = MEETING_SOURCE_LABELS.get(event.result.source, event.result.source.upper())
                    display = (
                        f"[{_format_timestamp(event.result.start_seconds)} - "
                        f"{_format_timestamp(event.result.end_seconds)}] "
                        f"[{label}] {event.result.text.strip()}"
                    )
                    transcript_log.append_result(
                        event.result,
                        label=label,
                        display=display,
                        completed=False,
                        stability="candidate",
                        reliability_score=event.reliability_score,
                        reliability_action=event.reliability_action,
                    )
                    log_process_event(
                        "client.candidate_saved",
                        source=source,
                        score=event.reliability_score,
                        action=event.reliability_action,
                        start_seconds=round(event.result.start_seconds, 3),
                        end_seconds=round(event.result.end_seconds, 3),
                        text=event.result.text[:160],
                    )
            with transcript_lock:
                entries = transcript_merger.add_result(event.result, completed=event.completed)
                pending_count = transcript_merger.pending_count
            log_process_event(
                "client.merger_add",
                source=source,
                completed=event.completed,
                emitted_count=len(entries),
                pending_count=pending_count,
                start_seconds=round(event.result.start_seconds, 3),
                end_seconds=round(event.result.end_seconds, 3),
                text=event.result.text[:160],
            )
            for entry in entries:
                log_process_event(
                    "client.merger_emit",
                    source=entry.result.source,
                    start_seconds=round(entry.result.start_seconds, 3),
                    end_seconds=round(entry.result.end_seconds, 3),
                    text=entry.result.text[:160],
                )
                _emit_merged_entry(entry, transcript_log, on_result, on_log=on_log, on_merged_entry=on_merged_entry)
                quality_tracker.observe_emit(entry.result.source)
                stats.add_result_received()

    # policy reconnect untuk setiap source, agar jika koneksi terputus, client akan mencoba reconnect sesuai policy yang ditentukan
    reconnect_policy = ReconnectPolicy(
        enabled=cfg.auto_reconnect,
        initial_backoff_seconds=cfg.reconnect_initial_backoff_seconds,
        max_backoff_seconds=cfg.reconnect_max_backoff_seconds,
        buffer_seconds=cfg.reconnect_buffer_seconds,
    )

    # client setiap source (mic/speaker) yang akan mengirim audio ke server WhisperLive atau koneksi terputus, client akan mencoba reconnect sesuai policy yang ditentukan.
    clients = {
        source: _BufferedReconnectClient(
            source,
            _connection_config_for_source(cfg, source),
            policy=reconnect_policy,
            on_transcript=handle_transcript,
            on_status=handle_status,
        )
        for source in active_sources
    }

    try:
        _connect_clients_parallel(clients, allow_initial_failure=cfg.auto_reconnect) # menghubungkan semua client ke server WhisperLive secara paralel, jika gagal, akan mencoba reconnect sesuai policy yang ditentukan

        capture_streams, reader_threads = _build_capture_streams(cfg, chunk_queue, _stop, stats) # stream audio untuk setiap source (mic/speaker) dan thread untuk membaca audio dari stream dan memasukkannya ke dalam queue chunk audio
        
        # tidak ada stream audio yang tersedia, maka akan menampilkan error dan mengembalikan stats
        if not capture_streams:
            _print("[ERROR] Tidak ada audio stream yang tersedia. Periksa perangkat audio Anda.")
            return stats

        active_streams: list[object] = []
        for stream in capture_streams:
            try:
                stream.start()                  # mulai menangkap audio dari perangkat
                active_streams.append(stream)   # menambahkan stream yang aktif ke daftar active_streams
            except Exception as exc:
                _log.warning("whisperlive session: failed to start stream: %s", exc)

        if not active_streams:
            _print("[ERROR] Gagal memulai audio stream. Periksa izin mikrofon / audio device.")
            return stats

        original_sigint = None
        original_sigbreak = None
        if install_signal_handlers:
            original_sigint = signal.getsignal(signal.SIGINT)
            original_sigbreak = signal.getsignal(signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else None

            def _sigint_handler(signum: int, frame: object) -> None:
                _log.info("whisperlive session: stop signal received signal=%s", signum)
                _stop.set()

            signal.signal(signal.SIGINT, _sigint_handler)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, _sigint_handler)
        for thread in reader_threads:
            thread.start()

        session_start = monotonic() # mencatat waktu mulai sesi untuk menghitung durasi sesi

        # menampilkan informasi sesi live WhisperLive ke stdout atau callback log
        _print("=" * 60)
        _print("Live transcription aktif via WhisperLive. Tekan Ctrl+C untuk berhenti.")
        _print(
            f"Server: {cfg.server_host}:{cfg.server_port}  |  "
            f"Model: {cfg.profile.model}  |  VAD: {cfg.profile.vad_threshold:.2f}"
        )
        _print(
            "Server VAD: "
            + ", ".join(f"{source.upper()}={_server_vad_for_source(cfg, source)}" for source in active_sources)
        )
        _print(
            "Debug metadata: source-specific preprocessing, replay-ready chunks, and per-source server VAD decisions."
        )
        _print(f"Sumber: {', '.join(source.upper() for source in active_sources)}")
        _print("=" * 60)

        try:
            while not _stop.is_set():
                # Loop utama hanya mengambil chunk yang sudah lolos preprocessing dan mengirimnya ke koneksi WebSocket sesuai source.
                try:
                    chunk = chunk_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                client = clients.get(chunk.source)
                if client is None:
                    stats.add_chunks_dropped()
                    continue
                try:
                    if chunk_archive is not None:
                        chunk_archive.save(chunk)
                    if rolling_archive is not None:
                        rolling_archive.save(chunk)
                    outcome = client.send(chunk)
                    stats.add_send_outcome(outcome)
                    log_process_event(
                        "client.chunk_sent",
                        source=chunk.source,
                        start_seconds=round(chunk.start_seconds, 3),
                        duration_seconds=round(chunk.duration_seconds, 3),
                        frame_count=chunk.frame_count,
                        input_rms_db=round(chunk.input_rms_db, 2),
                        output_rms_db=round(chunk.rms_db, 2),
                        sent=outcome.sent,
                        buffered=outcome.buffered,
                        dropped=outcome.dropped,
                        chunks_sent=stats.chunks_sent,
                        chunks_buffered=stats.chunks_buffered,
                    )
                    _log.debug(
                        "whisperlive chunk sent source=%s duration=%.2f rms=%.2f",
                        chunk.source,
                        chunk.duration_seconds,
                        chunk.rms_db,
                    )
                except Exception as exc:
                    stats.add_chunks_dropped()
                    log_process_event(
                        "client.chunk_send_error",
                        source=chunk.source,
                        error=str(exc),
                        chunks_dropped=stats.chunks_dropped,
                    )
                    _log.error("failed to send %s chunk to WhisperLive: %s", chunk.source, exc)
        finally:
            # Stop harus graceful: hentikan capture, drain queue, kirim
            # END_OF_AUDIO, tunggu server flush, lalu emit sisa merger.
            _stop.set()
            log_process_event("client.finalization_start")
            for stream in active_streams:
                try:
                    stream.stop()  # type: ignore[attr-defined]
                except Exception as exc:
                    _log.debug("error stopping stream: %s", exc)
            for thread in reader_threads:
                thread.join(timeout=3.0)
            if install_signal_handlers:
                if original_sigint is not None:
                    signal.signal(signal.SIGINT, original_sigint)
                if hasattr(signal, "SIGBREAK") and original_sigbreak is not None:
                    signal.signal(signal.SIGBREAK, original_sigbreak)

            drained = _drain_pending_chunks(chunk_queue, clients, stats, chunk_archive, rolling_archive)
            if drained:
                _print(f"Finalisasi: mengirim {drained} chunk tersisa...")
            flushed_reconnect = _flush_reconnect_buffers(clients, stats, timeout_seconds=min(5.0, cfg.final_drain_seconds))
            if flushed_reconnect:
                _print(f"Finalisasi: mengirim {flushed_reconnect} chunk buffer reconnect...")
            for client in clients.values():
                try:
                    client.finish_audio()
                except Exception as exc:
                    log_process_event("client.end_of_audio_error", source=client.source, error=str(exc))
                    _log.warning("failed to finish WhisperLive audio source=%s: %s", client.source, exc)
            if cfg.final_drain_seconds > 0:
                _print(f"Finalisasi: menunggu hasil server {cfg.final_drain_seconds:g}s...")
                _wait_for_final_results(stats, cfg.final_drain_seconds)

            elapsed = monotonic() - session_start
            with transcript_lock:
                flushed_entries = transcript_merger.flush()
            log_process_event(
                "client.merger_flush",
                flushed_count=len(flushed_entries),
                results_before_flush=stats.results_received,
            )
            for entry in flushed_entries:
                log_process_event(
                    "client.merger_emit",
                    source=entry.result.source,
                    start_seconds=round(entry.result.start_seconds, 3),
                    end_seconds=round(entry.result.end_seconds, 3),
                    text=entry.result.text[:160],
                    final_flush=True,
                )
                _emit_merged_entry(entry, transcript_log, on_result, on_log=on_log, on_merged_entry=on_merged_entry)
                quality_tracker.observe_emit(entry.result.source)
                stats.add_result_received()

            if rolling_archive is not None:
                rolling_archive.close()
                _print(f"Rolling audio archive: {rolling_archive.root}")

            transcript_summary = quality_tracker.summary()
            stats.set_transcript_summary(transcript_summary)

            _print("\n" + "=" * 60)
            _print(f"Sesi selesai. Durasi: {elapsed:.0f}s")
            _print(
                f"Chunks terkirim: {stats.chunks_sent}  |  "
                f"Buffered: {stats.chunks_buffered}  |  "
                f"Dropped: {stats.chunks_dropped}  |  "
                f"Reconnect: {stats.reconnect_successes}/{stats.reconnect_attempts}  |  "
                f"Results: {stats.results_received}"
            )
            if transcript_log is not None:
                _print(f"Transcript log: {cfg.log_transcript}")
            if chunk_archive is not None:
                _print(f"Chunk archive: {chunk_archive.root}")
            if cfg.source in ("speaker", "both") and stats.chunks_sent == 0:
                _print(
                    "[WARN] Tidak ada chunk audio speaker yang dikirim. "
                    "Pastikan ada audio meeting/media yang sedang diputar dan device loopback benar."
                )
                log_process_event(
                    "client.no_audio_chunks",
                    source="speaker",
                    reason="no_speaker_chunks_sent",
                    hint="play meeting/media audio and verify loopback device selection",
                )
            _print(
                "Transcript summary: "
                f"stable={transcript_summary.get('stable', 0)} "
                f"candidate={transcript_summary.get('candidate', 0)} "
                f"stable_ratio={transcript_summary.get('stable_ratio')}"
            )
            _print("=" * 60)
            log_process_event(
                "client.session_summary",
                transcript=transcript_summary,
                rolling_audio_dir=str(rolling_archive.root) if rolling_archive is not None else None,
            )
            log_process_event(
                "client.finalization_complete",
                chunks_sent=stats.chunks_sent,
                chunks_buffered=stats.chunks_buffered,
                chunks_dropped=stats.chunks_dropped,
                reconnect_attempts=stats.reconnect_attempts,
                reconnect_successes=stats.reconnect_successes,
                results_received=stats.results_received,
            )
    finally:
        for client in clients.values():
            client.close(send_end_of_audio=False)
        flush_process_log_summaries()
        clear_process_log_context()

    return stats


def _connect_clients_parallel(
    clients: dict[str, _BufferedReconnectClient],
    *,
    allow_initial_failure: bool = False,
) -> None:
    """Hubungkan mic dan speaker paralel agar startup live lebih cepat."""
    errors: list[tuple[str, BaseException]] = []
    error_lock = threading.Lock()

    def connect_one(source: str, client: _BufferedReconnectClient) -> None:
        try:
            client.connect_initial()
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
        if allow_initial_failure:
            for source, exc in errors:
                log_process_event("client.initial_connect_failed", source=source, error=str(exc))
                _log.warning("initial WhisperLive connect failed source=%s error=%s; reconnect will continue", source, exc)
            return
        for client in clients.values():
            client.close(send_end_of_audio=False)
        source, exc = errors[0]
        raise RuntimeError(f"failed to connect WhisperLive stream '{source}': {exc}") from exc


def _drain_pending_chunks(
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    clients: dict[str, _BufferedReconnectClient],
    stats: WhisperLiveSessionStats,
    chunk_archive: "_ChunkArchive | None" = None,
    rolling_archive: "_RollingAudioArchive | None" = None,
) -> int:
    """Kirim sisa chunk di queue sebelum END_OF_AUDIO."""
    drained = 0
    while True:
        try:
            chunk = chunk_queue.get_nowait()
        except queue.Empty:
            return drained

        client = clients.get(chunk.source)
        if client is None:
            stats.add_chunks_dropped()
            log_process_event(
                "client.chunk_send_drop",
                source=chunk.source,
                reason="missing_client_during_final_drain",
                chunks_dropped=stats.chunks_dropped,
            )
            continue
        try:
            if chunk_archive is not None:
                chunk_archive.save(chunk)
            if rolling_archive is not None:
                rolling_archive.save(chunk)
            outcome = client.send(chunk)
            stats.add_send_outcome(outcome)
            drained += outcome.sent
            log_process_event(
                "client.chunk_sent",
                source=chunk.source,
                start_seconds=round(chunk.start_seconds, 3),
                duration_seconds=round(chunk.duration_seconds, 3),
                frame_count=chunk.frame_count,
                input_rms_db=round(chunk.input_rms_db, 2),
                output_rms_db=round(chunk.rms_db, 2),
                sent=outcome.sent,
                buffered=outcome.buffered,
                dropped=outcome.dropped,
                chunks_sent=stats.chunks_sent,
                chunks_buffered=stats.chunks_buffered,
                final_drain=True,
            )
            _log.debug(
                "whisperlive final chunk sent source=%s duration=%.2f rms=%.2f",
                chunk.source,
                chunk.duration_seconds,
                chunk.rms_db,
            )
        except Exception as exc:
            stats.add_chunks_dropped()
            log_process_event(
                "client.chunk_send_error",
                source=chunk.source,
                error=str(exc),
                chunks_dropped=stats.chunks_dropped,
                final_drain=True,
            )
            _log.error("failed to send final %s chunk to WhisperLive: %s", chunk.source, exc)


def _flush_reconnect_buffers(
    clients: dict[str, _BufferedReconnectClient],
    stats: WhisperLiveSessionStats,
    *,
    timeout_seconds: float,
) -> int:
    deadline = perf_counter() + max(0.0, timeout_seconds)
    total_sent = 0
    while perf_counter() <= deadline:
        pending_before = sum(client.buffered_count for client in clients.values())
        if pending_before == 0:
            return total_sent

        progressed = False
        for source, client in clients.items():
            outcome = client.flush_buffer()
            if outcome.sent or outcome.buffered or outcome.dropped or outcome.reconnect_attempts:
                log_process_event(
                    "client.reconnect_buffer_flush_attempt",
                    source=source,
                    sent=outcome.sent,
                    dropped=outcome.dropped,
                    buffered_remaining=client.buffered_count,
                    buffered_seconds=round(client.buffered_seconds, 3),
                    reconnect_attempts=outcome.reconnect_attempts,
                    reconnect_successes=outcome.reconnect_successes,
                )
            stats.add_send_outcome(outcome)
            total_sent += outcome.sent
            progressed = progressed or bool(outcome.sent or outcome.dropped or outcome.reconnect_successes)

        if sum(client.buffered_count for client in clients.values()) == 0:
            return total_sent
        if not progressed:
            threading.Event().wait(0.25)

    remaining = sum(client.buffered_count for client in clients.values())
    if remaining:
        stats.add_chunks_dropped(remaining)
        log_process_event("client.reconnect_buffer_flush_timeout", remaining_chunks=remaining)
    return total_sent
