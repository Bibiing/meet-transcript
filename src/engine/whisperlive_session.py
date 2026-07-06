from __future__ import annotations

import logging
import queue
import signal
import threading
from collections import deque
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
from src.engine.transcript_merger import MEETING_SOURCE_LABELS, MergedTranscriptEntry, TranscriptMerger
from src.engine.whisper import TranscriptionResult, TranscriptionSegment
from src.engine.whisperlive_client import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
    WhisperLiveStreamClient,
)
from src.utils.logging import TranscriptLog, log_process_event
from src.utils.os_detector import AudioBackend, get_audio_backend

_log = logging.getLogger(__name__)


# konfigurasi default untuk sesi live WhisperLive
@dataclass(frozen=True, slots=True)
class WhisperLiveSessionConfig:
    # server dan koneksi
    server_host: str = "localhost"
    server_port: int = 9090
    use_wss: bool = False
    api_key: str | None = None
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    
    # audio capture dan preprocessing
    chunk_seconds: float = 0.5
    audio_format: Literal["float32", "int16", "uint8"] = "int16"
    source: Literal["mic", "speaker", "both"] = "both"
    sample_rate: int | None = None
    block_size: int = 1_024
    queue_size: int = 128
    mic_device: int | str | None = None
    speaker_device: int | str | None = None
    
    # vad (voice activity detection)
    mic_client_vad: bool = True
    speaker_client_vad: bool = False
    max_chunk_queue_size: int = 32
    
    # koneksi dan reconnect
    auto_reconnect: bool = True
    reconnect_initial_backoff_seconds: float = 1.0
    reconnect_max_backoff_seconds: float = 30.0
    reconnect_buffer_seconds: float = 30.0
    final_drain_seconds: float = 10.0
    
    # output dan logging
    show_partials: bool = True
    log_transcript: Path | None = None
    chunk_archive_dir: Path | None = None
    
    profile: WhisperLiveProfile = field(default_factory=WhisperLiveProfile)

# stats untuk sesi live WhisperLive
@dataclass
class WhisperLiveSessionStats:
    chunks_sent: int = 0
    chunks_dropped: int = 0
    chunks_buffered: int = 0
    reconnect_attempts: int = 0
    reconnect_successes: int = 0
    results_received: int = 0
    start_time: float = field(default_factory=perf_counter)

    # waktu yang telah berlalu sejak sesi dimulai
    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.start_time

# callback type untuk menerima hasil transcript, log, dan entry merger
TranscriptCallback = Callable[[TranscriptionResult], None]
LogCallback = Callable[[str], None]
MergedEntryCallback = Callable[[MergedTranscriptEntry], None]

# reconnect policy untuk client WhisperLive
@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    enabled: bool = True
    initial_backoff_seconds: float = 1.0    # waktu tunggu awal sebelum mencoba reconnect
    max_backoff_seconds: float = 30.0       # batas waktu tunggu maksimal sebelum mencoba reconnect
    buffer_seconds: float = 30.0            # batas buffer audio lokal untuk reconnect, dalam detik

# hasil pengiriman chunk audio ke server WhisperLive
@dataclass(frozen=True, slots=True)
class SendOutcome:
    sent: int = 0                   # jumlah chunk yang berhasil dikirim ke server
    buffered: int = 0               # jumlah chunk yang berhasil disimpan di buffer lokal untuk reconnect
    dropped: int = 0                # jumlah chunk yang diabaikan
    reconnect_attempts: int = 0     # jumlah upaya reconnect
    reconnect_successes: int = 0    # jumlah reconnect yang berhasil

# client wrapper dengan local audio buffer dan reconnect
# capture thread tidak boleh berhenti hanya karena WebSocket terputus. Wrapper ini menahan chunk per source dalam bounded ring buffer, mencoba reconnect memakai exponential backoff, lalu mengirim ulang buffer secara FIFO.
class _BufferedReconnectClient:
    def __init__(
        self,
        source: Literal["mic", "speaker"],
        connection_config: WhisperLiveConnectionConfig,
        *,
        policy: ReconnectPolicy,
        on_transcript: Callable[[str, list[dict], dict], None],
        on_status: Callable[[str, dict], None],
    ) -> None:
        self.source = source
        self.connection_config = connection_config
        self.policy = policy
        self._on_transcript = on_transcript
        self._upstream_status = on_status
        self._client: WhisperLiveStreamClient | None = None
        self._connected = False
        self._attempt = 0
        self._next_reconnect_at = 0.0
        self._buffer: deque[PreprocessedAudioChunk] = deque()
        self._buffered_seconds = 0.0
        self._lock = threading.RLock()

    # thread-safe property untuk memeriksa apakah client terhubung dan siap mengirim chunk
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected and self._client is not None and self._client.is_ready
        
    # memeriksa jumlah chunk yang tersimpan di buffer lokal
    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    # memeriksa total durasi audio yang tersimpan di buffer lokal
    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            return self._buffered_seconds

    # memaksa koneksi awal ke server WhisperLive, mengabaikan kebijakan reconnect
    def connect_initial(self) -> SendOutcome:
        return self._connect(force=True)

    # mengirim chunk audio ke server WhisperLive, atau menyimpannya di buffer lokal jika tidak terhubung. Jika reconnect diaktifkan, mencoba reconnect dan mengirim ulang buffer.
    def send(self, chunk: PreprocessedAudioChunk) -> SendOutcome:
        with self._lock:
            outcome = self._flush_locked() # flush buffer sebelum mengirim chunk baru

            if self.connected:
                try:
                    self._client.send_chunk(chunk)  # mengirim chunk ke server

                    # outcome dengan sent=1 chunk berhasil dikirim
                    return _combine_outcome(outcome, SendOutcome(sent=1))
                
                # jika gagal mengirim chunk karena koneksi terputus, tandai sebagai disconnected
                except Exception as exc:
                    self._mark_disconnected_locked(f"send failed: {exc}")
                
            # jika tidak terhubung, simpan chunk di buffer lokal
            buffered = self._buffer_chunk_locked(chunk)
            outcome = _combine_outcome(outcome, buffered) 

            # jika reconnect diaktifkan, coba reconnect dan flush buffer
            if self.policy.enabled:
                outcome = _combine_outcome(outcome, self._connect(force=False))
                outcome = _combine_outcome(outcome, self._flush_locked())
            return outcome

    # memaksa flush buffer lokal ke server WhisperLive, jika terhubung. Jika tidak terhubung, mencoba reconnect jika diaktifkan.
    def flush_buffer(self) -> SendOutcome:
        with self._lock:
            outcome = SendOutcome() # 
            
            if not self.connected and self.policy.enabled:
                outcome = _combine_outcome(outcome, self._connect(force=False)) 

            return _combine_outcome(outcome, self._flush_locked())

    def finish_audio(self) -> None:
        with self._lock:
            if self._client is not None and self.connected:
                self._client.finish_audio()

    def close(self, *, send_end_of_audio: bool = True) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._connected = False
        if client is not None:
            client.close(send_end_of_audio=send_end_of_audio)

    def _connect(self, *, force: bool) -> SendOutcome:
        if not self.policy.enabled and not force:
            return SendOutcome()
        now = perf_counter()
        if not force and now < self._next_reconnect_at:
            return SendOutcome()

        self._attempt += 1
        attempt = self._attempt
        self._emit_status(
            "CLIENT_RECONNECTING" if attempt > 1 else "CLIENT_CONNECTING_RETRY",
            attempt=attempt,
            buffered_chunks=len(self._buffer),
            buffered_seconds=round(self._buffered_seconds, 3),
        )
        self.close(send_end_of_audio=False)

        client = WhisperLiveStreamClient(
            self.source,
            self.connection_config,
            on_transcript=self._on_transcript,
            on_status=self._handle_status,
        )
        try:
            client.connect()
        except Exception as exc:
            if not self.policy.enabled:
                raise
            delay = self._schedule_next_reconnect_locked()
            self._client = None
            self._connected = False
            self._emit_status(
                "CLIENT_RECONNECT_FAILED",
                attempt=attempt,
                next_retry_seconds=round(delay, 3),
                error=str(exc),
                buffered_chunks=len(self._buffer),
                buffered_seconds=round(self._buffered_seconds, 3),
            )
            return SendOutcome(reconnect_attempts=1)

        self._client = client
        self._connected = True
        self._attempt = 0
        self._next_reconnect_at = 0.0
        self._emit_status(
            "CLIENT_RECONNECTED" if attempt > 1 else "CLIENT_CONNECTED_READY",
            attempt=attempt,
            buffered_chunks=len(self._buffer),
            buffered_seconds=round(self._buffered_seconds, 3),
        )
        return SendOutcome(reconnect_attempts=1, reconnect_successes=1)

    def _handle_status(self, source: str, message: dict) -> None:
        status = str(message.get("status") or "")
        if status == "SERVER_READY":
            with self._lock:
                self._connected = True
        elif status in {"CLIENT_RECV_ERROR", "CLIENT_REMOTE_CLOSED", "DISCONNECT", "CLIENT_READY_TIMEOUT"}:
            with self._lock:
                self._mark_disconnected_locked(status)
        self._upstream_status(source, message)

    def _mark_disconnected_locked(self, reason: str) -> None:
        self._connected = False
        self._schedule_next_reconnect_locked()
        log_process_event(
            "client.reconnect_scheduled",
            source=self.source,
            reason=reason,
            attempt=self._attempt,
            next_reconnect_at=round(self._next_reconnect_at, 3),
            buffered_chunks=len(self._buffer),
            buffered_seconds=round(self._buffered_seconds, 3),
        )

    def _schedule_next_reconnect_locked(self) -> float:
        exponent = max(0, self._attempt - 1)
        delay = min(
            max(0.1, self.policy.max_backoff_seconds),
            max(0.1, self.policy.initial_backoff_seconds) * (2**exponent),
        )
        self._next_reconnect_at = perf_counter() + delay
        return delay

    def _buffer_chunk_locked(self, chunk: PreprocessedAudioChunk) -> SendOutcome:
        if self.policy.buffer_seconds <= 0:
            return SendOutcome(dropped=1)

        dropped = 0
        while self._buffer and self._buffered_seconds + chunk.duration_seconds > self.policy.buffer_seconds:
            old = self._buffer.popleft()
            self._buffered_seconds = max(0.0, self._buffered_seconds - old.duration_seconds)
            dropped += 1

        if chunk.duration_seconds > self.policy.buffer_seconds:
            return SendOutcome(dropped=dropped + 1)

        self._buffer.append(chunk)
        self._buffered_seconds += chunk.duration_seconds
        log_process_event(
            "client.chunk_buffered",
            source=self.source,
            buffered_chunks=len(self._buffer),
            buffered_seconds=round(self._buffered_seconds, 3),
            dropped_oldest=dropped,
        )
        return SendOutcome(buffered=1, dropped=dropped)

    def _flush_locked(self) -> SendOutcome:
        if not self.connected or not self._buffer:
            return SendOutcome()

        sent = 0
        while self._buffer and self.connected:
            chunk = self._buffer.popleft()
            self._buffered_seconds = max(0.0, self._buffered_seconds - chunk.duration_seconds)
            try:
                self._client.send_chunk(chunk)  # type: ignore[union-attr]
                sent += 1
            except Exception as exc:
                self._buffer.appendleft(chunk)
                self._buffered_seconds += chunk.duration_seconds
                self._mark_disconnected_locked(f"buffer flush failed: {exc}")
                break

        if sent:
            log_process_event(
                "client.reconnect_buffer_flushed",
                source=self.source,
                sent=sent,
                remaining_chunks=len(self._buffer),
                remaining_seconds=round(self._buffered_seconds, 3),
            )
        return SendOutcome(sent=sent)

    def _emit_status(self, status: str, **fields: object) -> None:
        message = {"uid": f"{self.source}-reconnect", "status": status, **fields}
        log_process_event("client.reconnect_status", source=self.source, details=message)
        self._upstream_status(self.source, message)


def _combine_outcome(left: SendOutcome, right: SendOutcome) -> SendOutcome:
    return SendOutcome(
        sent=left.sent + right.sent,
        buffered=left.buffered + right.buffered,
        dropped=left.dropped + right.dropped,
        reconnect_attempts=left.reconnect_attempts + right.reconnect_attempts,
        reconnect_successes=left.reconnect_successes + right.reconnect_successes,
    )


def run_whisperlive_session(
    config: WhisperLiveSessionConfig | None = None,
    *,
    on_result: TranscriptCallback | None = None,
    stop_event: threading.Event | None = None,
    on_log: LogCallback | None = None,
    on_merged_entry: MergedEntryCallback | None = None,
    install_signal_handlers: bool = True,
) -> WhisperLiveSessionStats:
    """Capture mic/speaker audio dan stream setiap source ke WhisperLive.

    Args:
        config: Konfigurasi sesi.
        on_result: Callback setiap ada hasil transcript yang sudah di-emit.
        stop_event: Event eksternal untuk menghentikan sesi.
        on_log: Callback log/status untuk UI; None berarti print ke stdout.
        on_merged_entry: Callback entry hasil merge untuk GUI.
        install_signal_handlers: False saat berjalan di thread GUI/QThread.
    """

    cfg = config or WhisperLiveSessionConfig()
    stats = WhisperLiveSessionStats()
    _stop = stop_event or threading.Event()
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(maxsize=cfg.max_chunk_queue_size)
    reader_threads: list[threading.Thread] = []
    capture_streams: list[object] = []

    def _print(text: str) -> None:
        if on_log is not None:
            on_log(text)
        else:
            print(text, flush=True)

    transcript_log: TranscriptLog | None = None
    if cfg.log_transcript:
        transcript_log = TranscriptLog(cfg.log_transcript)
        transcript_log.save()
    transcript_merger = TranscriptMerger()
    transcript_lock = threading.Lock()
    candidate_transcript_keys: set[tuple[str, int, str]] = set()
    partial_preview = _PartialTranscriptPreview() if cfg.show_partials else None
    chunk_archive = _ChunkArchive(cfg.chunk_archive_dir) if cfg.chunk_archive_dir is not None else None
    log_process_event(
        "client.session_start",
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
        mic_client_vad=cfg.mic_client_vad,
        speaker_client_vad=cfg.speaker_client_vad,
        auto_reconnect=cfg.auto_reconnect,
        reconnect_initial_backoff_seconds=cfg.reconnect_initial_backoff_seconds,
        reconnect_max_backoff_seconds=cfg.reconnect_max_backoff_seconds,
        reconnect_buffer_seconds=cfg.reconnect_buffer_seconds,
    )

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
            _print(f"[{label}] options sent audio_format={message.get('audio_format')}")
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
            if not event.completed and transcript_log is not None and event.result.text.strip():
                # Candidate disimpan sekali per window agar transcript audit
                # tidak kosong ketika TVE menahan hasil sebagai belum final.
                candidate_key = (
                    event.result.source,
                    int(event.result.start_seconds // 3),
                    " ".join(event.result.text.lower().split())[:180],
                )
                if candidate_key not in candidate_transcript_keys:
                    candidate_transcript_keys.add(candidate_key)
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
    reconnect_policy = ReconnectPolicy(
        enabled=cfg.auto_reconnect,
        initial_backoff_seconds=cfg.reconnect_initial_backoff_seconds,
        max_backoff_seconds=cfg.reconnect_max_backoff_seconds,
        buffer_seconds=cfg.reconnect_buffer_seconds,
    )
    clients = {
        source: _BufferedReconnectClient(
            source,
            connection_config,
            policy=reconnect_policy,
            on_transcript=handle_transcript,
            on_status=handle_status,
        )
        for source in active_sources
    }

    try:
        _connect_clients_parallel(clients, allow_initial_failure=cfg.auto_reconnect)

        capture_streams, reader_threads = _build_capture_streams(cfg, chunk_queue, _stop, stats)
        if not capture_streams:
            _print("[ERROR] Tidak ada audio stream yang tersedia. Periksa perangkat audio Anda.")
            return stats

        active_streams: list[object] = []
        for stream in capture_streams:
            try:
                stream.start()  # type: ignore[attr-defined]
                active_streams.append(stream)
            except CaptureError as exc:
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

        session_start = monotonic()
        _print("=" * 60)
        _print("Live transcription aktif via WhisperLive. Tekan Ctrl+C untuk berhenti.")
        _print(
            f"Server: {cfg.server_host}:{cfg.server_port}  |  "
            f"Model: {cfg.profile.model}  |  VAD: {cfg.profile.vad_threshold:.2f}"
        )
        _print(f"Sumber: {', '.join(source.upper() for source in active_sources)}")
        _print("=" * 60)

        try:
            while not _stop.is_set():
                # Loop utama hanya mengambil chunk yang sudah lolos preprocessing
                # dan mengirimnya ke koneksi WebSocket sesuai source.
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
                    outcome = client.send(chunk)
                    stats.chunks_sent += outcome.sent
                    stats.chunks_dropped += outcome.dropped
                    stats.chunks_buffered += outcome.buffered
                    stats.reconnect_attempts += outcome.reconnect_attempts
                    stats.reconnect_successes += outcome.reconnect_successes
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
                    stats.chunks_dropped += 1
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

            drained = _drain_pending_chunks(chunk_queue, clients, stats, chunk_archive)
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
                stats.results_received += 1

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
            _print("=" * 60)
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

    return stats


def _build_capture_streams(
    cfg: WhisperLiveSessionConfig,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    stop_event: threading.Event,
    stats: WhisperLiveSessionStats,
) -> tuple[list[object], list[threading.Thread]]:
    """Bangun stream capture mic/speaker beserta reader thread-nya."""
    capture_streams: list[object] = []
    reader_threads: list[threading.Thread] = []
    mic_preprocess_config = PreprocessConfig(
        chunk_seconds=cfg.chunk_seconds,
        min_chunk_seconds=cfg.chunk_seconds,
    )
    if not cfg.mic_client_vad:
        mic_preprocess_config = _without_client_vad(mic_preprocess_config)
    speaker_preprocess_config = PreprocessConfig(
        chunk_seconds=cfg.chunk_seconds,
        min_chunk_seconds=cfg.chunk_seconds,
        vad_rms_threshold=0.008,
        vad_peak_threshold=0.03,
        vad_speech_fraction=0.20,
        min_input_rms_db=-48.0,
    )
    if not cfg.speaker_client_vad:
        # Speaker loopback biasanya sudah bersih, jadi VAD client boleh dimatikan
        # agar audio pelan dari meeting tidak ikut terbuang.
        speaker_preprocess_config = _without_client_vad(speaker_preprocess_config)

    if cfg.source in ("mic", "both"):
        try:
            mic_stream = MicrophoneStream(
                MicrophoneConfig(
                    sample_rate=cfg.sample_rate,
                    block_size=cfg.block_size,
                    device=cfg.mic_device,
                    queue_size=cfg.queue_size,
                )
            )
            capture_streams.append(mic_stream)
            log_process_event(
                "client.capture_backend",
                source="mic",
                backend="microphone",
                sample_rate=cfg.sample_rate,
                block_size=cfg.block_size,
                queue_size=cfg.queue_size,
                device=cfg.mic_device,
                client_vad=cfg.mic_client_vad,
            )
            reader_threads.append(_reader("mic", mic_stream, mic_preprocess_config, chunk_queue, stop_event, stats))
        except Exception as exc:  # pragma: no cover
            log_process_event("client.capture_backend_error", source="mic", backend="microphone", error=str(exc))
            _log.warning("whisperlive session: mic unavailable: %s", exc)

    if cfg.source in ("speaker", "both"):
        backend_audio = get_audio_backend()
        try:
            if backend_audio is AudioBackend.WASAPI_LOOPBACK:
                speaker_stream = WindowsLoopbackStream(
                    WindowsLoopbackConfig(
                        sample_rate=cfg.sample_rate,
                        block_size=cfg.block_size,
                        device=cfg.speaker_device,
                        queue_size=cfg.queue_size,
                    )
                )
                backend = "wasapi_loopback"
            elif backend_audio is AudioBackend.SCREENCAPTUREKIT:
                speaker_stream = MacSystemAudioStream(
                    MacSystemAudioConfig(
                        sample_rate=cfg.sample_rate or 48_000,
                        block_size=cfg.block_size,
                        queue_size=cfg.queue_size,
                    )
                )
                backend = "mac_system_audio"
            else:
                raise CaptureNotSupportedError(f"speaker capture is unsupported on {backend_audio.value}")
            capture_streams.append(speaker_stream)
            log_process_event(
                "client.capture_backend",
                source="speaker",
                backend=backend,
                backenn_audio=backend_audio.value,
                sample_rate=cfg.sample_rate,
                block_size=cfg.block_size,
                queue_size=cfg.queue_size,
                device=cfg.speaker_device,
                client_vad=cfg.speaker_client_vad,
            )
            reader_threads.append(_reader("speaker", speaker_stream, speaker_preprocess_config, chunk_queue, stop_event, stats))
        except CaptureNotSupportedError as exc:
            log_process_event("client.capture_backend_error", source="speaker", backenn_audio=backend_audio.value, error=str(exc))
            _log.warning("whisperlive session: speaker unavailable: %s", exc)

    return capture_streams, reader_threads


def _without_client_vad(config: PreprocessConfig) -> PreprocessConfig:
    """Kembalikan konfigurasi preprocessing tanpa gate VAD client."""
    return PreprocessConfig(
        target_sample_rate=config.target_sample_rate,
        chunk_seconds=config.chunk_seconds,
        highpass_cutoff_hz=config.highpass_cutoff_hz,
        highpass_order=config.highpass_order,
        target_rms_db=config.target_rms_db,
        clip_limit=config.clip_limit,
        vad_rms_threshold=0.0,
        vad_peak_threshold=0.0,
        vad_speech_fraction=0.0,
        min_input_rms_db=float("-inf"),
        min_chunk_seconds=config.min_chunk_seconds,
        max_normalization_gain_db=config.max_normalization_gain_db,
    )


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
            stats.chunks_dropped += 1
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
            outcome = client.send(chunk)
            stats.chunks_sent += outcome.sent
            stats.chunks_dropped += outcome.dropped
            stats.chunks_buffered += outcome.buffered
            stats.reconnect_attempts += outcome.reconnect_attempts
            stats.reconnect_successes += outcome.reconnect_successes
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
            stats.chunks_dropped += 1
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
            stats.chunks_sent += outcome.sent
            stats.chunks_dropped += outcome.dropped
            stats.chunks_buffered += outcome.buffered
            stats.reconnect_attempts += outcome.reconnect_attempts
            stats.reconnect_successes += outcome.reconnect_successes
            total_sent += outcome.sent
            progressed = progressed or bool(outcome.sent or outcome.dropped or outcome.reconnect_successes)

        if sum(client.buffered_count for client in clients.values()) == 0:
            return total_sent
        if not progressed:
            threading.Event().wait(0.25)

    remaining = sum(client.buffered_count for client in clients.values())
    if remaining:
        stats.chunks_dropped += remaining
        log_process_event("client.reconnect_buffer_flush_timeout", remaining_chunks=remaining)
    return total_sent


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
    min_wait_deadline = perf_counter() + min(timeout_seconds, 5.0)
    last_results = stats.results_received
    quiet_since = perf_counter()
    while perf_counter() < deadline:
        threading.Event().wait(0.25)
        if stats.results_received != last_results:
            last_results = stats.results_received
            quiet_since = perf_counter()
        if perf_counter() >= min_wait_deadline and perf_counter() - quiet_since >= 1.5:
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
    reliability_score: float | None = None
    reliability_action: str | None = None


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
        events.append(
            WhisperLiveTranscriptEvent(
                result=result,
                completed=bool(segment.get("completed", True)),
                reliability_score=_optional_float(segment.get("reliability_score")),
                reliability_action=str(segment.get("reliability_action") or "") or None,
            )
        )
    return events


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit_merged_entry(
    entry,
    transcript_log: TranscriptLog | None,
    on_result: TranscriptCallback | None,
    *,
    on_log: LogCallback | None = None,
    on_merged_entry: MergedEntryCallback | None = None,
) -> None:
    if on_log is not None:
        on_log(entry.display)
    else:
        print(entry.display, flush=True)
    if transcript_log is not None:
        transcript_log.append_result(entry.result, label=entry.label, display=entry.display)
    if on_result is not None:
        on_result(entry.result)
    if on_merged_entry is not None:
        on_merged_entry(entry)


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
