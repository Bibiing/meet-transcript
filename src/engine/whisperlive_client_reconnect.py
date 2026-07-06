from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

from src.engine.preprocessing import PreprocessedAudioChunk
from src.engine.whisperlive_client import WhisperLiveConnectionConfig, WhisperLiveStreamClient
from src.utils.logging import log_process_event

_log = logging.getLogger(__name__)

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
            outcome = SendOutcome()
            
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
    """Menggabungkan dua SendOutcome."""
    return SendOutcome(
        sent=left.sent + right.sent,
        buffered=left.buffered + right.buffered,
        dropped=left.dropped + right.dropped,
        reconnect_attempts=left.reconnect_attempts + right.reconnect_attempts,
        reconnect_successes=left.reconnect_successes + right.reconnect_successes,
    )
