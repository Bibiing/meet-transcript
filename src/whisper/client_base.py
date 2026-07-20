from __future__ import annotations

import json
import logging
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from src.preprocessing.core import PreprocessedAudioChunk
from src.version import app_version

_log = logging.getLogger(__name__)


class OutdatedClientError(RuntimeError):
    """Server menolak client ini karena versinya di bawah minimum (W4)."""


class TlsVerificationError(RuntimeError):
    """Sertifikat server tidak lolos verifikasi TLS (W2).

    Permanen sampai sertifikat/kepercayaan diperbaiki: mencoba ulang hanya
    mengulang handshake yang sama. TIDAK ADA opsi menonaktifkan verifikasi —
    tanpa auth, TLS adalah satu-satunya pelindung isi rapat.
    """

from src.whisper.models import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_HOTWORDS,
    WhisperLiveProfile,
    WhisperLiveConnectionConfig,
)


class WhisperLiveStreamClient:
    """Satu koneksi WebSocket WhisperLive untuk satu source audio."""

    END_OF_AUDIO = b"END_OF_AUDIO"

    def __init__(
        self,
        source: Literal["mic", "speaker"],
        config: WhisperLiveConnectionConfig | None = None,
        *,
        on_transcript: TranscriptCallback | None = None,
        on_status: StatusCallback | None = None,
        websocket_factory: WebSocketFactory | None = None,
        uid: str | None = None,
    ) -> None:
        self.source = source
        self.config = config or WhisperLiveConnectionConfig()
        self.uid = uid or f"{source}-{uuid.uuid4()}"
        self.on_transcript = on_transcript
        self.on_status = on_status
        self._websocket_factory = websocket_factory
        self._socket: Any | None = None
        self._recv_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        # W4: penolakan permanen dari server (mis. versi usang). Menunggu
        # SERVER_READY sampai ready_timeout habis tidak ada gunanya.
        self._fatal_error: str | None = None
        self._fatal_min_version: str = ""
        self._closed = False
        self._connected_at: float | None = None
        self._chunks_sent = 0
        self._bytes_sent = 0
        self._messages_received = 0
        self._segments_received = 0
        self._audio_finished = False

    @property
    def is_ready(self) -> bool:
        # Penolakan fatal membangunkan penunggu lewat _ready_event; stream itu
        # tetap TIDAK siap dipakai.
        return self._ready_event.is_set() and self._fatal_error is None

    def connect(self) -> None:
        """Buka WebSocket, kirim opsi, lalu tunggu SERVER_READY."""
        self._emit_status(
            "CLIENT_CONNECTING",
            url=self.config.url,
            connect_timeout=self.config.connect_timeout,
            ready_timeout=self.config.ready_timeout,
            model=self.config.profile.model,
            language=self.config.profile.language,
            vad_threshold=self.config.profile.vad_threshold,
        )
        headers = []
        if self.config.api_key:
            headers.append(f"Authorization: Bearer {self.config.api_key}")

        factory = self._websocket_factory or _default_websocket_factory()
        try:
            self._socket = factory(
                self.config.url,
                timeout=self.config.connect_timeout,
                header=headers,
            )
        except ssl.SSLError as exc:
            # W2: kegagalan TLS bukan gangguan jaringan biasa — ia permanen sampai
            # sertifikat/kepercayaan diperbaiki, dan pesannya harus dapat ditindaklanjuti.
            reason = _tls_failure_reason(exc)
            self._emit_status("CLIENT_TLS_ERROR", url=self.config.url, reason=reason, error=str(exc))
            raise TlsVerificationError(
                f"Secure connection to {self.config.url} failed: {reason}"
            ) from exc
        self._connected_at = perf_counter()
        self._emit_status("CLIENT_SOCKET_OPEN", url=self.config.url)
        self._socket.send(json.dumps(self._options()))
        self._emit_status("CLIENT_OPTIONS_SENT", audio_format=self.config.audio_format)
        if hasattr(self._socket, "settimeout"):
            self._socket.settimeout(None)
            _log.debug("whisperlive socket recv timeout disabled source=%s uid=%s", self.source, self.uid)

        self._stop_event.clear()
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name=f"whisperlive-{self.source}-recv",
            daemon=True,
        )
        self._recv_thread.start()

        ready_deadline = perf_counter() + self.config.ready_timeout
        next_wait_log = perf_counter() + 10.0
        while not self._ready_event.is_set():
            remaining = ready_deadline - perf_counter()
            if remaining <= 0:
                break
            self._ready_event.wait(timeout=min(1.0, remaining))
            if self._ready_event.is_set():
                break
            now = perf_counter()
            if now >= next_wait_log:
                self._emit_status(
                    "CLIENT_WAITING_SERVER_READY",
                    waited_seconds=round(now - (self._connected_at or now), 1),
                    remaining_seconds=round(max(0.0, ready_deadline - now), 1),
                    **self._diagnostics(),
                )
                next_wait_log = now + 10.0

        if self._fatal_error is not None:
            # Gagal cepat: server sudah menolak koneksi ini secara permanen.
            self.close()
            raise OutdatedClientError(
                f"WhisperLive stream '{self.source}' rejected by server: {self._fatal_error} "
                f"(minimum version {self._fatal_min_version or 'unknown'})"
            )

        if not self._ready_event.is_set():
            self.close()
            self._emit_status("CLIENT_READY_TIMEOUT", ready_timeout=self.config.ready_timeout)
            raise TimeoutError(f"WhisperLive stream '{self.source}' was not ready within {self.config.ready_timeout:g}s")

    def send_chunk(self, chunk: PreprocessedAudioChunk) -> None:
        """Encode satu chunk audio 16 kHz mono dan kirim sebagai binary frame."""
        if self._socket is None or self._closed:
            raise RuntimeError(f"WhisperLive stream '{self.source}' is not connected")
        if chunk.sample_rate != self.config.sample_rate:
            raise ValueError(f"WhisperLive expects {self.config.sample_rate} Hz chunks, got {chunk.sample_rate}")

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("WhisperLive chunks must be mono 1D float32 samples")
        samples = np.clip(samples, -1.0, 1.0)
        payload = _encode_audio_payload(samples, self.config.audio_format)
        self._socket.send_binary(payload)
        self._chunks_sent += 1
        self._bytes_sent += len(payload)
        _log.debug(
            "whisperlive chunk sent source=%s uid=%s chunk=%s duration=%.3f rms=%.2f bytes=%s",
            self.source,
            self.uid,
            self._chunks_sent,
            chunk.duration_seconds,
            chunk.rms_db,
            len(payload),
        )

    def finish_audio(self) -> None:
        """Kirim marker END_OF_AUDIO supaya server melakukan flush final."""
        if self._socket is None or self._closed or self._audio_finished:
            return
        try:
            self._socket.send_binary(self.END_OF_AUDIO)
            self._audio_finished = True
            self._emit_status("CLIENT_AUDIO_FINISHED", **self._diagnostics())
        except Exception as exc:
            _log.warning("whisperlive stream '%s' failed to send END_OF_AUDIO: %s", self.source, exc)
            self._emit_status("CLIENT_SEND_ERROR", error=str(exc))
    def close(self, *, send_end_of_audio: bool = True) -> None:
        was_closed = self._closed
        self._stop_event.set()
        if not was_closed:
            self._emit_status("CLIENT_CLOSING", **self._diagnostics())
        socket = self._socket
        if socket is not None and send_end_of_audio:
            self.finish_audio()
        self._closed = True
        if socket is not None:
            try:
                socket.close()
            except Exception as exc:
                _log.debug("whisperlive stream '%s' socket close error: %s", self.source, exc)
        thread = self._recv_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._socket = None
        if not was_closed:
            self._emit_status("CLIENT_CLOSED", **self._diagnostics())

    def _options(self) -> dict[str, Any]:
        """Susun payload opsi yang menjadi kontrak handshake client-server."""
        profile = self.config.profile
        return {
            "uid": self.uid,
            "language": profile.language,
            "task": profile.task,
            "model": profile.model,
            "use_vad": profile.use_vad,
            "vad_parameters": profile.vad_parameters(),
            "send_last_n_segments": profile.send_last_n_segments,
            "no_speech_thresh": profile.no_speech_thresh,
            "no_speech_threshold": profile.decode_no_speech_threshold,
            "log_prob_threshold": profile.log_prob_threshold,
            "compression_ratio_threshold": profile.compression_ratio_threshold,
            "condition_on_previous_text": profile.condition_on_previous_text,
            "repetition_penalty": profile.repetition_penalty,
            "no_repeat_ngram_size": profile.no_repeat_ngram_size,
            "hallucination_silence_threshold": profile.hallucination_silence_threshold,
            "temperature": profile.temperature,
            "beam_size": profile.beam_size,
            "clip_audio": profile.clip_audio,
            "same_output_threshold": profile.same_output_threshold,
            "word_timestamps": profile.word_timestamps,
            "local_agreement": profile.local_agreement,
            "local_agreement_window_seconds": profile.local_agreement_window_seconds,
            "local_agreement_hop_seconds": profile.local_agreement_hop_seconds,
            "local_agreement_trailing_guard_seconds": profile.local_agreement_trailing_guard_seconds,
            "local_agreement_retain_seconds": profile.local_agreement_retain_seconds,
            "dynamic_prompt": profile.dynamic_prompt,
            "dynamic_prompt_max_chars": profile.dynamic_prompt_max_chars,
            "speech_boundary_detection": profile.speech_boundary_detection,
            "speech_boundary_silence_seconds": profile.speech_boundary_silence_seconds,
            "speech_boundary_max_wait_seconds": profile.speech_boundary_max_wait_seconds,
            "initial_prompt": profile.initial_prompt,
            "hotwords": profile.hotwords,
            "source": self.source,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "audio_format": self.config.audio_format,
            # W4: server menegakkan versi minimum berdasarkan nilai ini. Server tanpa
            # kebijakan mengabaikannya (backward compatible).
            "client_version": app_version(),
        }

    def _recv_loop(self) -> None:
        """Baca message server sampai koneksi ditutup atau stop diminta."""
        assert self._socket is not None
        while not self._stop_event.is_set():
            try:
                raw_message = self._socket.recv()
            except Exception as exc:
                if not self._closed:
                    self._emit_status(
                        "CLIENT_RECV_ERROR",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                        **self._diagnostics(),
                    )
                return

            if not raw_message:
                if not self._closed:
                    self._emit_status("CLIENT_REMOTE_CLOSED", **self._diagnostics())
                return

            try:
                message = json.loads(raw_message)
            except (TypeError, json.JSONDecodeError):
                _log.warning("whisperlive invalid JSON source=%s uid=%s raw=%r", self.source, self.uid, raw_message)
                continue

            if message.get("uid") != self.uid:
                _log.debug(
                    "whisperlive ignored message for different uid source=%s expected=%s got=%s",
                    self.source,
                    self.uid,
                    message.get("uid"),
                )
                continue

            self._messages_received += 1
            if message.get("message") == "SERVER_READY":
                self._ready_event.set()
                self._emit_status("SERVER_READY", backend=message.get("backend"), **self._diagnostics())
                continue

            # W4: penolakan versi dinaikkan menjadi kode status tersendiri di sini —
            # tempat kontrak server di-parse — agar lapisan di atasnya tidak perlu
            # menggali `raw`. Penolakan ini permanen, bukan gangguan sementara.
            if message.get("code") == "OUTDATED_CLIENT":
                self._fatal_error = "OUTDATED_CLIENT"
                self._fatal_min_version = str(message.get("min_version") or "")
                # Bangunkan connect() SEBELUM emit status (pola yang sama dengan
                # SERVER_READY: penunggu tidak boleh tertahan callback status).
                self._ready_event.set()
                self._emit_status(
                    "OUTDATED_CLIENT",
                    min_version=str(message.get("min_version") or ""),
                    message=message.get("message"),
                    raw=message,
                    **self._diagnostics(),
                )
                continue

            if "status" in message:
                self._emit_status(
                    str(message.get("status") or "SERVER_STATUS"),
                    message=message.get("message"),
                    raw=message,
                    **self._diagnostics(),
                )
                continue

            if message.get("message"):
                self._emit_status(
                    str(message.get("message")),
                    raw=message,
                    **self._diagnostics(),
                )
                continue

            if "segments" in message and self.on_transcript is not None:
                segments = _tag_segments(self.source, message.get("segments") or [])
                self._segments_received += len(segments)
                _log.info(
                    "whisperlive segments received source=%s uid=%s count=%s total_segments=%s completed=%s",
                    self.source,
                    self.uid,
                    len(segments),
                    self._segments_received,
                    sum(1 for seg in segments if seg.get("completed")),
                )
                self.on_transcript(self.source, segments, message)

    def _emit_status(self, status: str, **fields: Any) -> None:
        message = {"uid": self.uid, "status": status, **fields}
        _log.info("whisperlive client status source=%s %s", self.source, message)
        if self.on_status is not None:
            self.on_status(self.source, message)

    def _diagnostics(self) -> dict[str, Any]:
        elapsed = None if self._connected_at is None else round(perf_counter() - self._connected_at, 3)
        return {
            "elapsed_seconds": elapsed,
            "chunks_sent": self._chunks_sent,
            "bytes_sent": self._bytes_sent,
            "messages_received": self._messages_received,
            "segments_received": self._segments_received,
        }


def _tag_segments(source: str, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for segment in segments:
        copied = dict(segment)
        copied.setdefault("source", source)
        tagged.append(copied)
    return tagged


def _encode_audio_payload(samples: np.ndarray, audio_format: Literal["float32", "int16", "uint8"]) -> bytes:
    """Konversi float32 internal menjadi format audio yang diminta server."""
    if audio_format == "float32":
        return np.ascontiguousarray(samples, dtype=np.float32).tobytes()
    if audio_format == "int16":
        pcm16 = np.round(samples * 32767.0).astype(np.int16)
        return np.ascontiguousarray(pcm16).tobytes()
    if audio_format == "uint8":
        pcm8 = np.round((samples + 1.0) * 127.5).astype(np.uint8)
        return np.ascontiguousarray(pcm8).tobytes()
    raise ValueError(f"unsupported WhisperLive audio format: {audio_format}")


def _tls_failure_reason(exc: ssl.SSLError) -> str:
    """Klasifikasikan kegagalan TLS menjadi sebab yang dapat ditindaklanjuti (W2)."""
    if isinstance(exc, ssl.SSLCertVerificationError):
        if exc.verify_code == 62 or "hostname mismatch" in str(exc).lower():
            return "certificate does not match the server name"
        if exc.verify_code in (18, 19, 20, 21):
            return "server certificate is not trusted by this computer"
        if "expired" in str(exc).lower():
            return "server certificate has expired"
        return f"server certificate verification failed ({exc.verify_message or exc.reason})"
    return f"TLS handshake failed ({exc.reason or exc})"


def _default_websocket_factory() -> WebSocketFactory:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "websocket-client is required for WhisperLive streaming. "
            "Install project dependencies or run `pip install websocket-client`."
        ) from exc
    return websocket.create_connection
