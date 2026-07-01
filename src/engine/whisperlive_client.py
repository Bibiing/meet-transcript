"""WhisperLive WebSocket client primitives."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from src.engine.preprocessing import PreprocessedAudioChunk


_log = logging.getLogger(__name__)

DEFAULT_READY_TIMEOUT_SECONDS = 300.0

DEFAULT_INITIAL_PROMPT = (
    "Transkrip rapat Bahasa Indonesia. Gunakan ejaan baku Bahasa Indonesia "
    "(EYD/PUEBI) hanya untuk kata yang terdengar jelas. Jangan menambah "
    "informasi, jangan mengganti makna, dan jangan menebak kreatif saat audio "
    "tidak jelas atau hening. Pertahankan istilah teknis, singkatan, nama "
    "sistem, dan kata serapan yang umum apa adanya."
)


DEFAULT_HOTWORDS = (
    "API, database, deployment, endpoint, server, staging, production, "
    "dashboard, authentication, authorization, billing, invoice, PLN, EYD, "
    "PUEBI, meteran, token listrik, pelanggan, tagihan, daya, gardu, trafo, "
    "integrasi, migrasi, aplikasi, layanan, user, admin, login, logout, "
    "repository, branch, commit, Docker, GPU, CPU"
)


TranscriptCallback = Callable[[str, list[dict[str, Any]], dict[str, Any]], None]
StatusCallback = Callable[[str, dict[str, Any]], None]
WebSocketFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class WhisperLiveProfile:
    """ASR profile sent to WhisperLive when a stream connects."""

    language: str = "id"
    task: Literal["transcribe", "translate"] = "transcribe"
    model: str = "large-v3-turbo"
    use_vad: bool = True
    vad_threshold: float = 0.55
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 500
    vad_speech_pad_ms: int = 200
    no_speech_thresh: float = 0.45
    decode_no_speech_threshold: float = 0.6
    log_prob_threshold: float = -0.8
    compression_ratio_threshold: float = 2.2
    condition_on_previous_text: bool = False
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    hallucination_silence_threshold: float = 1.0
    temperature: float = 0.0
    beam_size: int = 5
    clip_audio: bool = True
    same_output_threshold: int = 3
    send_last_n_segments: int = 10
    word_timestamps: bool = True
    local_agreement: bool = True
    local_agreement_window_seconds: float = 15.0
    local_agreement_hop_seconds: float = 2.0
    local_agreement_trailing_guard_seconds: float = 0.6
    local_agreement_retain_seconds: float = 1.0
    dynamic_prompt: bool = True
    dynamic_prompt_max_chars: int = 700
    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    hotwords: str = DEFAULT_HOTWORDS

    def vad_parameters(self) -> dict[str, int | float]:
        return {
            "threshold": self.vad_threshold,
            "min_speech_duration_ms": self.vad_min_speech_duration_ms,
            "min_silence_duration_ms": self.vad_min_silence_duration_ms,
            "speech_pad_ms": self.vad_speech_pad_ms,
        }


@dataclass(frozen=True, slots=True)
class WhisperLiveConnectionConfig:
    host: str = "localhost"
    port: int = 9090
    use_wss: bool = False
    sample_rate: int = 16_000
    channels: int = 1
    audio_format: Literal["float32", "int16", "uint8"] = "float32"
    connect_timeout: float = 10.0
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    api_key: str | None = None
    profile: WhisperLiveProfile = field(default_factory=WhisperLiveProfile)

    @property
    def url(self) -> str:
        protocol = "wss" if self.use_wss else "ws"
        return f"{protocol}://{self.host}:{self.port}"


class WhisperLiveStreamClient:
    """One WhisperLive WebSocket connection for one audio source."""

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
        self._closed = False
        self._connected_at: float | None = None
        self._chunks_sent = 0
        self._bytes_sent = 0
        self._messages_received = 0
        self._segments_received = 0

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def connect(self) -> None:
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
        self._socket = factory(
            self.config.url,
            timeout=self.config.connect_timeout,
            header=headers,
        )
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

        if not self._ready_event.is_set():
            self.close()
            self._emit_status("CLIENT_READY_TIMEOUT", ready_timeout=self.config.ready_timeout)
            raise TimeoutError(f"WhisperLive stream '{self.source}' was not ready within {self.config.ready_timeout:g}s")

    def send_chunk(self, chunk: PreprocessedAudioChunk) -> None:
        if self._socket is None or self._closed:
            raise RuntimeError(f"WhisperLive stream '{self.source}' is not connected")
        if chunk.sample_rate != self.config.sample_rate:
            raise ValueError(f"WhisperLive expects {self.config.sample_rate} Hz chunks, got {chunk.sample_rate}")

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("WhisperLive chunks must be mono 1D float32 samples")
        samples = np.ascontiguousarray(np.clip(samples, -1.0, 1.0), dtype=np.float32)
        payload = samples.tobytes()
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

    def close(self) -> None:
        was_closed = self._closed
        self._closed = True
        self._stop_event.set()
        if not was_closed:
            self._emit_status("CLIENT_CLOSING", **self._diagnostics())
        socket = self._socket
        if socket is not None:
            try:
                socket.send_binary(self.END_OF_AUDIO)
            except Exception:
                pass
            try:
                socket.close()
            except Exception:
                pass
        thread = self._recv_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._socket = None
        if not was_closed:
            self._emit_status("CLIENT_CLOSED", **self._diagnostics())

    def _options(self) -> dict[str, Any]:
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
            "initial_prompt": profile.initial_prompt,
            "hotwords": profile.hotwords,
            "source": self.source,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "audio_format": self.config.audio_format,
        }

    def _recv_loop(self) -> None:
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


def _default_websocket_factory() -> WebSocketFactory:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "websocket-client is required for WhisperLive streaming. "
            "Install project dependencies or run `pip install websocket-client`."
        ) from exc
    return websocket.create_connection
