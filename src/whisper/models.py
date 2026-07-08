from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
from time import perf_counter
from typing import Literal
from uuid import uuid4

from src.utils.formatter import format_timestamp


DEFAULT_READY_TIMEOUT_SECONDS = 300.0
DEFAULT_INITIAL_PROMPT = ""
DEFAULT_HOTWORDS = ""

def _new_session_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:8]}"

@dataclass(frozen=True, slots=True)
class WhisperLiveProfile:
    """Profil ASR yang dikirim ke WhisperLive saat stream tersambung."""

    language: str = "id"
    task: Literal["transcribe", "translate"] = "transcribe"
    model: str = "small"
    use_vad: bool = False
    vad_threshold: float = 0.55
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 500
    vad_speech_pad_ms: int = 200
    no_speech_thresh: float = 0.75
    decode_no_speech_threshold: float = 0.75
    log_prob_threshold: float = -1.2
    compression_ratio_threshold: float = 2.6
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
    local_agreement_window_seconds: float = 20.0
    local_agreement_hop_seconds: float = 3.0
    local_agreement_trailing_guard_seconds: float = 0.6
    local_agreement_retain_seconds: float = 1.0
    dynamic_prompt: bool = True
    dynamic_prompt_max_chars: int = 700
    speech_boundary_detection: bool = True
    speech_boundary_silence_seconds: float = 0.8
    speech_boundary_max_wait_seconds: float = 5.0
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
    audio_format: Literal["float32", "int16", "uint8"] = "int16"
    connect_timeout: float = 10.0
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    api_key: str | None = None
    profile: WhisperLiveProfile = field(default_factory=WhisperLiveProfile)

    @property
    def url(self) -> str:
        protocol = "wss" if self.use_wss else "ws"
        return f"{protocol}://{self.host}:{self.port}"


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
    
    # server-side VAD policy per source
    mic_server_vad: bool = True
    speaker_server_vad: bool = False

    # client preprocessing level policy
    mic_target_rms_db: float = -20.0
    mic_max_normalization_gain_db: float = 18.0
    mic_min_input_rms_db: float = -38.0
    speaker_target_rms_db: float = -23.0
    speaker_max_normalization_gain_db: float = 18.0
    max_chunk_queue_size: int = 32
    
    # koneksi dan reconnect
    auto_reconnect: bool = True
    reconnect_initial_backoff_seconds: float = 1.0
    reconnect_max_backoff_seconds: float = 30.0
    reconnect_buffer_seconds: float = 30.0
    final_drain_seconds: float = 10.0
    
    # output dan logging
    session_id: str = field(default_factory=_new_session_id)
    show_partials: bool = True
    log_transcript: Path | None = None
    resume_transcript_log: bool = False
    chunk_archive_dir: Path | None = None
    rolling_audio_archive_dir: Path | None = None
    rolling_audio_segment_seconds: float = 60.0
    process_log_include_hot_path: bool = False
    process_log_summary_interval_seconds: float = 5.0
    candidate_cache_max_entries: int = 2_000
    merger_emitted_cache_max_entries: int = 5_000
    
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
    transcript_summary: dict[str, object] = field(default_factory=dict)
    rolling_audio_dir: Path | None = None
    start_time: float = field(default_factory=perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    # waktu yang telah berlalu sejak sesi dimulai
    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.start_time

    def add_send_outcome(self, outcome: SendOutcome) -> None:
        with self._lock:
            self.chunks_sent += outcome.sent
            self.chunks_dropped += outcome.dropped
            self.chunks_buffered += outcome.buffered
            self.reconnect_attempts += outcome.reconnect_attempts
            self.reconnect_successes += outcome.reconnect_successes

    def add_chunks_dropped(self, count: int = 1) -> None:
        with self._lock:
            self.chunks_dropped += count

    def add_result_received(self, count: int = 1) -> None:
        with self._lock:
            self.results_received += count

    def set_transcript_summary(self, summary: dict[str, object]) -> None:
        with self._lock:
            self.transcript_summary = summary

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

@dataclass(frozen=True, slots=True)
class MergedTranscriptEntry:
    result: TranscriptionResult
    label: str

    @property
    def display(self) -> str:
        return (
            f"[{_format_timestamp(self.result.start_seconds)} - "
            f"{_format_timestamp(self.result.end_seconds)}] "
            f"[{self.label}] {self.result.text.strip()}"
        )


def _format_timestamp(seconds: float) -> str:
    return format_timestamp(seconds, include_millis=True)

@dataclass(frozen=True, slots=True)
class WhisperLiveReplayConfig:
    wav_path: Path
    source: Literal["mic", "speaker"] = "mic"
    server_host: str = "localhost"
    server_port: int = 9090
    use_wss: bool = False
    api_key: str | None = None
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    chunk_seconds: float = 0.5
    audio_format: Literal["float32", "int16", "uint8"] = "int16"
    realtime: bool = False
    profile: WhisperLiveProfile = field(default_factory=WhisperLiveProfile)

@dataclass(frozen=True, slots=True)
class WhisperLiveReplayResult:
    chunks_sent: int
    results_received: int


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    rejected_reason: str = ""


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    source: str
    text: str
    model_name: str
    language: str | None
    start_seconds: float
    duration_seconds: float
    segments: list[TranscriptionSegment] = field(default_factory=list)
    rejected_segments: list[TranscriptionSegment] = field(default_factory=list)
    warning: str = ""

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds

