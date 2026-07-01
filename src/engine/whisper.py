"""OpenAI Whisper integration for phase 4 transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from collections import Counter
import json
import wave

import numpy as np

from src.engine.preprocessing import PreprocessedAudioChunk


ModelLoader = Callable[[str, str | None], Any]
ResultCallback = Callable[["TranscriptionResult"], None]


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    """Runtime options for local OpenAI Whisper."""

    model_name: str = "small"         # upgraded from 'base' -- better Indonesian accuracy
    language: str | None = "id"       # default to Indonesian -- RC#3 fix
    task: str = "transcribe"
    device: str | None = None
    fp16: bool | None = None
    temperature: float = 0.0
    max_prompt_chars: int = 512
    overlap_seconds: float = 0.5
    file_chunk_seconds: float = 10.0
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.60         # relaxed from 0.45 -- RC#4 fix
    logprob_threshold: float = -1.0           # relaxed from -0.80 -- RC#4 fix
    compression_ratio_threshold: float = 2.2
    hallucination_silence_threshold: float = 1.0
    max_segment_no_speech_prob: float = 0.60  # relaxed from 0.50 -- RC#4 fix
    min_segment_avg_logprob: float = -1.0     # relaxed from -0.80 -- RC#4 fix
    max_segment_compression_ratio: float = 2.2
    max_repetition_ratio: float = 0.55


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


class OpenAIWhisperTranscriber:
    """Lazy local Whisper transcriber with context and overlap buffering."""

    def __init__(
        self,
        config: WhisperConfig | None = None,
        *,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.config = config or WhisperConfig()
        self._validate_config()
        self._model_loader = model_loader or _load_openai_whisper_model
        self._model: Any | None = None
        self._context_by_source: dict[str, str] = {}
        self._tail_by_source: dict[str, np.ndarray] = {}

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def transcribe_chunk(self, chunk: PreprocessedAudioChunk) -> TranscriptionResult:
        """Transcribe one preprocessed mono 16 kHz chunk."""

        model = self._get_model()
        audio = self._with_overlap(chunk)
        prompt = self._context_by_source.get(chunk.source, "")[-self.config.max_prompt_chars :]

        kwargs: dict[str, Any] = {
            "language": self.config.language,
            "task": self.config.task,
            "temperature": self.config.temperature,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "no_speech_threshold": self.config.no_speech_threshold,
            "logprob_threshold": self.config.logprob_threshold,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "hallucination_silence_threshold": self.config.hallucination_silence_threshold,
        }
        if prompt:
            kwargs["initial_prompt"] = prompt
        if self.config.fp16 is not None:
            kwargs["fp16"] = self.config.fp16

        raw = model.transcribe(audio, **kwargs)
        parsed_segments = _parse_segments(raw.get("segments", []), chunk.start_seconds)
        accepted_segments, rejected_segments = _filter_segments(parsed_segments, self.config)
        text = " ".join(segment.text for segment in accepted_segments).strip()
        warning = ""
        if not text and rejected_segments:
            warning = "all whisper segments rejected by quality gate"
        elif _has_excessive_repetition(text, self.config.max_repetition_ratio):
            rejected_segments.extend(
                segment for segment in accepted_segments if segment not in rejected_segments
            )
            accepted_segments = []
            text = ""
            warning = "transcript rejected because text repetition indicates hallucination"

        if text:
            self._context_by_source[chunk.source] = _append_context(
                self._context_by_source.get(chunk.source, ""),
                text,
                self.config.max_prompt_chars,
            )

        return TranscriptionResult(
            source=chunk.source,
            text=text,
            model_name=self.config.model_name,
            language=raw.get("language", self.config.language),
            start_seconds=chunk.start_seconds,
            duration_seconds=chunk.duration_seconds,
            segments=accepted_segments,
            rejected_segments=rejected_segments,
            warning=warning,
        )

    def transcribe_chunks(self, chunks: Iterable[PreprocessedAudioChunk]) -> list[TranscriptionResult]:
        return [self.transcribe_chunk(chunk) for chunk in chunks]

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(self.config.model_name, self.config.device)
        return self._model

    def _with_overlap(self, chunk: PreprocessedAudioChunk) -> np.ndarray:
        samples = np.asarray(chunk.samples, dtype=np.float32)
        tail = self._tail_by_source.get(chunk.source)
        if tail is not None and tail.size:
            audio = np.concatenate([tail, samples]).astype(np.float32)
        else:
            audio = samples.astype(np.float32, copy=True)

        overlap_frames = int(round(self.config.overlap_seconds * chunk.sample_rate))
        if overlap_frames > 0:
            self._tail_by_source[chunk.source] = samples[-overlap_frames:].copy()
        return audio

    def _validate_config(self) -> None:
        if not self.config.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.config.task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")
        if self.config.max_prompt_chars < 0:
            raise ValueError("max_prompt_chars must be non-negative")
        if self.config.overlap_seconds < 0:
            raise ValueError("overlap_seconds must be non-negative")
        if self.config.file_chunk_seconds <= 0:
            raise ValueError("file_chunk_seconds must be positive")
        if not 0 <= self.config.no_speech_threshold <= 1:
            raise ValueError("no_speech_threshold must be within [0, 1]")
        if not 0 <= self.config.max_segment_no_speech_prob <= 1:
            raise ValueError("max_segment_no_speech_prob must be within [0, 1]")
        if self.config.max_segment_compression_ratio <= 0:
            raise ValueError("max_segment_compression_ratio must be positive")


def transcribe_preprocessed_audio_dir(
    input_dir: Path = Path("audio"),
    *,
    config: WhisperConfig | None = None,
    output_path: Path | None = None,
    transcriber: OpenAIWhisperTranscriber | None = None,
    on_result: ResultCallback | None = None,
) -> list[TranscriptionResult]:
    """Transcribe `*.preprocessed.wav` files produced by phase 3."""

    engine = transcriber or OpenAIWhisperTranscriber(config or WhisperConfig())
    results: list[TranscriptionResult] = []
    for source in ("mic", "speaker"):
        path = input_dir / f"{source}.preprocessed.wav"
        if not path.exists():
            continue
        chunks = _read_preprocessed_wav_chunks(
            path,
            source=source,
            chunk_seconds=engine.config.file_chunk_seconds,
        )
        for result in engine.transcribe_chunks(chunks):
            results.append(result)
            if on_result is not None and result.text:
                on_result(result)

    if output_path is not None:
        write_transcript_json(output_path, results)
    return results


def write_transcript_json(path: Path, results: list[TranscriptionResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "source": result.source,
            "text": result.text,
            "model_name": result.model_name,
            "language": result.language,
            "start_seconds": result.start_seconds,
            "end_seconds": result.end_seconds,
            "duration_seconds": result.duration_seconds,
            "warning": result.warning,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                }
                for segment in result.segments
            ],
            "rejected_segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                    "reason": segment.rejected_reason,
                }
                for segment in result.rejected_segments
            ],
        }
        for result in results
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_openai_whisper_model(model_name: str, device: str | None) -> Any:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run `uv add openai-whisper \"numba>=0.62\"` "
            "or install it into the current uv environment."
        ) from exc

    if device:
        return whisper.load_model(model_name, device=device)
    return whisper.load_model(model_name)


def _read_preprocessed_wav_chunks(
    path: Path,
    *,
    source: str,
    chunk_seconds: float,
) -> list[PreprocessedAudioChunk]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if channels != 1:
        raise ValueError(f"preprocessed WAV must be mono, got channels={channels}")
    if sample_rate != 16_000:
        raise ValueError(f"preprocessed WAV must be 16 kHz, got sample_rate={sample_rate}")
    if sample_width != 2:
        raise ValueError(f"preprocessed WAV must be 16-bit PCM, got sample_width={sample_width}")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    chunk_frames = max(1, int(round(chunk_seconds * sample_rate)))
    chunks: list[PreprocessedAudioChunk] = []
    for start in range(0, samples.shape[0], chunk_frames):
        chunk_samples = samples[start : start + chunk_frames]
        if chunk_samples.size == 0:
            continue
        rms_db = _rms_db(chunk_samples)
        chunks.append(
            PreprocessedAudioChunk(
                source=source,
                samples=chunk_samples,
                sample_rate=sample_rate,
                start_seconds=start / sample_rate,
                duration_seconds=chunk_samples.shape[0] / sample_rate,
                rms_db=rms_db,
                input_rms_db=rms_db,
            )
        )
    return chunks


def _parse_segments(raw_segments: Any, offset_seconds: float) -> list[TranscriptionSegment]:
    segments: list[TranscriptionSegment] = []
    for segment in raw_segments or []:
        start = float(segment.get("start", 0.0)) + offset_seconds
        end = float(segment.get("end", 0.0)) + offset_seconds
        text = str(segment.get("text", "")).strip()
        segments.append(
            TranscriptionSegment(
                start=start,
                end=end,
                text=text,
                avg_logprob=_optional_float(segment.get("avg_logprob")),
                no_speech_prob=_optional_float(segment.get("no_speech_prob")),
                compression_ratio=_optional_float(segment.get("compression_ratio")),
            )
        )
    return segments


def _filter_segments(
    segments: list[TranscriptionSegment],
    config: WhisperConfig,
) -> tuple[list[TranscriptionSegment], list[TranscriptionSegment]]:
    accepted: list[TranscriptionSegment] = []
    rejected: list[TranscriptionSegment] = []

    for segment in segments:
        reason = _rejection_reason(segment, config)
        if reason:
            rejected.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                    compression_ratio=segment.compression_ratio,
                    rejected_reason=reason,
                )
            )
            continue
        accepted.append(segment)

    return accepted, rejected


def _rejection_reason(segment: TranscriptionSegment, config: WhisperConfig) -> str:
    if not segment.text.strip():
        return "empty text"
    if segment.avg_logprob is not None and segment.avg_logprob < config.min_segment_avg_logprob:
        return f"avg_logprob {segment.avg_logprob:.2f} below {config.min_segment_avg_logprob:.2f}"
    if segment.no_speech_prob is not None and segment.no_speech_prob > config.max_segment_no_speech_prob:
        return f"no_speech_prob {segment.no_speech_prob:.2f} above {config.max_segment_no_speech_prob:.2f}"
    if segment.compression_ratio is not None and segment.compression_ratio > config.max_segment_compression_ratio:
        return f"compression_ratio {segment.compression_ratio:.2f} above {config.max_segment_compression_ratio:.2f}"
    if _is_common_hallucination(segment.text):
        return "common whisper hallucination phrase"
    return ""


def _has_excessive_repetition(text: str, max_repetition_ratio: float) -> bool:
    tokens = [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 10:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < max_repetition_ratio:
        return True
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return bool(trigrams and max(Counter(trigrams).values()) >= 3)


def _is_common_hallucination(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    # Exact and substring phrases that Whisper often hallucinates
    phrases = {
        # English
        "terima kasih telah menonton",
        "thanks for watching",
        "subtitles by",
        "subtitle by",
        "thank you for watching",
        # Indonesian YouTube/media hallucinations
        "terima kasih sudah menonton",
        "terima kasih kerana menonton",   # Malaysian variant observed in testing
        "sampai jumpa di video berikutnya",
        "subscribe dan like",
        "jangan lupa subscribe",
        "jangan lupa like",
        "semoga bermanfaat",
        "selamat datang di channel",
        "selamat datang di saluran",
        "ayo kita mulai",
        "musik",
        "[musik]",
        "(musik)",
        "latar belakang musik",
    }
    return any(phrase in normalized for phrase in phrases)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 1e-8:
        return float("-inf")
    return float(20 * np.log10(rms))


def _append_context(current: str, text: str, max_chars: int) -> str:
    if max_chars == 0:
        return ""
    combined = f"{current} {text}".strip()
    return combined[-max_chars:]
