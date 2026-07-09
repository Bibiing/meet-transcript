from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
try:
    from faster_whisper.vad import VadOptions
except ModuleNotFoundError:
    @dataclass
    class VadOptions:  # type: ignore[no-redef]
        """Fallback used by lightweight tests when faster-whisper is not installed."""

        threshold: float = 0.5
        min_speech_duration_ms: int = 250
        max_speech_duration_s: float = float("inf")
        min_silence_duration_ms: int = 2000
        speech_pad_ms: int = 400

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)

# transcript
@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def _asdict(self):
        warn(
            "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float]

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class TranscriptionOptions:
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: List[float]
    initial_prompt: Optional[Union[str, Iterable[int]]]
    prefix: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    max_new_tokens: Optional[int]
    clip_timestamps: Union[str, List[float]]
    hallucination_silence_threshold: Optional[float]
    hotwords: Optional[str]


@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


# local agreement

# representasi kata beserta timestamp dan sinyal confidence ASR
@dataclass(frozen=True, slots=True)
class HypothesisWord:
    text: str
    start: float
    end: float
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None

    @property
    def normalized(self) -> str:
        tokens = _WORD_RE.findall(self.text.lower())
        return "".join(tokens)


# Hasil local agreement: completed, partial, dan batas waktu confirmed.
@dataclass(frozen=True, slots=True)
class AgreementResult:

    completed: list[dict]
    partial: dict | None
    confirmed_until: float


# postprocessing.py
# konfigurasi kebijakan anti-halusinasi
@dataclass(frozen=True, slots=True)
class SegmentFilterConfig:
    max_consecutive_word_repeats: int = 2
    repeated_text_window: int = 8
    max_repeated_text_count: int = 1
    min_unique_token_ratio: float = 0.35
    repeated_ngram_size: int = 3
    max_repeated_ngram_count: int = 2
    drop_short_silence_phrases_after_repeat: bool = True
    emit_threshold: float = 0.80
    review_threshold: float = 0.70
    pending_window: int = 12
    pending_repeat_boost: float = 0.18
    max_pending_age: int = 4
    short_phrase_min_duration_seconds: float = 2.0
    short_phrase_max_no_speech_prob: float = 0.35
    short_phrase_min_avg_logprob: float = -0.80

# factory untuk membuat segment filter
@dataclass
class SegmentPostProcessorFactory:
    config: SegmentFilterConfig = field(default_factory=SegmentFilterConfig)

    def new_session(self, client_uid: str | None = None) -> Any:
        from whisper_live.postprocessing import SegmentStabilizer
        return SegmentStabilizer(self.config, client_uid=client_uid)

# evaluasi
@dataclass(frozen=True, slots=True)
class SegmentEvaluation:
    score: float
    action: str
    factors: dict[str, float]


# batch_inference.py
# representasi request untuk batch inference
@dataclass
class BatchRequest:    
    audio: np.ndarray
    language: Optional[str] = None
    task: str = "transcribe"
    initial_prompt: Optional[str] = None
    use_vad: bool = True
    vad_parameters: Optional[Dict] = None
    word_timestamps: bool = False
    client_uid: Optional[str] = None
    temperature: float = 0.0
    beam_size: int = 5
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float = 2.2
    log_prob_threshold: float = -0.8
    no_speech_threshold: float = 0.6
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    hallucination_silence_threshold: float = 1.0
    # Signaling
    future: threading.Event = field(default_factory=threading.Event)
    # Results (filled by batch worker)
    result: Optional[Any] = None
    info: Optional[Any] = None
    error: Optional[Exception] = None
