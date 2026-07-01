"""Segment post-processing for live ASR hardening."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any


_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


MEDIA_HALLUCINATION_PHRASES = {
    "thanks for watching",
    "thank you for watching",
    "terima kasih sudah menonton",
    "terima kasih telah menonton",
    "terima kasih kerana menonton",
    "subtitles by",
    "subtitle by",
    "sampai jumpa di video berikutnya",
    "jangan lupa subscribe",
    "subscribe dan like",
    "jangan lupa like",
    "selamat datang di channel",
    "selamat datang di saluran",
    "pertahankan istilah teknis",
    "transkrip meeting bahasa indonesia",
    "technical terms api database deployment",
}

SHORT_SILENCE_PHRASES = {
    "thank you",
    "thanks",
    "terima kasih",
}


@dataclass(frozen=True, slots=True)
class SegmentFilterConfig:
    """Tunable server-side anti-hallucination policy."""

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


@dataclass
class SegmentPostProcessorFactory:
    """Factory that gives every client session its own filter state."""

    config: SegmentFilterConfig = field(default_factory=SegmentFilterConfig)

    def new_session(self) -> "SegmentStabilizer":
        return SegmentStabilizer(self.config)


class SegmentStabilizer:
    """Validate, score, hold, and normalize WhisperLive segment hypotheses."""

    def __init__(self, config: SegmentFilterConfig | None = None) -> None:
        self.config = config or SegmentFilterConfig()
        self._recent_texts: deque[str] = deque(maxlen=self.config.repeated_text_window)
        self._pending: deque[dict[str, Any]] = deque(maxlen=self.config.pending_window)
        self.filtered_count = 0
        self.pending_count = 0
        self.emitted_count = 0
        self._turn = 0

    def __call__(self, segment: dict[str, Any]) -> dict[str, Any] | None:
        self._turn += 1
        self._expire_pending()

        text = normalize_text(str(segment.get("text", "")))
        if not text:
            self.filtered_count += 1
            return None

        lowered = text.lower()
        completed = bool(segment.get("completed", False))

        if is_media_hallucination(lowered):
            self.filtered_count += 1
            return None

        if self._should_drop_repeated_short_phrase(lowered, completed):
            self.filtered_count += 1
            return None

        collapsed = collapse_repeated_words(text, self.config.max_consecutive_word_repeats)
        collapsed = collapse_repeated_sentences(collapsed)
        if is_degenerate_repetition(collapsed, self.config):
            self.filtered_count += 1
            return None

        normalized_collapsed = collapsed.lower()
        if completed and self._recent_texts.count(normalized_collapsed) >= self.config.max_repeated_text_count:
            self.filtered_count += 1
            return None

        evaluation = evaluate_segment(
            segment,
            collapsed,
            seen_before=self._pending_count(normalized_collapsed) > 0,
            emit_threshold=self.config.emit_threshold,
            review_threshold=self.config.review_threshold,
        )
        processed = dict(segment)
        processed["text"] = collapsed
        processed["reliability_score"] = round(evaluation.score, 3)
        processed["reliability_action"] = evaluation.action
        processed["reliability_factors"] = evaluation.factors

        if completed and evaluation.score < self.config.emit_threshold:
            self._hold_pending(processed, normalized_collapsed)
            return None

        if completed:
            self._recent_texts.append(normalized_collapsed)
            self._remove_pending(normalized_collapsed)
        self.emitted_count += 1
        return processed

    def _should_drop_repeated_short_phrase(self, lowered: str, completed: bool) -> bool:
        if lowered not in SHORT_SILENCE_PHRASES:
            return False
        if not self.config.drop_short_silence_phrases_after_repeat:
            return False
        return not completed or lowered in self._recent_texts

    def _hold_pending(self, segment: dict[str, Any], normalized_text: str) -> None:
        self.pending_count += 1
        self._remove_pending(normalized_text)
        held = dict(segment)
        held["_normalized_text"] = normalized_text
        held["_turn"] = self._turn
        self._pending.append(held)

    def _pending_count(self, normalized_text: str) -> int:
        return sum(1 for item in self._pending if item.get("_normalized_text") == normalized_text)

    def _remove_pending(self, normalized_text: str) -> None:
        self._pending = deque(
            [item for item in self._pending if item.get("_normalized_text") != normalized_text],
            maxlen=self.config.pending_window,
        )

    def _expire_pending(self) -> None:
        self._pending = deque(
            [
                item
                for item in self._pending
                if self._turn - int(item.get("_turn", self._turn)) <= self.config.max_pending_age
            ],
            maxlen=self.config.pending_window,
        )


@dataclass(frozen=True, slots=True)
class SegmentEvaluation:
    score: float
    action: str
    factors: dict[str, float]


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def evaluate_segment(
    segment: dict[str, Any],
    text: str,
    *,
    seen_before: bool = False,
    emit_threshold: float = 0.80,
    review_threshold: float = 0.70,
) -> SegmentEvaluation:
    factors = {
        "asr_confidence": score_asr_confidence(segment),
        "language": score_language_shape(text),
        "context": score_context_consistency(text, seen_before=seen_before),
        "stability": score_stability(segment, seen_before=seen_before),
        "dictionary": score_dictionary_shape(text),
    }
    score = (
        factors["asr_confidence"] * 0.34
        + factors["language"] * 0.16
        + factors["context"] * 0.16
        + factors["stability"] * 0.24
        + factors["dictionary"] * 0.10
    )
    score = max(0.0, min(1.0, score))
    if score >= emit_threshold:
        action = "emit"
    elif score >= review_threshold:
        action = "review"
    else:
        action = "pending"
    return SegmentEvaluation(score=score, action=action, factors={key: round(value, 3) for key, value in factors.items()})


def score_asr_confidence(segment: dict[str, Any]) -> float:
    score = 0.86

    no_speech_prob = _optional_float(segment.get("no_speech_prob"))
    if no_speech_prob is not None:
        score -= max(0.0, no_speech_prob - 0.25) * 0.75

    avg_logprob = _optional_float(segment.get("avg_logprob"))
    if avg_logprob is not None:
        if avg_logprob >= -0.45:
            score += 0.08
        elif avg_logprob < -1.4:
            score -= 0.28
        elif avg_logprob < -1.0:
            score -= 0.16

    compression_ratio = _optional_float(segment.get("compression_ratio"))
    if compression_ratio is not None:
        if compression_ratio > 2.6:
            score -= 0.24
        elif compression_ratio > 2.2:
            score -= 0.12

    words = segment.get("words")
    if isinstance(words, list) and words:
        probabilities = [
            _optional_float(word.get("probability"))
            for word in words
            if isinstance(word, dict) and _optional_float(word.get("probability")) is not None
        ]
        if probabilities:
            avg_probability = sum(probabilities) / len(probabilities)
            score = (score * 0.75) + (avg_probability * 0.25)

    return max(0.0, min(1.0, score))


def score_stability(segment: dict[str, Any], *, seen_before: bool) -> float:
    if seen_before:
        return 1.0
    return 0.92 if bool(segment.get("completed", False)) else 0.35


def score_context_consistency(text: str, *, seen_before: bool) -> float:
    if seen_before:
        return 1.0
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    if len(tokens) <= 2:
        return 0.72
    if has_sentence_boundary(text):
        return 0.86
    return 0.78


def score_language_shape(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    ratio = len(alpha_tokens) / len(tokens)
    if ratio < 0.45:
        return 0.45
    if is_media_hallucination(text.lower()):
        return 0.0
    return 0.84 if len(tokens) >= 3 else 0.76


def score_dictionary_shape(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(token.lower() for token in tokens)) / len(tokens)
    if unique_ratio < 0.35:
        return 0.35
    if any(token.isupper() and len(token) > 1 for token in tokens):
        return 0.9
    return 0.82


def has_sentence_boundary(text: str) -> bool:
    return text.endswith((".", "?", "!", "।"))


def is_media_hallucination(lowered_text: str) -> bool:
    normalized = normalize_text(lowered_text)
    return any(phrase in normalized for phrase in MEDIA_HALLUCINATION_PHRASES)


def collapse_repeated_words(text: str, max_repeats: int = 2) -> str:
    words = text.split()
    if not words:
        return ""

    collapsed: list[str] = []
    last_key = ""
    repeat_count = 0
    for word in words:
        key = word.strip(".,!?;:()[]{}\"'").lower()
        if key and key == last_key:
            repeat_count += 1
        else:
            last_key = key
            repeat_count = 1
        if repeat_count <= max_repeats:
            collapsed.append(word)
    return " ".join(collapsed).strip()


def collapse_repeated_sentences(text: str) -> str:
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]
    if len(pieces) <= 1:
        return text

    output: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        key = piece.strip(" .!?").lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(piece)
    return " ".join(output).strip()


def is_degenerate_repetition(text: str, config: SegmentFilterConfig | None = None) -> bool:
    cfg = config or SegmentFilterConfig()
    tokens = [match.group(0).lower() for match in _WORD_RE.finditer(text)]
    if len(tokens) < 8:
        return False

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < cfg.min_unique_token_ratio:
        return True

    n = cfg.repeated_ngram_size
    if len(tokens) < n * 2:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for index in range(0, len(tokens) - n + 1):
        gram = tuple(tokens[index : index + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] > cfg.max_repeated_ngram_count:
            return True
    return False


def _tokens(text: str) -> list[str]:
    return [match.group(0) for match in _WORD_RE.finditer(text)]


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
