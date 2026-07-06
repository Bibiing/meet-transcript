"""Post-processing segment untuk memperkuat kualitas live ASR.

Layer ini memperlakukan output Whisper sebagai hipotesis, bukan kebenaran final.
Setiap segment dinormalisasi, dicek halusinasi/repetisi, diberi reliability
score, lalu diputuskan: emit, review, pending, atau drop.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from whisper_live.process_logging import log_process_event, preview_text


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
    "jangan menambah informasi",
    "jangan mengganti makna",
    "jangan menebak kreatif",
    "audio tidak jelas",
    "pertahankan istilah teknis",
    "preserve technical terms",
    "do not add information",
    "do not change meaning",
    "do not guess",
}

SHORT_SILENCE_PHRASES = {
    "terima",
    "kasih",
    "thank you",
    "thanks",
    "terima kasih",
}


@dataclass(frozen=True, slots=True)
class SegmentFilterConfig:
    """Konfigurasi kebijakan anti-halusinasi di sisi server."""

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


@dataclass
class SegmentPostProcessorFactory:
    """Factory agar setiap sesi client punya state filter sendiri."""

    config: SegmentFilterConfig = field(default_factory=SegmentFilterConfig)

    def new_session(self, client_uid: str | None = None) -> "SegmentStabilizer":
        return SegmentStabilizer(self.config, client_uid=client_uid)


class SegmentStabilizer:
    """Validasi, scoring, pending, dan normalisasi hipotesis segment."""

    def __init__(self, config: SegmentFilterConfig | None = None, *, client_uid: str | None = None) -> None:
        self.config = config or SegmentFilterConfig()
        self.client_uid = client_uid
        self._recent_segment_keys: deque[str] = deque(maxlen=self.config.repeated_text_window)
        # Segment yang sudah pernah lolos review dicatat sebagai konteks. Jika
        # muncul lagi pada window berikutnya, stability/context score dapat naik
        # menjadi emit.
        self._reviewed_texts: deque[str] = deque(maxlen=self.config.pending_window)
        self._pending: deque[dict[str, Any]] = deque(maxlen=self.config.pending_window)
        self.filtered_count = 0
        self.pending_count = 0
        self.emitted_count = 0
        self._turn = 0

    def __call__(self, segment: dict[str, Any]) -> dict[str, Any] | None:
        """Proses satu segment; return None berarti segment ditahan/dibuang."""
        self._turn += 1
        self._expire_pending()

        text = normalize_text(str(segment.get("text", "")))
        if not text:
            self.filtered_count += 1
            self._log_drop(segment, "empty_text")
            return None

        lowered = text.lower()
        phrase_key = short_phrase_key(lowered)
        completed = bool(segment.get("completed", False))

        if is_media_hallucination(lowered):
            self.filtered_count += 1
            self._log_drop(segment, "media_hallucination", text)
            return None

        if self._should_drop_repeated_short_phrase(phrase_key, completed):
            self.filtered_count += 1
            self._log_drop(segment, "repeated_short_phrase", text)
            return None

        collapsed = collapse_repeated_words(text, self.config.max_consecutive_word_repeats)
        collapsed = collapse_repeated_sentences(collapsed)
        if is_degenerate_repetition(collapsed, self.config):
            self.filtered_count += 1
            self._log_drop(segment, "degenerate_repetition", collapsed)
            return None

        normalized_collapsed = collapsed.lower()
        segment_key = self._segment_key(normalized_collapsed, segment)
        if completed and self._recent_segment_keys.count(segment_key) >= self.config.max_repeated_text_count:
            self.filtered_count += 1
            self._log_drop(segment, "recent_duplicate", collapsed)
            return None

        if completed and short_phrase_key(normalized_collapsed) in SHORT_SILENCE_PHRASES:
            if not has_strong_short_phrase_evidence(segment, self.config):
                self.filtered_count += 1
                self._log_drop(segment, "short_phrase_low_evidence", collapsed)
                return None

        evaluation = evaluate_segment(
            segment,
            collapsed,
            seen_before=self._seen_before(normalized_collapsed),
            emit_threshold=self.config.emit_threshold,
            review_threshold=self.config.review_threshold,
        )
        processed = dict(segment)
        processed["text"] = collapsed
        processed["reliability_score"] = round(evaluation.score, 3)
        processed["reliability_action"] = evaluation.action
        processed["reliability_factors"] = evaluation.factors
        log_process_event(
            "server.tve_score",
            uid=self.client_uid,
            completed=completed,
            score=processed["reliability_score"],
            action=evaluation.action,
            factors=evaluation.factors,
            text=preview_text(collapsed),
            start=processed.get("start"),
            end=processed.get("end"),
        )

        if evaluation.score < self.config.review_threshold:
            if completed:
                self._hold_pending(processed, normalized_collapsed)
            else:
                self.pending_count += 1
            log_process_event(
                "server.tve_pending",
                uid=self.client_uid,
                completed=completed,
                score=processed["reliability_score"],
                action=evaluation.action,
                text=preview_text(collapsed),
                pending_count=self.pending_count,
            )
            return None

        if completed:
            self._recent_segment_keys.append(segment_key)
            self._remove_pending(normalized_collapsed)
            if evaluation.action == "review":
                self._reviewed_texts.append(normalized_collapsed)
        self.emitted_count += 1
        log_process_event(
            "server.tve_emit",
            uid=self.client_uid,
            completed=completed,
            score=processed["reliability_score"],
            action=evaluation.action,
            text=preview_text(collapsed),
            emitted_count=self.emitted_count,
        )
        return processed

    def _log_drop(self, segment: dict[str, Any], reason: str, text: str | None = None) -> None:
        log_process_event(
            "server.tve_drop",
            uid=self.client_uid,
            reason=reason,
            completed=bool(segment.get("completed", False)),
            text=preview_text(text if text is not None else segment.get("text")),
            filtered_count=self.filtered_count,
            start=segment.get("start"),
            end=segment.get("end"),
        )

    def _should_drop_repeated_short_phrase(self, phrase_key: str, completed: bool) -> bool:
        if phrase_key not in SHORT_SILENCE_PHRASES:
            return False
        if not self.config.drop_short_silence_phrases_after_repeat:
            return False
        return not completed

    def _hold_pending(self, segment: dict[str, Any], normalized_text: str) -> None:
        self.pending_count += 1
        self._remove_pending(normalized_text)
        held = dict(segment)
        held["_normalized_text"] = normalized_text
        held["_turn"] = self._turn
        self._pending.append(held)

    def _pending_count(self, normalized_text: str) -> int:
        return sum(1 for item in self._pending if item.get("_normalized_text") == normalized_text)

    def _seen_before(self, normalized_text: str) -> bool:
        return self._pending_count(normalized_text) > 0 or normalized_text in self._reviewed_texts

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

    @staticmethod
    def _segment_key(normalized_text: str, segment: dict[str, Any]) -> str:
        return (
            f"{normalized_text}|"
            f"{_rounded_time(segment.get('start'))}|"
            f"{_rounded_time(segment.get('end'))}"
        )


@dataclass(frozen=True, slots=True)
class SegmentEvaluation:
    score: float
    action: str
    factors: dict[str, float]


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _rounded_time(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "?"


def evaluate_segment(
    segment: dict[str, Any],
    text: str,
    *,
    seen_before: bool = False,
    emit_threshold: float = 0.80,
    review_threshold: float = 0.70,
) -> SegmentEvaluation:
    """Hitung reliability score dari beberapa indikator kualitas segment."""
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
    """Skor confidence dari no_speech_prob, avg_logprob, compression, dan words."""
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


def short_phrase_key(text: str) -> str:
    return " ".join(token.lower() for token in _tokens(text))


def has_strong_short_phrase_evidence(segment: dict[str, Any], config: SegmentFilterConfig) -> bool:
    """Validasi phrase pendek seperti 'terima kasih' agar tidak mudah halu."""
    duration = _optional_float(segment.get("end"))
    start = _optional_float(segment.get("start"))
    if duration is not None and start is not None:
        if duration - start < config.short_phrase_min_duration_seconds:
            return False

    no_speech_prob = _optional_float(segment.get("no_speech_prob"))
    if no_speech_prob is None or no_speech_prob > config.short_phrase_max_no_speech_prob:
        return False

    avg_logprob = _optional_float(segment.get("avg_logprob"))
    if avg_logprob is None or avg_logprob < config.short_phrase_min_avg_logprob:
        return False

    compression_ratio = _optional_float(segment.get("compression_ratio"))
    if compression_ratio is not None and compression_ratio > 2.2:
        return False

    return True


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
