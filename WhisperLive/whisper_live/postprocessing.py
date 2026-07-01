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


@dataclass
class SegmentPostProcessorFactory:
    """Factory that gives every client session its own filter state."""

    config: SegmentFilterConfig = field(default_factory=SegmentFilterConfig)

    def new_session(self) -> "SegmentStabilizer":
        return SegmentStabilizer(self.config)


class SegmentStabilizer:
    """Normalize, collapse, and drop unstable WhisperLive segments."""

    def __init__(self, config: SegmentFilterConfig | None = None) -> None:
        self.config = config or SegmentFilterConfig()
        self._recent_texts: deque[str] = deque(maxlen=self.config.repeated_text_window)
        self.filtered_count = 0

    def __call__(self, segment: dict[str, Any]) -> dict[str, Any] | None:
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

        processed = dict(segment)
        processed["text"] = collapsed
        if completed:
            self._recent_texts.append(normalized_collapsed)
        return processed

    def _should_drop_repeated_short_phrase(self, lowered: str, completed: bool) -> bool:
        if lowered not in SHORT_SILENCE_PHRASES:
            return False
        if not self.config.drop_short_silence_phrases_after_repeat:
            return False
        return not completed or lowered in self._recent_texts


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


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
