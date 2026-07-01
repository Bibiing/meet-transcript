"""Local agreement stabilizer for streaming Whisper hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class HypothesisWord:
    text: str
    start: float
    end: float

    @property
    def normalized(self) -> str:
        tokens = _WORD_RE.findall(self.text.lower())
        return "".join(tokens)


@dataclass(frozen=True, slots=True)
class AgreementResult:
    completed: list[dict]
    partial: dict | None
    confirmed_until: float


class LocalAgreementStabilizer:
    """Finalize only the prefix that is stable across two consecutive windows."""

    def __init__(
        self,
        *,
        trailing_guard_seconds: float = 0.6,
        min_words: int = 1,
    ) -> None:
        self.trailing_guard_seconds = max(0.0, trailing_guard_seconds)
        self.min_words = max(1, min_words)
        self.confirmed_until = 0.0
        self._previous_words: list[HypothesisWord] = []

    def update(
        self,
        segments: Iterable[object],
        *,
        offset_seconds: float,
        window_duration_seconds: float,
        no_speech_threshold: float = 0.45,
    ) -> AgreementResult:
        words = [
            word
            for word in _words_from_segments(
                segments,
                offset_seconds=offset_seconds,
                no_speech_threshold=no_speech_threshold,
            )
            if word.end > self.confirmed_until + 0.05 and word.normalized
        ]

        if not words:
            self._previous_words = []
            return AgreementResult([], None, self.confirmed_until)

        common_count = _common_prefix_count(self._previous_words, words)
        stable_words = words[:common_count]
        max_stable_end = offset_seconds + window_duration_seconds - self.trailing_guard_seconds
        stable_words = [word for word in stable_words if word.end <= max_stable_end]

        completed: list[dict] = []
        if len(stable_words) >= self.min_words:
            text = _join_words(stable_words)
            end = stable_words[-1].end
            if text and end > self.confirmed_until + 0.05:
                segment = {
                    "start": stable_words[0].start,
                    "end": end,
                    "text": text,
                    "completed": True,
                }
                completed.append(segment)
                self.confirmed_until = end

        remaining_words = [word for word in words if word.end > self.confirmed_until + 0.05]
        partial = None
        if remaining_words:
            partial = {
                "start": remaining_words[0].start,
                "end": remaining_words[-1].end,
                "text": _join_words(remaining_words),
                "completed": False,
            }

        self._previous_words = remaining_words
        return AgreementResult(completed, partial, self.confirmed_until)


def _words_from_segments(
    segments: Iterable[object],
    *,
    offset_seconds: float,
    no_speech_threshold: float,
) -> list[HypothesisWord]:
    words: list[HypothesisWord] = []
    for segment in segments:
        if _float_attr(segment, "no_speech_prob", 0.0) > no_speech_threshold:
            continue
        segment_words = getattr(segment, "words", None)
        if segment_words:
            for word in segment_words:
                text = str(getattr(word, "word", "")).strip()
                if not text:
                    continue
                start = offset_seconds + _float_attr(word, "start", _float_attr(segment, "start", 0.0))
                end = offset_seconds + _float_attr(word, "end", _float_attr(segment, "end", start))
                if end > start:
                    words.append(HypothesisWord(text=text, start=start, end=end))
            continue

        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        tokens = text.split()
        if not tokens:
            continue
        start = offset_seconds + _float_attr(segment, "start", 0.0)
        end = offset_seconds + _float_attr(segment, "end", start)
        duration = max(0.01, end - start)
        step = duration / len(tokens)
        for index, token in enumerate(tokens):
            word_start = start + (index * step)
            word_end = start + ((index + 1) * step)
            words.append(HypothesisWord(text=token, start=word_start, end=word_end))
    return words


def _common_prefix_count(previous: list[HypothesisWord], current: list[HypothesisWord]) -> int:
    count = 0
    for left, right in zip(previous, current):
        if left.normalized != right.normalized:
            break
        count += 1
    return count


def _join_words(words: list[HypothesisWord]) -> str:
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    return re.sub(r"\s+([,.!?;:])", r"\1", text).strip()


def _float_attr(obj: object, name: str, default: float) -> float:
    try:
        return float(getattr(obj, name, default))
    except (TypeError, ValueError):
        return default
