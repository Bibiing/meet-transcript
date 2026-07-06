"""Local agreement stabilizer untuk hipotesis Whisper streaming.

Whisper pada sliding window sering mengubah kata terakhir. Modul ini hanya
mem-finalkan prefix kata yang muncul sama pada dua window berturut-turut,
sedangkan sisa kata dikirim sebagai partial/candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class HypothesisWord:
    """Representasi kata beserta timestamp dan sinyal confidence ASR."""

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


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Hasil local agreement: completed, partial, dan batas waktu confirmed."""

    completed: list[dict]
    partial: dict | None
    confirmed_until: float


class LocalAgreementStabilizer:
    """Finalkan hanya prefix yang stabil pada dua window berurutan."""

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
        """Evaluasi window ASR baru dan tentukan bagian stable vs partial."""
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
                    **_confidence_fields(stable_words),
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
                **_confidence_fields(remaining_words),
            }

        self._previous_words = remaining_words
        return AgreementResult(completed, partial, self.confirmed_until)


def _words_from_segments(
    segments: Iterable[object],
    *,
    offset_seconds: float,
    no_speech_threshold: float,
) -> list[HypothesisWord]:
    """Ekstrak kata dari segment Whisper, memakai word timestamp jika tersedia."""
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
                    words.append(
                        HypothesisWord(
                            text=text,
                            start=start,
                            end=end,
                            no_speech_prob=_optional_float_attr(segment, "no_speech_prob"),
                            avg_logprob=_optional_float_attr(segment, "avg_logprob"),
                            compression_ratio=_optional_float_attr(segment, "compression_ratio"),
                        )
                    )
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
            words.append(
                HypothesisWord(
                    text=token,
                    start=word_start,
                    end=word_end,
                    no_speech_prob=_optional_float_attr(segment, "no_speech_prob"),
                    avg_logprob=_optional_float_attr(segment, "avg_logprob"),
                    compression_ratio=_optional_float_attr(segment, "compression_ratio"),
                )
            )
    return words


def _common_prefix_count(previous: list[HypothesisWord], current: list[HypothesisWord]) -> int:
    """Hitung jumlah kata awal yang sama antara window sebelumnya dan saat ini."""
    count = 0
    for left, right in zip(previous, current):
        if left.normalized != right.normalized:
            break
        count += 1
    return count


def _join_words(words: list[HypothesisWord]) -> str:
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    return re.sub(r"\s+([,.!?;:])", r"\1", text).strip()


def _confidence_fields(words: list[HypothesisWord]) -> dict[str, float]:
    no_speech_values = [word.no_speech_prob for word in words if word.no_speech_prob is not None]
    avg_logprob_values = [word.avg_logprob for word in words if word.avg_logprob is not None]
    compression_values = [word.compression_ratio for word in words if word.compression_ratio is not None]
    fields: dict[str, float] = {}
    if no_speech_values:
        fields["no_speech_prob"] = max(no_speech_values)
    if avg_logprob_values:
        fields["avg_logprob"] = sum(avg_logprob_values) / len(avg_logprob_values)
    if compression_values:
        fields["compression_ratio"] = max(compression_values)
    return fields


def _float_attr(obj: object, name: str, default: float) -> float:
    try:
        return float(getattr(obj, name, default))
    except (TypeError, ValueError):
        return default


def _optional_float_attr(obj: object, name: str) -> float | None:
    value = getattr(obj, name, None)
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
