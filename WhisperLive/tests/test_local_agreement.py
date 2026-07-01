from __future__ import annotations

from dataclasses import dataclass

from whisper_live.local_agreement import LocalAgreementStabilizer


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]
    no_speech_prob: float = 0.0


def _segment(text: str, words: list[str], *, no_speech_prob: float = 0.0) -> Segment:
    word_objects = [Word(word=word, start=index * 0.4, end=(index + 1) * 0.4) for index, word in enumerate(words)]
    return Segment(text=text, start=0.0, end=len(words) * 0.4, words=word_objects, no_speech_prob=no_speech_prob)


def test_local_agreement_finalizes_repeated_prefix_only_on_second_window() -> None:
    stabilizer = LocalAgreementStabilizer(trailing_guard_seconds=0.0)

    first = stabilizer.update(
        [_segment("halo selamat pagi", ["halo", "selamat", "pagi"])],
        offset_seconds=0.0,
        window_duration_seconds=3.0,
    )
    second = stabilizer.update(
        [_segment("halo selamat pagi semua", ["halo", "selamat", "pagi", "semua"])],
        offset_seconds=0.0,
        window_duration_seconds=4.0,
    )

    assert first.completed == []
    assert first.partial is not None
    assert [segment["text"] for segment in second.completed] == ["halo selamat pagi"]
    assert second.partial is not None
    assert second.partial["text"] == "semua"


def test_local_agreement_ignores_high_no_speech_segments() -> None:
    stabilizer = LocalAgreementStabilizer(trailing_guard_seconds=0.0)

    result = stabilizer.update(
        [_segment("terima kasih", ["terima", "kasih"], no_speech_prob=0.95)],
        offset_seconds=0.0,
        window_duration_seconds=2.0,
        no_speech_threshold=0.6,
    )

    assert result.completed == []
    assert result.partial is None
