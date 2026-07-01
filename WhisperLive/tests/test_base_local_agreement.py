from __future__ import annotations

from dataclasses import dataclass

from whisper_live.backend.base import ServeClientBase


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


class FakeWebSocket:
    def send(self, message: str) -> None:
        self.message = message


class DummyClient(ServeClientBase):
    def transcribe_audio(self):
        return []

    def handle_transcription_output(self, result, duration):
        return None


def _segment(words: list[str]) -> Segment:
    word_objects = [Word(word=word, start=index * 0.5, end=(index + 1) * 0.5) for index, word in enumerate(words)]
    return Segment(text=" ".join(words), start=0.0, end=len(words) * 0.5, words=word_objects)


def test_base_local_agreement_updates_transcript_and_offset() -> None:
    client = DummyClient(
        "uid",
        FakeWebSocket(),
        local_agreement=True,
        local_agreement_trailing_guard_seconds=0.0,
        local_agreement_retain_seconds=0.5,
    )
    client.processing_offset = 0.0

    assert client.update_segments([_segment(["halo", "semua"])], 2.0) is not None
    partial = client.update_segments([_segment(["halo", "semua", "lanjut"])], 3.0)

    assert client.transcript[0]["text"] == "halo semua"
    assert client.transcript[0]["completed"] is True
    assert client.timestamp_offset == 0.5
    assert partial is not None
    assert partial["text"] == "lanjut"
