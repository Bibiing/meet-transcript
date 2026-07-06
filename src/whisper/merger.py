"""Timestamp-based transcript merger for dual-stream live captions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.whisper.models import TranscriptionResult


MEETING_SOURCE_LABELS = {
    "mic": "Me",
    "speaker": "Meeting",
}


from src.whisper.models import MergedTranscriptEntry


@dataclass(slots=True)
class TranscriptMerger:
    """Merge mic/speaker transcript results into stable timestamp order."""

    source_labels: dict[str, str] = field(default_factory=lambda: dict(MEETING_SOURCE_LABELS))
    reorder_delay_seconds: float = 0.75
    max_emitted_keys: int = 5_000
    _pending: list[TranscriptionResult] = field(default_factory=list)
    _emitted_keys: set[tuple[str, float, float, str]] = field(default_factory=set)
    _emitted_order: deque[tuple[str, float, float, str]] = field(default_factory=deque)
    _max_seen_start: float = 0.0

    def add_result(self, result: TranscriptionResult, *, completed: bool = True) -> list[MergedTranscriptEntry]:
        """Add one result and return entries ready to emit."""

        if not completed or not result.text.strip():
            return []

        key = _result_key(result)
        if key in self._emitted_keys:
            return []

        self._remember_emitted_key(key)
        self._pending.append(result)
        self._max_seen_start = max(self._max_seen_start, result.start_seconds)
        return self.pop_ready()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def pop_ready(self) -> list[MergedTranscriptEntry]:
        watermark = self._max_seen_start - self.reorder_delay_seconds
        ready = [result for result in self._pending if result.start_seconds <= watermark]
        if not ready:
            return []

        ready_keys = {_result_key(result) for result in ready}
        self._pending = [result for result in self._pending if _result_key(result) not in ready_keys]
        ready.sort(key=lambda result: (result.start_seconds, result.end_seconds, result.source))
        return [self._entry(result) for result in ready]

    def flush(self) -> list[MergedTranscriptEntry]:
        ready = sorted(self._pending, key=lambda result: (result.start_seconds, result.end_seconds, result.source))
        self._pending = []
        return [self._entry(result) for result in ready]

    def _entry(self, result: TranscriptionResult) -> MergedTranscriptEntry:
        return MergedTranscriptEntry(
            result=result,
            label=self.source_labels.get(result.source, result.source.upper()),
        )

    def _remember_emitted_key(self, key: tuple[str, float, float, str]) -> None:
        self._emitted_keys.add(key)
        self._emitted_order.append(key)
        while len(self._emitted_order) > max(1, self.max_emitted_keys):
            oldest = self._emitted_order.popleft()
            self._emitted_keys.discard(oldest)


def _result_key(result: TranscriptionResult) -> tuple[str, float, float, str]:
    return (
        result.source,
        round(result.start_seconds, 3),
        round(result.end_seconds, 3),
        " ".join(result.text.lower().split()),
    )

