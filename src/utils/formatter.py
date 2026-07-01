"""Transcript formatting helpers for the CLI view."""

from __future__ import annotations

from dataclasses import dataclass

from src.engine.whisper import TranscriptionResult


SOURCE_LABELS = {
    "mic": "MIC",
    "speaker": "SPEAKER",
}


@dataclass(frozen=True, slots=True)
class TranscriptLine:
    source: str
    label: str
    start_seconds: float
    end_seconds: float
    text: str


def result_to_line(result: TranscriptionResult) -> TranscriptLine:
    """Convert one transcription result into a display-ready line."""

    return TranscriptLine(
        source=result.source,
        label=SOURCE_LABELS.get(result.source, result.source.upper()),
        start_seconds=result.start_seconds,
        end_seconds=result.end_seconds,
        text=result.text.strip(),
    )


def format_transcript_line(line: TranscriptLine) -> str:
    """Format one line as a compact chat-style transcript entry."""

    start = _format_timestamp(line.start_seconds)
    end = _format_timestamp(line.end_seconds)
    text = line.text if line.text else "<empty>"
    return f"[{start} - {end}] [{line.label}] {text}"


def format_transcript_results(results: list[TranscriptionResult]) -> list[str]:
    """Format transcription results sorted by timestamp then source."""

    lines = [result_to_line(result) for result in results if result.text.strip()]
    lines.sort(key=lambda item: (item.start_seconds, item.source))
    return [format_transcript_line(line) for line in lines]


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
