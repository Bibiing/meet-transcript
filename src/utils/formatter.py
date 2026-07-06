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

# convert hasil transkripsi menjadi baris yang siap ditampilkan
def result_to_line(result: TranscriptionResult) -> TranscriptLine:

    return TranscriptLine(
        source=result.source,
        label=SOURCE_LABELS.get(result.source, result.source.upper()),
        start_seconds=result.start_seconds,
        end_seconds=result.end_seconds,
        text=result.text.strip(),
    )

# format hasil transkripsi chat style
def format_transcript_line(line: TranscriptLine) -> str:
    start = _format_timestamp(line.start_seconds)
    end = _format_timestamp(line.end_seconds)
    text = line.text if line.text else "<empty>"
    return f"[{start} - {end}] [{line.label}] {text}"

# format hasil transkripsi menjadi list string
def format_transcript_results(results: list[TranscriptionResult]) -> list[str]:
    lines = [result_to_line(result) for result in results if result.text.strip()]
    lines.sort(key=lambda item: (item.start_seconds, item.source))
    return [format_transcript_line(line) for line in lines]

# format timestamp menjadi string
def _format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
