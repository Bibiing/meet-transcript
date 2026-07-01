"""Persistent realtime transcript log."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.engine.whisper import TranscriptionResult
from src.utils.formatter import format_transcript_line, result_to_line


@dataclass(slots=True)
class TranscriptLog:
    """Append-only transcript backup that is saved after every result."""

    path: Path
    entries: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> TranscriptLog:
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("transcript log must contain a JSON list")
        entries = [entry for entry in payload if isinstance(entry, dict)]
        return cls(path=path, entries=entries)

    def append_result(
        self,
        result: TranscriptionResult,
        *,
        label: str | None = None,
        display: str | None = None,
    ) -> dict[str, object]:
        line = result_to_line(result)
        entry: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": result.source,
            "label": label or line.label,
            "start_seconds": result.start_seconds,
            "end_seconds": result.end_seconds,
            "duration_seconds": result.duration_seconds,
            "text": result.text,
            "display": display or format_transcript_line(line),
            "model_name": result.model_name,
            "language": result.language,
            "segments": [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in result.segments
            ],
        }
        self.entries.append(entry)
        self.save()
        return entry

    def save(self) -> Path:
        target_path = self.path.resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.entries, indent=2, ensure_ascii=False)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        ) as tmp_file:
            tmp_file.write(payload)
            tmp_path = Path(tmp_file.name)

        try:
            os.replace(str(tmp_path), str(target_path))
        except PermissionError:
            target_path.write_text(payload, encoding="utf-8")
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return self.path
