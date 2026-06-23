from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TranscriptToken:
    text: str
    start_frame_index: int
    end_frame_index: int


@dataclass(slots=True)
class TranscriptionChunkResult:
    chunk_start_frame_index: int
    chunk_end_frame_index: int
    partial_text: str
    tokens: list[TranscriptToken] = field(default_factory=list)
    processing_seconds: float = 0.0


@dataclass(slots=True)
class TranscriptSnapshot:
    final_text: str
    partial_text: str
    committed_tokens: list[TranscriptToken] = field(default_factory=list)
    pending_tokens: list[TranscriptToken] = field(default_factory=list)
    processed_chunk_count: int = 0


@dataclass(slots=True)
class RealtimeSessionResult:
    diagnostics: str
    created_files: list[Path] = field(default_factory=list)
    snapshots: list[TranscriptSnapshot] = field(default_factory=list)
    final_snapshot: TranscriptSnapshot | None = None
