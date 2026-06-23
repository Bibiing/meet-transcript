from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rttranscriber.model.transcription import RealtimeSessionResult, TranscriptSnapshot
from rttranscriber.services.realtime_transcription import (
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionCoordinator,
)


class SessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class SessionViewState:
    status: SessionStatus = SessionStatus.IDLE
    diagnostics: str = ""
    final_text: str = ""
    partial_text: str = ""
    processed_chunk_count: int = 0
    created_files: list[Path] = field(default_factory=list)
    snapshots: list[TranscriptSnapshot] = field(default_factory=list)
    error_message: str = ""


class RealtimeSessionViewModel:
    """ViewModel yang memproyeksikan hasil use case ke state yang mudah dirender."""

    def __init__(self, coordinator: RealtimeTranscriptionCoordinator) -> None:
        self._coordinator = coordinator
        self._state = SessionViewState()

    @property
    def state(self) -> SessionViewState:
        return self._state

    def run(self, config: RealtimeTranscriptionConfig) -> SessionViewState:
        self._state = SessionViewState(status=SessionStatus.RUNNING)
        try:
            result = self._coordinator.run(config)
        except Exception as exc:
            self._state = SessionViewState(
                status=SessionStatus.FAILED,
                error_message=str(exc),
            )
            raise

        self._state = self._project_result(result)
        return self._state

    def _project_result(self, result: RealtimeSessionResult) -> SessionViewState:
        final_snapshot = result.final_snapshot or TranscriptSnapshot(final_text="", partial_text="")
        return SessionViewState(
            status=SessionStatus.COMPLETED,
            diagnostics=result.diagnostics,
            final_text=final_snapshot.final_text,
            partial_text=final_snapshot.partial_text,
            processed_chunk_count=final_snapshot.processed_chunk_count,
            created_files=result.created_files,
            snapshots=result.snapshots,
        )
