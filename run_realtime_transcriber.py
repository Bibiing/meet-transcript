from __future__ import annotations

from pathlib import Path
import sys

from rttranscriber.pysoundio_microphone_source import PySoundIoCapture
from rttranscriber.services.realtime_transcription import (
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionCoordinator,
)
from rttranscriber.services.transcription_engine import EnergyTokenTranscriptEngine
from rttranscriber.viewmodels.realtime_session import RealtimeSessionViewModel
from rttranscriber.views.terminal_session_view import CliRealtimeView


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    coordinator = RealtimeTranscriptionCoordinator(
        audio_source=PySoundIoCapture(),
        transcript_engine=EnergyTokenTranscriptEngine(),
        output_directory=Path("artifacts/python_chunks_phase3"),
    )
    view_model = RealtimeSessionViewModel(coordinator)
    state = view_model.run(RealtimeTranscriptionConfig())
    print(CliRealtimeView().render(state))


if __name__ == "__main__":
    main()
