"""QThread wrapper that runs the WhisperLive session in the background."""

from __future__ import annotations

import socket
import threading
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.engine.transcript_merger import MergedTranscriptEntry
from src.engine.whisperlive_client import WhisperLiveProfile
from src.engine.whisperlive_session import (
    WhisperLiveSessionConfig,
    WhisperLiveSessionStats,
    run_whisperlive_session,
)


class SessionWorker(QThread):
    """Runs a live WhisperLive session on a background thread.

    Signals are emitted back to the GUI thread for every transcript entry,
    log line, status change, and stats update.
    """

    # Signals ------------------------------------------------------------------
    transcript_received = pyqtSignal(dict)   # {source, label, start, end, text}
    log_line = pyqtSignal(str)               # one log/status line
    status_changed = pyqtSignal(str)         # "connecting" | "running" | "stopping" | "stopped"
    stats_updated = pyqtSignal(int, int, int, float)  # sent, dropped, results, elapsed
    error = pyqtSignal(str)                  # error message
    session_finished = pyqtSignal()          # emitted when run() returns

    def __init__(self, config: WhisperLiveSessionConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._stop_event = threading.Event()
        self._stats: WhisperLiveSessionStats | None = None

    # -- Public API ------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the session to stop gracefully."""
        self._stop_event.set()
        self.status_changed.emit("stopping")

    # -- QThread overrides -----------------------------------------------------

    def run(self) -> None:  # noqa: D401 – Qt expects ``run``
        self.status_changed.emit("connecting")
        try:
            stats = run_whisperlive_session(
                self._config,
                stop_event=self._stop_event,
                on_log=self._on_log,
                on_merged_entry=self._on_merged_entry,
                install_signal_handlers=False,
            )
            self._stats = stats
        except Exception as exc:
            self.error.emit(str(exc))
            self._on_log(f"[ERROR] {exc}")
        finally:
            self.status_changed.emit("stopped")
            self.session_finished.emit()

    # -- Callbacks (called from background threads) ----------------------------

    def _on_log(self, text: str) -> None:
        self.log_line.emit(text)
        # Detect when the session is actually running (banner printed).
        if "Live transcription aktif" in text:
            self.status_changed.emit("running")

    def _on_merged_entry(self, entry: MergedTranscriptEntry) -> None:
        result = entry.result
        self.transcript_received.emit({
            "source": result.source,
            "label": entry.label,
            "start_seconds": result.start_seconds,
            "end_seconds": result.end_seconds,
            "text": result.text,
        })


def check_server_health(host: str = "localhost", port: int = 9090) -> bool:
    """Quick TCP probe to see if the WhisperLive server is reachable."""
    try:
        with socket.create_connection((host, port), timeout=1.2):
            return True
    except OSError:
        return False
