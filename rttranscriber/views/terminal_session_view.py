from __future__ import annotations

from rttranscriber.viewmodels.realtime_session import SessionViewState


class CliRealtimeView:
    """View terminal sederhana yang hanya merender state dari ViewModel."""

    def render(self, state: SessionViewState) -> str:
        lines = [
            f"status={state.status}",
            state.diagnostics,
            f"processed_chunks={state.processed_chunk_count}",
            f"partial_text={state.partial_text or '<empty>'}",
            f"final_text={state.final_text or '<empty>'}",
            f"generated_chunks={len(state.created_files)}",
        ]
        if state.created_files:
            lines.append(f"chunk_file={state.created_files[0]}")
        if state.error_message:
            lines.append(f"error={state.error_message}")
        return "\n".join(lines)
