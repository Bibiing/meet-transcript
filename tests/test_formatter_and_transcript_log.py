from __future__ import annotations

import json
from pathlib import Path

from src.engine.whisper import TranscriptionResult
from src.utils.formatter import format_transcript_results
from src.utils.transcript_log import TranscriptLog


def test_format_transcript_results_sorts_and_labels_sources() -> None:
    results = [
        TranscriptionResult(
            source="speaker",
            text="jawaban peserta",
            model_name="base",
            language="id",
            start_seconds=2.0,
            duration_seconds=1.0,
        ),
        TranscriptionResult(
            source="mic",
            text="pertanyaan pengguna",
            model_name="base",
            language="id",
            start_seconds=0.0,
            duration_seconds=1.0,
        ),
    ]

    lines = format_transcript_results(results)

    assert lines == [
        "[00:00 - 00:01] [MIC] pertanyaan pengguna",
        "[00:02 - 00:03] [SPEAKER] jawaban peserta",
    ]


def test_transcript_log_appends_and_saves_json() -> None:
    path = Path("tmp") / "phase5" / "transcript_log.json"
    log = TranscriptLog.load(path)
    log.entries.clear()

    entry = log.append_result(
        TranscriptionResult(
            source="mic",
            text="halo dunia",
            model_name="base",
            language="id",
            start_seconds=0.0,
            duration_seconds=1.0,
        )
    )

    assert entry["display"] == "[00:00 - 00:01] [MIC] halo dunia"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["source"] == "mic"
    assert payload[0]["text"] == "halo dunia"
