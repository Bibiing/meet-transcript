from __future__ import annotations

import json
from pathlib import Path

from src.engine.whisper import TranscriptionResult
from src.utils.formatter import format_transcript_results
from src.utils.logging import TranscriptLog


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


def test_transcript_log_appends_and_saves_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    log = TranscriptLog.load(path)

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
    payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert payload[0]["source"] == "mic"
    assert payload[0]["text"] == "halo dunia"
    assert payload[0]["completed"] is True
    assert payload[0]["stability"] == "stable"


def test_transcript_log_loads_legacy_json_list(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text(
        json.dumps(
            [
                {
                    "source": "mic",
                    "text": "format lama",
                    "display": "[00:00 - 00:01] [MIC] format lama",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log = TranscriptLog.load(path)

    assert len(log.entries) == 1
    assert log.entries[0]["text"] == "format lama"


def test_transcript_log_can_store_candidate_metadata(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    log = TranscriptLog(path)

    entry = log.append_result(
        TranscriptionResult(
            source="speaker",
            text="candidate text",
            model_name="small",
            language="id",
            start_seconds=1.0,
            duration_seconds=2.0,
        ),
        completed=False,
        stability="candidate",
        reliability_score=0.73,
        reliability_action="review",
    )

    assert entry["completed"] is False
    assert entry["stability"] == "candidate"
    assert entry["reliability_score"] == 0.73
    payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert payload[0]["reliability_action"] == "review"
