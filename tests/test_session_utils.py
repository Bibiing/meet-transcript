from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from src.preprocessing.core import PreprocessedAudioChunk
from src.whisper.models import TranscriptionResult
from src.whisper.session_utils import (
    TranscriptQualityTracker,
    WhisperLiveTranscriptEvent,
    _RollingAudioArchive,
)


def _chunk(source: str, start_seconds: float, duration_seconds: float = 0.5) -> PreprocessedAudioChunk:
    sample_rate = 16_000
    samples = np.ones(int(sample_rate * duration_seconds), dtype=np.float32) * 0.05
    return PreprocessedAudioChunk(
        source=source,
        samples=samples,
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        rms_db=-20.0,
        input_rms_db=-30.0,
    )


def _event(source: str, text: str, *, completed: bool) -> WhisperLiveTranscriptEvent:
    return WhisperLiveTranscriptEvent(
        result=TranscriptionResult(
            source=source,
            text=text,
            model_name="small",
            language="id",
            start_seconds=0.0,
            duration_seconds=1.0,
        ),
        completed=completed,
    )


def test_transcript_quality_tracker_summarizes_candidate_and_stable_counts() -> None:
    tracker = TranscriptQualityTracker()

    tracker.observe_event(_event("speaker", "draft pendek", completed=False))
    tracker.observe_event(_event("speaker", "hasil final lengkap", completed=True))
    tracker.observe_event(_event("mic", "ok", completed=True))
    tracker.observe_emit("speaker")

    summary = tracker.summary()

    assert summary["candidate"] == 1
    assert summary["stable"] == 2
    assert summary["stable_ratio"] == 0.667
    by_source = summary["by_source"]
    assert by_source["speaker"]["candidate"] == 1
    assert by_source["speaker"]["stable_emitted"] == 1
    assert by_source["mic"]["short_stable"] == 1


def test_rolling_audio_archive_writes_segments_and_metadata(tmp_path: Path) -> None:
    archive = _RollingAudioArchive(
        tmp_path,
        session_id="session-1",
        segment_seconds=1.0,
        metadata={"model": "small"},
    )

    assert archive.save(_chunk("speaker", 0.0)) == []
    paths = archive.save(_chunk("speaker", 0.5))
    archive.close()

    assert len(paths) == 1
    assert paths[0].exists()
    with wave.open(str(paths[0]), "rb") as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 16_000

    metadata = json.loads((tmp_path / "session-1" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model"] == "small"
    assert metadata["segments"][0]["source"] == "speaker"
    assert metadata["segments"][0]["chunk_count"] == 2
