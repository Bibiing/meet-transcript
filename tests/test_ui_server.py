from __future__ import annotations

import json
from pathlib import Path

from src.ui.server import UiOptions, archive_transcript, build_live_command, transcript_payload


def test_build_live_command_uses_python_module_and_transcript_log(tmp_path: Path) -> None:
    transcript_log = tmp_path / "meeting.json"
    command = build_live_command(
        UiOptions(
            host="localhost",
            port=9090,
            source="speaker",
            model="medium",
            chunk_seconds=0.5,
            mic_device=3,
            speaker_device="Headset",
            mic_client_vad=False,
            speaker_client_vad=False,
            vad_threshold=0.5,
            no_speech_thresh=0.45,
            local_agreement=True,
            local_agreement_window_seconds=15.0,
            local_agreement_hop_seconds=2.0,
            dynamic_prompt=True,
            transcript_log=transcript_log,
            hide_partials=True,
        )
    )

    assert "-m" in command
    assert "src.main" in command
    assert "--live" in command
    assert "--source" in command
    assert "speaker" in command
    assert "--whisper-model" in command
    assert "medium" in command
    assert "--live-chunk-seconds" in command
    assert "0.5" in command
    assert "--mic-device" in command
    assert "3" in command
    assert "--speaker-device" in command
    assert "Headset" in command
    assert "--no-mic-client-vad" in command
    assert "--no-speaker-client-vad" in command
    assert "--transcript-log" in command
    assert str(transcript_log) in command
    assert "--hide-partials" in command
    assert "--local-agreement" in command
    assert "--dynamic-prompt" in command
    assert "--speech-boundary-detection" in command
    assert "--speech-boundary-silence-seconds" in command
    assert "--local-agreement-window-seconds" in command


def test_transcript_payload_counts_sources(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text(
        json.dumps(
            [
                {"source": "mic", "text": "halo"},
                {"source": "speaker", "text": "selamat pagi"},
                {"source": "speaker", "text": "lanjut"},
            ]
        ),
        encoding="utf-8",
    )

    payload = transcript_payload(path)

    assert payload["count"] == 3
    assert payload["counts"]["mic"] == 1
    assert payload["counts"]["speaker"] == 2
    assert payload["exists"] is True


def test_transcript_payload_counts_jsonl_sources(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text(
        "\n".join(
            [
                json.dumps({"source": "mic", "text": "halo"}),
                json.dumps({"source": "speaker", "text": "selamat pagi"}),
            ]
        ),
        encoding="utf-8",
    )

    payload = transcript_payload(path)

    assert payload["count"] == 2
    assert payload["counts"]["mic"] == 1
    assert payload["counts"]["speaker"] == 1
    assert payload["error"] is None


def test_archive_transcript_moves_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "transcript_log.json"
    path.write_text("[]", encoding="utf-8")

    archived = archive_transcript(path)

    assert archived is not None
    assert archived.exists()
    assert not path.exists()
