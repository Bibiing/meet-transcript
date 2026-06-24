from __future__ import annotations

from pathlib import Path

from src.capture.phase2_smoke import CaptureResult
from src.main import main


def test_main_status_smoke(capsys) -> None:
    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Detected OS:" in output
    assert "Audio backend:" in output
    assert "Smoke run completed." in output


def test_main_record_phase2_prints_mic_and_speaker(monkeypatch, capsys) -> None:
    captured_options = {}

    def fake_run_phase2_capture(options):
        captured_options["options"] = options
        return [
            CaptureResult(
                source="mic",
                path=Path("audio") / "mic.wav",
                frame_count=48_000,
                duration_seconds=1.0,
                sample_rate=48_000,
                channels=1,
            ),
            CaptureResult(source="speaker", warning="loopback unavailable"),
        ]

    monkeypatch.setattr("src.main.run_phase2_capture", fake_run_phase2_capture)

    exit_code = main(["--record", "--seconds", "1", "--source", "both"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured_options["options"].output_dir == Path("audio")
    assert captured_options["options"].seconds == 1
    assert "[MIC] saved=audio\\mic.wav" in output or "[MIC] saved=audio/mic.wav" in output
    assert "[SPEAKER] warning: loopback unavailable" in output
