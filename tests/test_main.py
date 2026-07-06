from __future__ import annotations

from pathlib import Path

from src.capture.recorder import CaptureResult
from src.engine.preprocess_runtime import PreprocessResult
from src.engine.whisper import TranscriptionResult
from src.main import main


def test_main_status_run(capsys) -> None:
    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Detected OS:" in output
    assert "Audio backend:" in output
    assert "Status run completed." in output


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


def test_main_preprocess_prints_phase3_results(monkeypatch, capsys) -> None:
    captured_args = {}

    def fake_preprocess_audio_dir(input_dir, output_dir):
        captured_args["input_dir"] = input_dir
        captured_args["output_dir"] = output_dir
        return [
            PreprocessResult(
                source="mic",
                input_path=Path("audio") / "mic.wav",
                output_path=Path("audio") / "mic.preprocessed.wav",
                chunk_count=1,
                duration_seconds=2.5,
            )
        ]

    monkeypatch.setattr("src.main.preprocess_audio_dir", fake_preprocess_audio_dir)

    exit_code = main(["--preprocess", "--output-dir", "audio"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured_args["input_dir"] == Path("audio")
    assert captured_args["output_dir"] == Path("audio")
    assert "[MIC] preprocessed=" in output


def test_main_preprocess_prints_vad_drop_as_skip(monkeypatch, capsys) -> None:
    def fake_preprocess_audio_dir(input_dir, output_dir):
        return [
            PreprocessResult(
                source="speaker",
                input_path=Path("audio") / "speaker.wav",
                output_path=None,
                chunk_count=0,
                duration_seconds=0.0,
                warning="no speech chunk passed VAD",
            )
        ]

    monkeypatch.setattr("src.main.preprocess_audio_dir", fake_preprocess_audio_dir)

    exit_code = main(["--preprocess"])

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "[SPEAKER] preprocess skipped: no speech chunk passed VAD" in output


def test_main_transcribe_prints_phase4_results(monkeypatch, capsys) -> None:
    captured = {}

    def fake_transcribe_preprocessed_audio_dir(input_dir, *, config, output_path, on_result):
        captured["input_dir"] = input_dir
        captured["model_name"] = config.model_name
        captured["output_path"] = output_path
        results = [
            TranscriptionResult(
                source="mic",
                text="halo dunia",
                model_name=config.model_name,
                language="id",
                start_seconds=0.0,
                duration_seconds=1.0,
            )
        ]
        for result in results:
            on_result(result)
        return results

    monkeypatch.setattr("src.main.transcribe_preprocessed_audio_dir", fake_transcribe_preprocessed_audio_dir)

    transcript_log = Path("tmp") / "phase5" / "main_transcript_log.json"
    exit_code = main(
        [
            "--transcribe",
            "--whisper-model",
            "small",
            "--whisper-language",
            "id",
            "--transcript-log",
            str(transcript_log),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured["input_dir"] == Path("audio")
    assert captured["model_name"] == "small"
    assert captured["output_path"] == Path("audio") / "transcript.phase4.json"
    assert "[00:00 - 00:01] [MIC] halo dunia" in output
    assert f"Transcript log: {transcript_log}" in output
    assert transcript_log.exists()
