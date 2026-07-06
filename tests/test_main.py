from __future__ import annotations

from pathlib import Path

from src.capture.recorder import CaptureResult
from src.preprocessing.file_processing import PreprocessResult
from src.main import main


def test_main_status_run(capsys) -> None:
    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Detected OS:" in output
    assert "Audio backend:" in output
    assert "Status run completed." in output


def test_main_record_mode_prints_mic_and_speaker(monkeypatch, capsys) -> None:
    captured_options = {}

    def fake_run_capture(options):
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

    monkeypatch.setattr("src.main.run_capture", fake_run_capture)

    exit_code = main(["--mode", "record", "--seconds", "1", "--source", "both"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured_options["options"].output_dir == Path("audio")
    assert captured_options["options"].seconds == 1
    assert "[MIC] saved=audio\\mic.wav" in output or "[MIC] saved=audio/mic.wav" in output
    assert "[SPEAKER] warning: loopback unavailable" in output


def test_main_preprocess_mode_prints_results(monkeypatch, capsys) -> None:
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

    exit_code = main(["--mode", "preprocess", "--output-dir", "audio"])

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

    exit_code = main(["--mode", "preprocess"])

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "[SPEAKER] preprocess skipped: no speech chunk passed VAD" in output


def test_main_replay_file_mode_uses_replay_config(monkeypatch, capsys, tmp_path: Path) -> None:
    captured = {}
    wav_path = tmp_path / "speaker.wav"
    wav_path.write_bytes(b"placeholder")

    class FakeReplayResult:
        chunks_sent = 3
        results_received = 2

    def fake_replay(config):
        captured["config"] = config
        return FakeReplayResult()

    monkeypatch.setattr("src.main.replay_wav_to_whisperlive", fake_replay)

    exit_code = main(["--mode", "replay-file", "--replay-file", str(wav_path), "--replay-source", "speaker"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured["config"].wav_path == wav_path
    assert captured["config"].source == "speaker"
    assert "Replay selesai. Chunks terkirim: 3 | Results: 2" in output
