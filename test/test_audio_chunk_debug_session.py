from pathlib import Path
from rttranscriber.audio_chunk_debug_session import AudioChunkDebugConfig, AudioChunkDebugRunner


def test_audio_chunk_debug_runner_detects_backend() -> None:
    runner = AudioChunkDebugRunner(AudioChunkDebugConfig(capture_seconds=2), Path("artifacts/python_test_chunks"))
    diagnostics, _files = runner.run()
    assert "backend=" in diagnostics
