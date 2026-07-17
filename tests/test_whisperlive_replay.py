from __future__ import annotations

from pathlib import Path

import numpy as np

from src.capture.models import AudioFrame
from src.capture.wav_sink import write_frames_to_wav
from src.whisper.capture import _build_mic_preprocess_config, _build_speaker_preprocess_config
from src.whisper.models import WhisperLiveReplayConfig, WhisperLiveSessionConfig
from src.whisper.replay import _preprocess_config_for_replay, _read_wav_as_frame


def test_replay_dsp_config_identical_to_live_per_source() -> None:
    # Prasyarat validitas A/B: DSP replay HARUS identik dengan live per-source.
    for source, live_builder in (
        ("speaker", _build_speaker_preprocess_config),
        ("mic", _build_mic_preprocess_config),
    ):
        replay_cfg = WhisperLiveReplayConfig(wav_path=Path("x.wav"), source=source, chunk_seconds=0.5)
        got = _preprocess_config_for_replay(replay_cfg)
        expected = live_builder(WhisperLiveSessionConfig(chunk_seconds=0.5))
        assert got == expected, f"replay DSP menyimpang dari live untuk source={source}"

    # Nilai spesifik speaker (regresi terhadap default -20/24 yang lama keliru).
    speaker = _preprocess_config_for_replay(
        WhisperLiveReplayConfig(wav_path=Path("x.wav"), source="speaker", chunk_seconds=0.5)
    )
    assert speaker.target_rms_db == -23.0
    assert speaker.max_normalization_gain_db == 18.0
    assert speaker.chunk_seconds == 0.5
    assert speaker.noise_reduction_enabled is False


def test_replay_dsp_chunk_seconds_flows_from_config() -> None:
    cfg = _preprocess_config_for_replay(
        WhisperLiveReplayConfig(wav_path=Path("x.wav"), source="speaker", chunk_seconds=1.0)
    )
    assert cfg.chunk_seconds == 1.0
    assert cfg.min_chunk_seconds == 1.0


def test_read_wav_as_frame_preserves_source_and_shape() -> None:
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = (0.1 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
    path = Path("tmp") / "tests" / "replay.wav"
    write_frames_to_wav(
        path,
        [
            AudioFrame(
                source="mic",
                samples=samples,
                sample_rate=sample_rate,
                channels=1,
                timestamp_seconds=0.0,
            )
        ],
    )

    frame = _read_wav_as_frame(path, "speaker")

    assert frame.source == "speaker"
    assert frame.sample_rate == sample_rate
    assert frame.channels == 1
    assert frame.samples.shape == (sample_rate, 1)
    assert np.max(np.abs(frame.samples)) > 0.05
