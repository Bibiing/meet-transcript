from __future__ import annotations

from pathlib import Path

import numpy as np

from src.capture.audio_frame import AudioFrame
from src.capture.wav_sink import write_frames_to_wav
from src.engine.whisperlive_replay import _read_wav_as_frame


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
