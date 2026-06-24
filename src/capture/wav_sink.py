"""WAV writer utilities for phase 2 capture smoke tests."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from src.capture.audio_frame import AudioFrame


def write_frames_to_wav(path: str | Path, frames: list[AudioFrame]) -> Path:
    """Write captured float32 frames to 16-bit PCM WAV."""

    if not frames:
        raise ValueError("cannot write an empty WAV file")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]

    for frame in frames:
        if frame.sample_rate != first.sample_rate:
            raise ValueError("all frames must use the same sample rate")
        if frame.channels != first.channels:
            raise ValueError("all frames must use the same channel count")

    audio = np.concatenate([frame.samples for frame in frames], axis=0)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(first.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(first.sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return output_path
