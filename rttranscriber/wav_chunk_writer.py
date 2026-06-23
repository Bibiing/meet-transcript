from __future__ import annotations

from pathlib import Path
import wave

from .audio_models import AudioChunk


class WavChunkSink:
    def __init__(self, output_directory: Path) -> None:
        self._output_directory = output_directory
        self._output_directory.mkdir(parents=True, exist_ok=True)
        self._sequence = 0

    def publish(self, chunk: AudioChunk) -> Path:
        output_path = self._output_directory / f"chunk_{self._sequence}.wav"
        self._sequence += 1

        # Modul wave cukup untuk Phase 2 dan lebih mudah dipahami daripada
        # menulis header WAV manual di Python.
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(chunk.audio_format.channels)
            wav_file.setsampwidth(chunk.audio_format.bits_per_sample // 8)
            wav_file.setframerate(chunk.audio_format.sample_rate)
            wav_file.writeframes(
                b"".join(int(sample).to_bytes(2, byteorder="little", signed=True) for sample in chunk.samples)
            )
        return output_path
