from pathlib import Path

from rttranscriber.audio_models import AudioChunk, AudioFormat
from rttranscriber.wav_chunk_writer import WavChunkSink


def test_wav_chunk_sink(tmp_path: Path) -> None:
    sink = WavChunkSink(tmp_path)
    output = sink.publish(
        AudioChunk(
            start_seconds=0.0,
            end_seconds=1.0,
            start_frame_index=0,
            end_frame_index=16000,
            audio_format=AudioFormat(sample_rate=16000, channels=1, bits_per_sample=16),
            samples=[123] * 16000,
        )
    )
    assert output.exists()
    assert output.read_bytes()[:4] == b"RIFF"
