from rttranscriber.audio_models import AudioFormat, AudioFrame
from rttranscriber.services.audio_pipeline import AudioFrameNormalizer, TimestampedPcmRingBuffer


def test_normalize_and_chunk() -> None:
    normalizer = AudioFrameNormalizer()
    frame = AudioFrame(
        timestamp_seconds=0.0,
        frame_index=0,
        audio_format=AudioFormat(sample_rate=48000, channels=2, bits_per_sample=16),
        samples=[1000, -1000, 2000, -2000, 3000, -3000, 4000, -4000, 5000, -5000, 6000, -6000],
    )
    normalized = normalizer.normalize(frame)
    assert normalized.audio_format.sample_rate == 16000
    assert normalized.audio_format.channels == 1
    assert len(normalized.samples) == 2

    ring = TimestampedPcmRingBuffer(16000 * 10)
    mono_format = AudioFormat(sample_rate=16000, channels=1, bits_per_sample=16)
    for second in range(6):
        ring.push(
            AudioFrame(
                timestamp_seconds=float(second),
                frame_index=second * 16000,
                audio_format=mono_format,
                samples=[second] * 16000,
            )
        )
    chunk = ring.build_chunk(16000 * 6)
    assert chunk is not None
    assert len(chunk.samples) == 16000 * 6
