from pathlib import Path

from rttranscriber.model.audio import AudioChunk, AudioFormat, AudioFrame
from rttranscriber.services.audio_pipeline import SlidingWindowChunkScheduler
from rttranscriber.services.realtime_transcription import (
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionCoordinator,
    TranscriptAssembler,
)
from rttranscriber.services.transcription_engine import EnergyTokenTranscriptEngine
from rttranscriber.viewmodels.realtime_session import RealtimeSessionViewModel


class FakeAudioSource:
    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = frames[:]
        self._started = False

    def diagnostics(self) -> str:
        return "backend=FAKE\ninput_devices=1"

    def start(self) -> None:
        self._started = True

    def read_frame(self, timeout: float = 1.0) -> AudioFrame | None:
        if not self._started or not self._frames:
            return None
        return self._frames.pop(0)

    def stop(self) -> None:
        self._started = False


def _make_frame(second: int, amplitude: int = 2000) -> AudioFrame:
    return AudioFrame(
        timestamp_seconds=float(second),
        frame_index=second * 16000,
        audio_format=AudioFormat(sample_rate=16000, channels=1, bits_per_sample=16),
        samples=[amplitude] * 16000,
    )


def test_sliding_window_scheduler_produces_overlapping_chunks() -> None:
    scheduler = SlidingWindowChunkScheduler(window_seconds=6, hop_seconds=2, buffer_seconds=12)
    chunks = []
    for second in range(8):
        chunks.extend(scheduler.push(_make_frame(second)))

    assert len(chunks) == 2
    assert chunks[0].start_frame_index == 0
    assert chunks[0].end_frame_index == 16000 * 6
    assert chunks[1].start_frame_index == 16000 * 2
    assert chunks[1].end_frame_index == 16000 * 8


def test_transcript_assembler_commits_stable_prefix() -> None:
    engine = EnergyTokenTranscriptEngine(token_window_seconds=1.0)
    chunk_a = scheduler_chunk(start_second=0, end_second=6, amplitude=1800)
    chunk_b = scheduler_chunk(start_second=2, end_second=8, amplitude=1800)
    assembler = TranscriptAssembler(hop_samples=16000 * 2)

    first = assembler.apply(engine.transcribe(chunk_a))
    second = assembler.apply(engine.transcribe(chunk_b))

    assert first.partial_text
    assert second.final_text.split()[:2] == ["jelas", "jelas"]


def test_realtime_session_viewmodel_runs_with_fake_source(tmp_path: Path) -> None:
    source = FakeAudioSource([_make_frame(second, amplitude=2500) for second in range(8)])
    coordinator = RealtimeTranscriptionCoordinator(
        audio_source=source,
        transcript_engine=EnergyTokenTranscriptEngine(token_window_seconds=1.0),
        output_directory=tmp_path,
    )
    view_model = RealtimeSessionViewModel(coordinator)

    state = view_model.run(
        RealtimeTranscriptionConfig(
            capture_seconds=1,
            window_seconds=6,
            hop_seconds=2,
            write_debug_wav=True,
        )
    )

    assert state.status == "completed"
    assert "backend=FAKE" in state.diagnostics
    assert state.processed_chunk_count >= 1
    assert state.created_files
    assert state.final_text


def scheduler_chunk(start_second: int, end_second: int, amplitude: int) -> AudioChunk:
    start_frame = start_second * 16000
    end_frame = end_second * 16000
    return AudioChunk(
        start_seconds=float(start_second),
        end_seconds=float(end_second),
        start_frame_index=start_frame,
        end_frame_index=end_frame,
        audio_format=AudioFormat(sample_rate=16000, channels=1, bits_per_sample=16),
        samples=[amplitude] * (end_frame - start_frame),
    )
