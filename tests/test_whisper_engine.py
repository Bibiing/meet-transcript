from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from src.capture.audio_frame import AudioFrame
from src.capture.wav_sink import write_frames_to_wav
from src.engine.preprocess_runtime import preprocess_wav_file
from src.engine.preprocessing import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk
from src.engine.transcription_worker import AsyncWhisperWorker
from src.engine.whisper import (
    OpenAIWhisperTranscriber,
    WhisperConfig,
    transcribe_preprocessed_audio_dir,
)


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": np.asarray(audio), "kwargs": kwargs})
        text = f"hello {len(self.calls)}"
        return {
            "text": text,
            "language": kwargs.get("language") or "en",
            "segments": [{"start": 0.0, "end": 0.5, "text": text}],
        }


class LowConfidenceWhisperModel:
    def transcribe(self, audio, **kwargs):
        return {
            "text": "terima kasih telah menonton",
            "language": "id",
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "terima kasih telah menonton",
                    "avg_logprob": -1.4,
                    "no_speech_prob": 0.2,
                    "compression_ratio": 0.8,
                }
            ],
        }


def _chunk(samples: np.ndarray, source: str = "mic") -> PreprocessedAudioChunk:
    return PreprocessedAudioChunk(
        source=source,
        samples=samples.astype(np.float32),
        sample_rate=16_000,
        start_seconds=0.0,
        duration_seconds=samples.shape[0] / 16_000,
        rms_db=-20.0,
    )


def test_whisper_transcriber_defaults_to_small_and_passes_language() -> None:
    model = FakeWhisperModel()
    loaded = {}

    def loader(model_name, device):
        loaded["model_name"] = model_name
        loaded["device"] = device
        return model

    transcriber = OpenAIWhisperTranscriber(
        WhisperConfig(language="id", fp16=False),
        model_loader=loader,
    )

    result = transcriber.transcribe_chunk(_chunk(np.ones(16_000, dtype=np.float32) * 0.1))

    assert loaded == {"model_name": "small", "device": None}  # default is now 'small'
    assert result.text == "hello 1"
    assert result.model_name == "small"
    assert model.calls[0]["kwargs"]["language"] == "id"
    assert model.calls[0]["kwargs"]["fp16"] is False


def test_whisper_transcriber_uses_custom_model_context_and_overlap() -> None:
    model = FakeWhisperModel()
    transcriber = OpenAIWhisperTranscriber(
        WhisperConfig(model_name="small", overlap_seconds=0.5),
        model_loader=lambda model_name, device: model,
    )

    first = _chunk(np.ones(16_000, dtype=np.float32) * 0.1)
    second = _chunk(np.ones(16_000, dtype=np.float32) * 0.2)
    transcriber.transcribe_chunk(first)
    second_result = transcriber.transcribe_chunk(second)

    assert second_result.model_name == "small"
    assert model.calls[1]["audio"].shape[0] == 24_000
    assert model.calls[1]["kwargs"]["initial_prompt"] == "hello 1"


def test_whisper_transcriber_rejects_low_confidence_hallucination() -> None:
    transcriber = OpenAIWhisperTranscriber(
        WhisperConfig(min_segment_avg_logprob=-1.0),
        model_loader=lambda model_name, device: LowConfidenceWhisperModel(),
    )

    result = transcriber.transcribe_chunk(_chunk(np.ones(16_000, dtype=np.float32) * 0.01))

    # avg_logprob=-1.4 is below the new threshold of -1.0, should still be rejected
    assert result.text == ""
    assert result.warning == "all whisper segments rejected by quality gate"
    assert result.rejected_segments
    assert "avg_logprob" in result.rejected_segments[0].rejected_reason


def test_transcribe_preprocessed_audio_dir_writes_json() -> None:
    input_dir = Path("tmp") / "phase4" / "audio"
    input_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    samples = np.sin(2 * np.pi * 440 * np.arange(sample_rate, dtype=np.float32) / sample_rate) * 0.1
    raw_path = input_dir / "mic.wav"
    preprocessed_path = input_dir / "mic.preprocessed.wav"
    write_frames_to_wav(
        raw_path,
        [
            AudioFrame(
                source="mic",
                samples=samples.reshape(-1, 1),
                sample_rate=sample_rate,
                channels=1,
                timestamp_seconds=0.0,
            )
        ],
    )
    preprocess_wav_file(
        raw_path,
        preprocessed_path,
        source="mic",
        preprocessor=AudioPreprocessor(PreprocessConfig(chunk_seconds=1.0)),
    )

    model = FakeWhisperModel()
    transcriber = OpenAIWhisperTranscriber(model_loader=lambda model_name, device: model)
    output_path = input_dir / "transcript.json"

    callbacks = []
    results = transcribe_preprocessed_audio_dir(
        input_dir,
        transcriber=transcriber,
        output_path=output_path,
        on_result=callbacks.append,
    )

    assert len(results) == 1
    assert callbacks == results
    assert results[0].text == "hello 1"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["source"] == "mic"
    assert payload[0]["text"] == "hello 1"


def test_async_whisper_worker_transcribes_one_chunk() -> None:
    model = FakeWhisperModel()
    transcriber = OpenAIWhisperTranscriber(model_loader=lambda model_name, device: model)

    async def run() -> None:
        input_queue: asyncio.Queue[PreprocessedAudioChunk] = asyncio.Queue()
        await input_queue.put(_chunk(np.ones(16_000, dtype=np.float32) * 0.1))
        worker = AsyncWhisperWorker(transcriber, input_queue)
        result = await worker.run_once()
        queued = await worker.read_result()
        assert result.text == "hello 1"
        assert queued.text == "hello 1"

    asyncio.run(run())
