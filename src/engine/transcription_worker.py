"""Async transcription worker for phase 4."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.engine.preprocessing import PreprocessedAudioChunk
from src.engine.whisper import OpenAIWhisperTranscriber, TranscriptionResult


@dataclass(slots=True)
class AsyncWhisperWorker:
    """Consume preprocessed chunks and publish Whisper transcription results."""

    transcriber: OpenAIWhisperTranscriber
    input_queue: asyncio.Queue[PreprocessedAudioChunk]
    max_output_size: int = 32
    output_queue: asyncio.Queue[TranscriptionResult] = field(init=False)

    def __post_init__(self) -> None:
        if self.max_output_size <= 0:
            raise ValueError("max_output_size must be positive")
        self.output_queue = asyncio.Queue(maxsize=self.max_output_size)

    async def run_once(self) -> TranscriptionResult:
        chunk = await self.input_queue.get()
        result = await asyncio.to_thread(self.transcriber.transcribe_chunk, chunk)
        await self.output_queue.put(result)
        return result

    async def read_result(self) -> TranscriptionResult:
        return await self.output_queue.get()
