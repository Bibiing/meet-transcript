"""Async queue helpers for phase 3 audio chunks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.capture.audio_frame import AudioFrame
from src.engine.preprocessing import AudioPreprocessor, PreprocessedAudioChunk


@dataclass(slots=True)
class AsyncPreprocessQueue:
    """Preprocess frames and publish accepted speech chunks to asyncio.Queue."""

    preprocessor: AudioPreprocessor
    maxsize: int = 32
    queue: asyncio.Queue[PreprocessedAudioChunk] = field(init=False)

    def __post_init__(self) -> None:
        if self.maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.queue = asyncio.Queue(maxsize=self.maxsize)

    async def publish_frames(self, frames: list[AudioFrame]) -> int:
        chunks = self.preprocessor.preprocess_frames(frames)
        for chunk in chunks:
            await self.queue.put(chunk)
        return len(chunks)

    async def read(self) -> PreprocessedAudioChunk:
        return await self.queue.get()
