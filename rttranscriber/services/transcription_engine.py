from __future__ import annotations

from math import sqrt
from time import perf_counter
from typing import Protocol

from rttranscriber.model.audio import AudioChunk
from rttranscriber.model.transcription import TranscriptToken, TranscriptionChunkResult


class TranscriptEngine(Protocol):
    """Kontrak engine transcript agar pipeline Phase 3 mudah diganti."""

    def transcribe(self, chunk: AudioChunk) -> TranscriptionChunkResult: ...


class EnergyTokenTranscriptEngine:
    """
    Engine Python murni untuk Phase 3.

    Engine ini tidak mencoba melakukan speech-to-text semantik. Tugasnya adalah
    menyediakan output transcript deterministik agar worker async, merger, dan
    MVVM state bisa divalidasi tanpa dependency model eksternal.
    """

    def __init__(self, token_window_seconds: float = 0.5, silence_threshold: float = 250.0) -> None:
        self._token_window_seconds = token_window_seconds
        self._silence_threshold = silence_threshold

    def transcribe(self, chunk: AudioChunk) -> TranscriptionChunkResult:
        start = perf_counter()
        samples_per_token = max(1, int(chunk.audio_format.sample_rate * self._token_window_seconds))
        tokens: list[TranscriptToken] = []

        for offset in range(0, len(chunk.samples), samples_per_token):
            bucket = chunk.samples[offset : offset + samples_per_token]
            if not bucket:
                continue

            rms = sqrt(sum(sample * sample for sample in bucket) / len(bucket))
            if rms < self._silence_threshold:
                continue

            token_text = self._map_rms_to_token(rms)
            start_frame = chunk.start_frame_index + offset
            end_frame = min(start_frame + len(bucket), chunk.end_frame_index)
            tokens.append(
                TranscriptToken(
                    text=token_text,
                    start_frame_index=start_frame,
                    end_frame_index=end_frame,
                )
            )

        partial_text = " ".join(token.text for token in tokens)
        return TranscriptionChunkResult(
            chunk_start_frame_index=chunk.start_frame_index,
            chunk_end_frame_index=chunk.end_frame_index,
            partial_text=partial_text,
            tokens=tokens,
            processing_seconds=perf_counter() - start,
        )

    def _map_rms_to_token(self, rms: float) -> str:
        if rms < 1200.0:
            return "pelan"
        if rms < 5000.0:
            return "jelas"
        return "tegas"
