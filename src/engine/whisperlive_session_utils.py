from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

from src.capture.audio_frame import AudioFrame
from src.capture.wav_sink import write_frames_to_wav
from src.engine.preprocessing import PreprocessedAudioChunk
from src.engine.transcript_merger import MEETING_SOURCE_LABELS
from src.engine.whisper import TranscriptionResult, TranscriptionSegment
from src.utils.logging import TranscriptLog

# callback type untuk menerima hasil transcript, log, dan entry merger
TranscriptCallback = Callable[[TranscriptionResult], None]
LogCallback = Callable[[str], None]
MergedEntryCallback = Callable[["MergedTranscriptEntry"], None]


class _ChunkArchive:
    """Menyimpan setiap chunk audio ke dalam file WAV untuk debugging."""
    def __init__(self, root_dir: Path) -> None:
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = root_dir / session_id
        self._counters = {"mic": 0, "speaker": 0}
        self._lock = threading.Lock()

    def save(self, chunk: PreprocessedAudioChunk) -> Path:
        with self._lock:
            index = self._counters.get(chunk.source, 0) + 1
            self._counters[chunk.source] = index
        output_path = self.root / chunk.source / f"{index:06d}.wav"
        frame = AudioFrame(
            source=chunk.source,  # type: ignore[arg-type]
            samples=chunk.samples.reshape(-1, 1),
            sample_rate=chunk.sample_rate,
            channels=1,
            timestamp_seconds=chunk.start_seconds,
        )
        return write_frames_to_wav(output_path, [frame])


def _wait_for_final_results(stats: "WhisperLiveSessionStats", timeout_seconds: float) -> None:
    """Menunggu hasil akhir dari server sebelum menutup sesi."""
    deadline = perf_counter() + timeout_seconds
    min_wait_deadline = perf_counter() + min(timeout_seconds, 5.0)
    last_results = stats.results_received
    quiet_since = perf_counter()
    while perf_counter() < deadline:
        threading.Event().wait(0.25)
        if stats.results_received != last_results:
            last_results = stats.results_received
            quiet_since = perf_counter()
        if perf_counter() >= min_wait_deadline and perf_counter() - quiet_since >= 1.5:
            return


class _PartialTranscriptPreview:
    """Menampilkan preview transcript parsial di console UI sebelum hasil final."""
    def __init__(self, *, min_interval_seconds: float = 0.75) -> None:
        self._last_text_by_source: dict[str, str] = {}
        self._last_print_by_source: dict[str, float] = {}
        self._min_interval_seconds = min_interval_seconds

    def show(self, source: str, segments: list[dict]) -> None:
        partials = [segment for segment in segments if not segment.get("completed", True)]
        if not partials:
            return

        segment = partials[-1]
        text = str(segment.get("text", "")).strip()
        if len(text) < 3:
            return

        normalized = " ".join(text.split())
        if normalized == self._last_text_by_source.get(source):
            return

        now = perf_counter()
        last_print = self._last_print_by_source.get(source, 0.0)
        if now - last_print < self._min_interval_seconds:
            return

        self._last_text_by_source[source] = normalized
        self._last_print_by_source[source] = now
        label = MEETING_SOURCE_LABELS.get(source, source.upper())
        start = _float_or_zero(segment.get("start"))
        end = _float_or_zero(segment.get("end"))
        print(f"[live {_format_timestamp(start)} - {_format_timestamp(end)}] [{label}] {normalized}", flush=True)


def _requested_sources(source: Literal["mic", "speaker", "both"]) -> list[Literal["mic", "speaker"]]:
    """Mengembalikan daftar sumber audio yang diminta (mic, speaker, atau keduanya)."""
    return ["mic", "speaker"] if source == "both" else [source]


def _results_from_segments(
    source: str,
    model_name: str,
    language: str | None,
    segments: list[dict],
) -> list[TranscriptionResult]:
    """Mengubah format JSON segment menjadi TranscriptionResult objects."""
    return [event.result for event in _events_from_segments(source, model_name, language, segments)]


@dataclass(frozen=True, slots=True)
class WhisperLiveTranscriptEvent:
    result: TranscriptionResult
    completed: bool
    reliability_score: float | None = None
    reliability_action: str | None = None


def _events_from_segments(
    source: str,
    model_name: str,
    language: str | None,
    segments: list[dict],
) -> list[WhisperLiveTranscriptEvent]:
    """Mengubah format segment JSON dari WhisperLive menjadi format internal event Transcript."""
    events: list[WhisperLiveTranscriptEvent] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = _float_or_zero(segment.get("start"))
        end = _float_or_zero(segment.get("end"))
        duration = max(0.0, end - start)
        result = (
            TranscriptionResult(
                source=source,
                text=text,
                model_name=model_name,
                language=language,
                start_seconds=start,
                duration_seconds=duration,
                segments=[
                    TranscriptionSegment(
                        start=start,
                        end=end,
                        text=text,
                    )
                ],
            )
        )
        events.append(
            WhisperLiveTranscriptEvent(
                result=result,
                completed=bool(segment.get("completed", True)),
                reliability_score=_optional_float(segment.get("reliability_score")),
                reliability_action=str(segment.get("reliability_action") or "") or None,
            )
        )
    return events


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit_merged_entry(
    entry,
    transcript_log: TranscriptLog | None,
    on_result: TranscriptCallback | None,
    *,
    on_log: LogCallback | None = None,
    on_merged_entry: MergedEntryCallback | None = None,
) -> None:
    """Mengirim hasil transcript ke berbagai output callback dan log."""
    if on_log is not None:
        on_log(entry.display)
    else:
        print(entry.display, flush=True)
        
    if transcript_log is not None:
        transcript_log.append_result(entry.result, label=entry.label, display=entry.display)
        
    if on_result is not None:
        on_result(entry.result)
        
    if on_merged_entry is not None:
        on_merged_entry(entry)


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_timestamp(seconds: float) -> str:
    """Memformat nilai detik menjadi string HH:MM:SS atau MM:SS."""
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
