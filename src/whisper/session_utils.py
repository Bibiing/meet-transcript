from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

from src.capture.models import AudioFrame
from src.capture.wav_sink import write_frames_to_wav
from src.preprocessing.core import PreprocessedAudioChunk
from src.whisper.merger import MEETING_SOURCE_LABELS
from src.whisper.models import TranscriptionResult, TranscriptionSegment
from src.utils.formatter import format_timestamp
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


@dataclass
class TranscriptQualityTracker:
    """Mengumpulkan metrik kualitas transcript per session."""

    session_started_at: float = field(default_factory=perf_counter)
    candidate_count_by_source: dict[str, int] = field(default_factory=lambda: {"mic": 0, "speaker": 0})
    stable_count_by_source: dict[str, int] = field(default_factory=lambda: {"mic": 0, "speaker": 0})
    stable_emitted_by_source: dict[str, int] = field(default_factory=lambda: {"mic": 0, "speaker": 0})
    short_stable_by_source: dict[str, int] = field(default_factory=lambda: {"mic": 0, "speaker": 0})
    first_stable_latency_by_source: dict[str, float] = field(default_factory=dict)

    def observe_event(self, event: "WhisperLiveTranscriptEvent") -> None:
        source = event.result.source if event.result.source in {"mic", "speaker"} else "other"
        if source not in self.candidate_count_by_source:
            self.candidate_count_by_source[source] = 0
            self.stable_count_by_source[source] = 0
            self.stable_emitted_by_source[source] = 0
            self.short_stable_by_source[source] = 0

        if event.completed:
            self.stable_count_by_source[source] += 1
            if source not in self.first_stable_latency_by_source:
                self.first_stable_latency_by_source[source] = round(perf_counter() - self.session_started_at, 3)
            if len(event.result.text.split()) < 3:
                self.short_stable_by_source[source] += 1
        else:
            self.candidate_count_by_source[source] += 1

    def observe_emit(self, source: str) -> None:
        key = source if source in {"mic", "speaker"} else "other"
        if key not in self.stable_emitted_by_source:
            self.stable_emitted_by_source[key] = 0
        self.stable_emitted_by_source[key] += 1

    def summary(self) -> dict[str, object]:
        sources = sorted(
            set(self.candidate_count_by_source)
            | set(self.stable_count_by_source)
            | set(self.stable_emitted_by_source)
        )
        by_source: dict[str, dict[str, object]] = {}
        total_candidate = 0
        total_stable = 0
        for source in sources:
            candidate = int(self.candidate_count_by_source.get(source, 0))
            stable = int(self.stable_count_by_source.get(source, 0))
            total = candidate + stable
            total_candidate += candidate
            total_stable += stable
            by_source[source] = {
                "candidate": candidate,
                "stable": stable,
                "stable_ratio": round(stable / total, 3) if total else None,
                "stable_emitted": int(self.stable_emitted_by_source.get(source, 0)),
                "short_stable": int(self.short_stable_by_source.get(source, 0)),
                "first_stable_latency_seconds": self.first_stable_latency_by_source.get(source),
            }

        total = total_candidate + total_stable
        return {
            "candidate": total_candidate,
            "stable": total_stable,
            "stable_ratio": round(total_stable / total, 3) if total else None,
            "by_source": by_source,
        }


class _RollingAudioArchive:
    """Menyimpan chunk preprocessed menjadi WAV segment besar per source."""

    def __init__(
        self,
        root_dir: Path,
        *,
        session_id: str,
        segment_seconds: float = 60.0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.root = root_dir / session_id
        self.segment_seconds = max(1.0, float(segment_seconds))
        self.metadata = metadata or {}
        self._buffers: dict[str, list[PreprocessedAudioChunk]] = {"mic": [], "speaker": []}
        self._durations: dict[str, float] = {"mic": 0.0, "speaker": 0.0}
        self._counters: dict[str, int] = {"mic": 0, "speaker": 0}
        self._segments: list[dict[str, object]] = []
        self._lock = threading.Lock()

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def save(self, chunk: PreprocessedAudioChunk) -> list[Path]:
        with self._lock:
            source = chunk.source
            if source not in self._buffers:
                self._buffers[source] = []
                self._durations[source] = 0.0
                self._counters[source] = 0
            self._buffers[source].append(chunk)
            self._durations[source] += chunk.duration_seconds
            if self._durations[source] >= self.segment_seconds:
                return [self._flush_source_locked(source)]
            return []

    def close(self) -> list[Path]:
        with self._lock:
            paths: list[Path] = []
            for source in list(self._buffers):
                if self._buffers[source]:
                    paths.append(self._flush_source_locked(source))
            self._write_metadata_locked()
            return paths

    def _flush_source_locked(self, source: str) -> Path:
        chunks = self._buffers[source]
        if not chunks:
            raise ValueError("cannot flush an empty rolling audio buffer")

        self._counters[source] += 1
        index = self._counters[source]
        start_seconds = chunks[0].start_seconds
        end_seconds = chunks[-1].start_seconds + chunks[-1].duration_seconds
        output_path = self.root / source / f"{index:06d}-{int(start_seconds * 1000):010d}-{int(end_seconds * 1000):010d}.wav"
        frames = [
            AudioFrame(
                source=chunk.source,  # type: ignore[arg-type]
                samples=chunk.samples.reshape(-1, 1),
                sample_rate=chunk.sample_rate,
                channels=1,
                timestamp_seconds=chunk.start_seconds,
            )
            for chunk in chunks
        ]
        write_frames_to_wav(output_path, frames)
        self._segments.append(
            {
                "source": source,
                "path": str(output_path.relative_to(self.root)),
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "chunk_count": len(chunks),
                "sample_rate": chunks[0].sample_rate,
            }
        )
        self._buffers[source] = []
        self._durations[source] = 0.0
        self._write_metadata_locked()
        return output_path

    def _write_metadata_locked(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "segment_seconds": self.segment_seconds,
            **self.metadata,
            "segments": self._segments,
        }
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return format_timestamp(seconds)
