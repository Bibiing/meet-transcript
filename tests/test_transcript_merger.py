from __future__ import annotations

from src.whisper.merger import TranscriptMerger
from src.whisper.models import TranscriptionResult


def _result(source: str, start: float, end: float, text: str) -> TranscriptionResult:
    return TranscriptionResult(
        source=source,
        text=text,
        model_name="large-v3-turbo",
        language="id",
        start_seconds=start,
        duration_seconds=end - start,
    )


def test_merger_emits_meeting_labels() -> None:
    merger = TranscriptMerger(reorder_delay_seconds=0.0)

    entries = merger.add_result(_result("mic", 0.0, 1.0, "halo"))
    entries += merger.add_result(_result("speaker", 1.0, 2.0, "siap"))

    assert [entry.label for entry in entries] == ["Me", "Meeting"]
    assert "[Me] halo" in entries[0].display
    assert "[Meeting] siap" in entries[1].display


def test_merger_sorts_buffered_results_by_timestamp() -> None:
    merger = TranscriptMerger(reorder_delay_seconds=0.5)

    assert merger.add_result(_result("speaker", 2.0, 3.0, "kedua")) == []
    ready = merger.add_result(_result("mic", 0.5, 1.0, "pertama"))

    assert [entry.result.text for entry in ready] == ["pertama"]
    assert [entry.result.text for entry in merger.flush()] == ["kedua"]


def test_merger_deduplicates_repeated_history_segments() -> None:
    merger = TranscriptMerger(reorder_delay_seconds=0.0)
    result = _result("speaker", 1.0, 2.0, "agenda hari ini")

    first = merger.add_result(result)
    second = merger.add_result(result)

    assert len(first) == 1
    assert second == []


def test_merger_skips_incomplete_partial_segments() -> None:
    merger = TranscriptMerger(reorder_delay_seconds=0.0)

    assert merger.add_result(_result("mic", 0.0, 1.0, "belum stabil"), completed=False) == []
    assert merger.flush() == []


def test_merger_bounds_emitted_key_cache() -> None:
    merger = TranscriptMerger(reorder_delay_seconds=0.0, max_emitted_keys=2)
    first = _result("speaker", 0.0, 1.0, "pertama")

    assert merger.add_result(first)
    assert merger.add_result(_result("speaker", 1.0, 2.0, "kedua"))
    assert merger.add_result(_result("speaker", 2.0, 3.0, "ketiga"))

    # Key pertama sudah keluar dari bounded dedupe cache, sehingga boleh muncul
    # lagi pada sesi panjang tanpa membuat set tumbuh tanpa batas.
    assert merger.add_result(first)
