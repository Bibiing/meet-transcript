"""Riwayat transkrip persisten (fix penyimpanan ephemeral onefile) + arsip."""
from __future__ import annotations

import json
import time

import pytest


def _write_jsonl(path, entries):
    """Tulis transkrip dalam format NYATA (JSONL, satu entri per baris)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    # Arahkan direktori data per-user ke tmp agar tidak menyentuh %LOCALAPPDATA% asli.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from src import paths

    return paths


def test_data_dir_is_per_user_not_derived_from_file(data_dir):
    # Regresi inti: path TIDAK boleh relatif terhadap __file__ (ephemeral di onefile).
    p = str(data_dir.current_transcript_log())
    assert data_dir.app_data_dir().name == "ListenPLN"
    assert "onefile" not in p
    assert p.replace("\\", "/").endswith("ListenPLN/transcript_log.json")


def test_archive_skips_empty_transcript(data_dir):
    from src.core import engine

    cur = data_dir.current_transcript_log()
    # Sumber JSONL: hanya entri non-final (candidate) -> tidak ada transkrip final.
    _write_jsonl(cur, [{"source": "mic", "text": "x", "completed": False}])
    assert engine.archive_transcript_to_history(cur) is None
    assert engine.list_history() == []

    # Berkas kosong -> juga tidak diarsipkan.
    cur.write_text("", encoding="utf-8")
    assert engine.archive_transcript_to_history(cur) is None


def test_archive_persists_and_lists_newest_first(data_dir):
    from src.core import engine

    cur = data_dir.current_transcript_log()

    _write_jsonl(cur, [{"source": "mic", "text": "satu", "completed": True}])
    first = engine.archive_transcript_to_history(cur)
    assert first is not None and first.exists()
    # Berkas history HARUS berformat {"entries": [...]} (self-describing).
    saved = json.loads(first.read_text(encoding="utf-8"))
    assert isinstance(saved, dict) and len(saved["entries"]) == 1

    time.sleep(0.02)
    _write_jsonl(cur, [{"source": "mic", "text": "a", "completed": True}, {"source": "spk", "text": "b", "completed": True}])
    second = engine.archive_transcript_to_history(cur)
    assert second is not None and second != first  # nama unik walau berdekatan

    hist = engine.list_history()
    assert len(hist) == 2
    assert hist[0]["mtime"] >= hist[1]["mtime"]  # terbaru dulu
    assert hist[0]["entry_count"] == 2


def test_archive_unique_name_within_same_second(data_dir, monkeypatch):
    from src.core import engine

    cur = data_dir.current_transcript_log()
    _write_jsonl(cur, [{"text": "x", "completed": True}])

    # Bekukan timestamp -> dua arsip pada "detik" yang sama tidak boleh saling menimpa.
    class _Frozen:
        @staticmethod
        def now():
            import datetime as _dt
            return _dt.datetime(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr(engine, "datetime", _Frozen)
    a = engine.archive_transcript_to_history(cur)
    b = engine.archive_transcript_to_history(cur)
    assert a is not None and b is not None and a != b
    assert len(engine.list_history()) == 2
