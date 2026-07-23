"""UI: history dibuka di window utama (bukan window baru) + export PDF-only."""
from __future__ import annotations

import inspect
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

import src.qt_client as qt  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_export_is_pdf_only():
    src = inspect.getsource(qt.CompactWidget.export_transcript)
    assert "PDF (*.pdf)" in src
    assert "_export_transcript_pdf" in src
    # Tidak ada lagi jalur tulis .md / .txt.
    assert "Markdown" not in src and "Text files" not in src
    assert "# Transcript" not in src and "[{src}]" not in src


def test_entries_to_html_labels_sources():
    html = qt._entries_to_html([
        {"source": "mic", "text": "aku"},
        {"source": "speaker", "text": "kamu"},
        {"source": "other", "text": "lain"},
    ])
    assert "[Me]" in html and "aku" in html
    assert "[Speaker]" in html and "kamu" in html
    assert "[other]" in html


def test_show_history_entries_renders_in_main_window(qapp):
    w = qt.CompactWidget()
    w.show_history_entries([
        {"source": "mic", "text": "halo saya"},
        {"source": "speaker", "text": "halo pembicara"},
    ])
    assert w._viewing_history is True
    html = w.preview.toHtml()
    assert "halo saya" in html and "[Me]" in html
    assert "halo pembicara" in html and "[Speaker]" in html


# Regresi bug #3: saat mode riwayat, Export harus memakai entri yang ditampilkan,
# bukan log transkrip live (yang bisa kosong -> "No transcript to export").
def test_export_uses_history_entries_when_viewing_history(qapp, monkeypatch):
    w = qt.CompactWidget()
    w.show_history_entries([{"source": "mic", "text": "isi riwayat"}])

    # Log live kosong: tanpa fix, export akan gagal mendeteksi transkrip.
    monkeypatch.setattr(qt, "transcript_payload", lambda: {"entries": []})
    monkeypatch.setattr(
        qt.QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: ("C:/tmp/out.pdf", "PDF (*.pdf)")),
    )
    monkeypatch.setattr(qt.QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(qt.QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    captured = {}
    monkeypatch.setattr(qt, "_export_transcript_pdf", lambda path, entries: captured.update(entries=entries))

    w.export_transcript()
    assert captured.get("entries") == [{"source": "mic", "text": "isi riwayat"}]


def test_export_uses_live_payload_when_not_viewing_history(qapp, monkeypatch):
    w = qt.CompactWidget()
    assert w._viewing_history is False
    monkeypatch.setattr(qt, "transcript_payload", lambda: {"entries": [{"source": "mic", "text": "live"}]})
    monkeypatch.setattr(
        qt.QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: ("C:/tmp/out.pdf", "PDF (*.pdf)")),
    )
    monkeypatch.setattr(qt.QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    captured = {}
    monkeypatch.setattr(qt, "_export_transcript_pdf", lambda path, entries: captured.update(entries=entries))

    w.export_transcript()
    assert captured.get("entries") == [{"source": "mic", "text": "live"}]


def test_refresh_does_not_overwrite_history_view(qapp):
    w = qt.CompactWidget()
    w.show_history_entries([{"source": "mic", "text": "riwayat"}])
    w.refresh()  # tidak boleh menimpa
    assert "riwayat" in w.preview.toHtml()


def test_starting_record_exits_history_mode(qapp):
    w = qt.CompactWidget()
    w.show_history_entries([{"source": "mic", "text": "riwayat"}])
    w._start_live = lambda: None  # cegah spawn nyata
    w._ensure_microphone_permission = lambda: True
    w.toggle_record(True)
    assert w._viewing_history is False


def test_history_dialog_has_no_open_folder_button(qapp):
    from src import paths
    import src.core.engine as engine

    dlg = qt.HistoryDialog()
    texts = [b.text().lower() for b in dlg.findChildren(QtWidgets.QPushButton)]
    assert not any("folder" in t for t in texts)
    # Tombol yang ada: Buka (utama) + Tutup.
    assert any("buka" in t for t in texts) and any("tutup" in t for t in texts)


def test_open_button_disabled_until_selection(tmp_path, monkeypatch, qapp):
    import json
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import importlib
    from src import paths as _paths
    importlib.reload(_paths)
    import src.core.engine as engine

    cur = _paths.current_transcript_log()
    _paths.ensure_dir(cur.parent)
    cur.write_text(json.dumps({"source": "mic", "text": "isi", "completed": True}), encoding="utf-8")
    engine.archive_transcript_to_history(cur)

    dlg = qt.HistoryDialog()
    assert dlg.list.count() == 1
    assert dlg.open_btn.isEnabled() is False
    dlg.list.setCurrentRow(0)
    assert dlg.open_btn.isEnabled() is True


def test_list_history_includes_preview(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import importlib
    from src import paths as _paths
    importlib.reload(_paths)
    import src.core.engine as engine

    cur = _paths.current_transcript_log()
    _paths.ensure_dir(cur.parent)
    cur.write_text(json.dumps({"source": "mic", "text": "kalimat pertama", "completed": True}), encoding="utf-8")
    engine.archive_transcript_to_history(cur)
    hist = engine.list_history()
    assert hist and hist[0]["preview"] == "kalimat pertama"
