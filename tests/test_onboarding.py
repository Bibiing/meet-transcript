from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from src.onboarding import (  # noqa: E402
    FIRST_RUN_KEY,
    FirstRunWizard,
    MicrophoneGuidanceDialog,
    ensure_microphone_ready,
    is_first_run,
    mark_first_run_completed,
    maybe_run_first_run_wizard,
)
from src.permissions import PRIVACY_ALLOWED, PRIVACY_DENIED, MicrophoneAccess  # noqa: E402

READY = MicrophoneAccess(accessible=True, privacy=PRIVACY_ALLOWED)
BLOCKED = MicrophoneAccess(accessible=False, privacy=PRIVACY_DENIED)


class DictStore:
    def __init__(self, data: dict | None = None) -> None:
        self.data = dict(data or {})

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value


class BoomStore(DictStore):
    def get(self, key: str):
        raise RuntimeError("registry unavailable")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# Flag first-run -----------------------------------------------------------
def test_is_first_run_true_when_flag_absent() -> None:
    assert is_first_run(DictStore()) is True


def test_mark_completed_makes_it_not_first_run() -> None:
    store = DictStore()
    mark_first_run_completed(store)
    assert store.data[FIRST_RUN_KEY] == "true"
    assert is_first_run(store) is False


def test_is_first_run_false_when_store_unreadable() -> None:
    # Store bermasalah tidak boleh memaksa wizard di setiap startup.
    assert is_first_run(BoomStore()) is False


# Dispatch wizard ----------------------------------------------------------
class FakeWizard:
    created: list["FakeWizard"] = []

    def __init__(self, result: int) -> None:
        self._result = result
        self.exec_called = 0

    def exec(self) -> int:
        self.exec_called += 1
        return self._result


def _factory(result: int):
    def make(parent, **kwargs):
        wizard = FakeWizard(result)
        FakeWizard.created.append(wizard)
        return wizard

    return make


def test_wizard_skipped_when_not_first_run() -> None:
    FakeWizard.created.clear()
    store = DictStore({FIRST_RUN_KEY: "true"})
    assert maybe_run_first_run_wizard(store, wizard_factory=_factory(QtWidgets.QDialog.DialogCode.Accepted)) is True
    assert FakeWizard.created == []


def test_wizard_accepted_marks_flag_and_continues() -> None:
    FakeWizard.created.clear()
    store = DictStore()
    ok = maybe_run_first_run_wizard(store, wizard_factory=_factory(QtWidgets.QDialog.DialogCode.Accepted))
    assert ok is True
    assert FakeWizard.created[0].exec_called == 1
    assert is_first_run(store) is False


def test_wizard_cancelled_signals_safe_exit_without_marking_flag() -> None:
    FakeWizard.created.clear()
    store = DictStore()
    ok = maybe_run_first_run_wizard(store, wizard_factory=_factory(QtWidgets.QDialog.DialogCode.Rejected))
    assert ok is False
    assert store.data == {}  # belum selesai -> wizard tampil lagi di startup berikutnya


def test_wizard_skipped_without_store() -> None:
    assert maybe_run_first_run_wizard(None) is True


# Gate permission ----------------------------------------------------------
def test_ensure_microphone_ready_true_without_dialog_when_ready() -> None:
    assert ensure_microphone_ready(None, checker=lambda device: READY) is True


def test_ensure_microphone_ready_passes_device_to_checker() -> None:
    seen = {}

    def checker(device):
        seen["device"] = device
        return READY

    ensure_microphone_ready(None, device=2, checker=checker)
    assert seen["device"] == 2


def test_guidance_dialog_retry_accepts_once_microphone_becomes_ready(qapp) -> None:
    results = [BLOCKED, READY]
    dialog = MicrophoneGuidanceDialog(None, BLOCKED, checker=lambda device: results.pop(0))
    dialog._retry()  # masih diblokir -> dialog tetap terbuka
    assert dialog.isVisible() is False and dialog.result() == 0
    dialog._retry()  # sudah diizinkan -> lanjut
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


# Halaman wizard -----------------------------------------------------------
def test_wizard_has_exactly_three_pages(qapp) -> None:
    # Welcome -> Izin Mikrofon -> Selesai (tanpa langkah server / test connection).
    wizard = FirstRunWizard(checker=lambda device: READY)
    assert len(wizard.pageIds()) == 3


def test_permission_page_does_not_lock_user_out_when_microphone_blocked(qapp) -> None:
    # Wizard = onboarding, bukan penegakan: mikrofon terblokir tidak boleh mengunci
    # pengguna di luar aplikasi (Settings harus tetap dapat dijangkau).
    wizard = FirstRunWizard(checker=lambda device: BLOCKED)
    page = wizard.page(wizard.pageIds()[1])
    page.initializePage()
    assert page.ready is False
    assert page.isComplete() is True
    status = page._status.text()
    assert "privasi Windows" in status  # penyebab tetap dijelaskan
    assert "tetap dapat melanjutkan" in status  # dan konsekuensinya disebut eksplisit


def test_permission_page_reports_ready_microphone(qapp) -> None:
    wizard = FirstRunWizard(checker=lambda device: READY)
    page = wizard.page(wizard.pageIds()[1])
    page.initializePage()
    assert page.ready is True
    assert page.isComplete() is True
    assert page._status.text() == "Mikrofon siap digunakan."


def test_permission_page_retry_updates_state_once_microphone_becomes_ready(qapp) -> None:
    results = [BLOCKED, READY]
    wizard = FirstRunWizard(checker=lambda device: results.pop(0))
    page = wizard.page(wizard.pageIds()[1])
    page.initializePage()
    assert page.ready is False
    page._refresh()
    assert page.ready is True


def test_finish_page_text_follows_real_microphone_state(qapp) -> None:
    # Jangan menjanjikan "siap" bila mikrofon tidak siap.
    blocked_wizard = FirstRunWizard(checker=lambda device: BLOCKED)
    blocked_wizard.restart()
    for _ in range(2):
        blocked_wizard.next()
    assert "belum dapat digunakan" in blocked_wizard.currentPage()._text.text()

    ready_wizard = FirstRunWizard(checker=lambda device: READY)
    ready_wizard.restart()
    for _ in range(2):
        ready_wizard.next()
    assert "Mikrofon siap" in ready_wizard.currentPage()._text.text()
