"""Regresi: silent device rebinding & coupling Settings <-> validasi audio."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from src.core.engine import UiOptions  # noqa: E402
from src.qt_client import DEFAULT_DEVICE_LABEL, SettingsDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_default_device_entry_exists(qapp) -> None:
    dlg = SettingsDialog(options=UiOptions())
    assert dlg.mic_select.itemText(0) == DEFAULT_DEVICE_LABEL
    assert dlg.spk_select.itemText(0) == DEFAULT_DEVICE_LABEL
    assert dlg.mic_select.itemData(0) is None


def test_none_device_selects_default_entry(qapp) -> None:
    # "ikuti default sistem" HARUS punya representasi di UI.
    dlg = SettingsDialog(options=UiOptions())
    assert dlg.mic_select.currentData() is None
    assert dlg.spk_select.currentData() is None


def test_saving_without_touching_device_keeps_binding_none(qapp) -> None:
    # Regresi utama: mengubah hostname saja tidak boleh memindahkan mikrofon.
    dlg = SettingsDialog(options=UiOptions())
    dlg.host_input.setText("server-baru.example")
    opts = dlg.to_options()
    assert opts.host == "server-baru.example"
    assert opts.mic_device is None
    assert opts.speaker_device is None


def test_explicit_device_selection_round_trips(qapp) -> None:
    dlg = SettingsDialog(options=UiOptions())
    if dlg.mic_select.count() < 2:
        pytest.skip("tidak ada perangkat input konkret di mesin ini")
    dlg.mic_select.setCurrentIndex(1)
    chosen = dlg.mic_select.currentData()
    opts = dlg.to_options()
    assert opts.mic_device == chosen

    reopened = SettingsDialog(options=opts)
    assert reopened.mic_select.currentData() == chosen
    reopened.mic_select.setCurrentIndex(0)
    assert reopened.to_options().mic_device is None
