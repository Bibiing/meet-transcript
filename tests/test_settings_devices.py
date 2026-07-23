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


# Regresi bug #2: nama model di dropdown WAJIB cocok dengan daftar tervalidasi
# server WhisperLive; "large" polos gagal dimuat dan tak boleh muncul.
def test_model_options_match_whisperlive_model_sizes(qapp) -> None:
    from pathlib import Path

    dlg = SettingsDialog(options=UiOptions())
    items = [dlg.model_select.itemText(i) for i in range(dlg.model_select.count())]

    backend_src = Path("WhisperLive/whisper_live/backend/faster_whisper_backend.py").read_text(
        encoding="utf-8"
    )
    # Setiap opsi harus muncul sebagai literal di daftar model_sizes server.
    for name in items:
        assert f'"{name}"' in backend_src, f"model '{name}' tidak ada di WhisperLive model_sizes"

    assert "large" not in items  # "large" polos gagal dimuat
    assert "large-v3" in items and "medium" in items


def test_model_default_is_medium(qapp) -> None:
    dlg = SettingsDialog(options=UiOptions(model="medium"))
    assert dlg.model_select.currentText() == "medium"


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


# --- Advanced: rebalans, round-trip, dan UI helper -------------------------

# Regresi: to_options() dahulu membuat UiOptions() KOSONG, sehingga setiap field
# yang tidak diwakili UI diam-diam kembali ke default saat Settings disimpan.
# Kini justru MAYORITAS field tidak diekspos, jadi jaminan ini kritis.
def test_to_options_preserves_fields_not_exposed_in_ui(qapp) -> None:
    from pathlib import Path

    source = UiOptions(
        use_tls=True,
        hotwords="penyulang, gardu induk",
        initial_prompt="Rapat operasional distribusi.",
        mic_server_vad=False,
        speaker_server_vad=True,
        no_speech_thresh=0.42,
        local_agreement=False,
        speech_boundary_detection=False,
        speech_boundary_silence_seconds=1.7,
        mic_target_rms_db=-11.0,
        speaker_max_normalization_gain_db=25.0,
        chunk_seconds=1.5,
        auto_reconnect=False,
        reconnect_buffer_seconds=45.0,
        final_drain_seconds=20.0,
        ready_timeout=120.0,
        local_agreement_window_seconds=33.0,
        transcript_log=Path("X:/custom/log.json"),
    )
    opts = SettingsDialog(options=source).to_options()

    assert opts.use_tls is True, "TLS tidak boleh dimatikan diam-diam oleh Settings"
    assert opts.hotwords == "penyulang, gardu induk"
    assert opts.initial_prompt == "Rapat operasional distribusi."
    assert opts.mic_server_vad is False
    assert opts.speaker_server_vad is True
    assert opts.no_speech_thresh == 0.42
    assert opts.local_agreement is False
    assert opts.speech_boundary_detection is False
    assert opts.speech_boundary_silence_seconds == 1.7
    assert opts.mic_target_rms_db == -11.0
    assert opts.speaker_max_normalization_gain_db == 25.0
    assert opts.chunk_seconds == 1.5
    assert opts.auto_reconnect is False
    assert opts.reconnect_buffer_seconds == 45.0
    assert opts.final_drain_seconds == 20.0
    assert opts.ready_timeout == 120.0
    assert opts.local_agreement_window_seconds == 33.0
    assert opts.transcript_log == Path("X:/custom/log.json")


def test_advanced_exposes_only_the_agreed_settings(qapp) -> None:
    """Setelan yang sudah pasti/default tidak boleh muncul di Advanced."""
    dlg = SettingsDialog(options=UiOptions())
    for removed in (
        "hotwords_input", "initial_prompt_input", "use_tls_label",
        "mic_server_vad_cb", "speaker_server_vad_cb", "no_speech_thresh",
        "local_agreement_cb", "speech_boundary_cb", "speech_boundary_silence",
        "speech_boundary_max_wait", "mic_target_rms", "mic_max_gain",
        "spk_target_rms", "spk_max_gain", "chunk_seconds", "auto_reconnect_cb",
        "reconnect_buffer", "final_drain", "ready_timeout",
    ):
        assert not hasattr(dlg, removed), f"{removed} seharusnya tidak lagi ditampilkan"
    # Yang disepakati tetap ada.
    assert hasattr(dlg, "vad_threshold") and hasattr(dlg, "mic_webrtc_ns_cb")
    assert hasattr(dlg, "host_input") and hasattr(dlg, "port_input")


def test_exposed_advanced_settings_round_trip(qapp) -> None:
    source = UiOptions(vad_threshold=0.31, mic_webrtc_ns=True)
    opts = SettingsDialog(options=source).to_options()
    assert opts.vad_threshold == pytest.approx(0.31)
    assert opts.mic_webrtc_ns is True


# Slider + angka: keduanya WAJIB selalu menampilkan nilai yang sama.
def test_slider_and_spinbox_stay_in_sync(qapp) -> None:
    dlg = SettingsDialog(options=UiOptions(vad_threshold=0.55))
    slider, spin = dlg.vad_threshold_slider, dlg.vad_threshold

    assert slider.value() == pytest.approx(55, abs=1)

    spin.setValue(0.20)
    assert slider.value() == pytest.approx(20, abs=1)

    slider.setValue(80)
    assert spin.value() == pytest.approx(0.80, abs=0.01)


def test_slider_range_matches_spinbox_range(qapp) -> None:
    dlg = SettingsDialog(options=UiOptions())
    assert dlg.vad_threshold.minimum() == 0.0
    assert dlg.vad_threshold.maximum() == 1.0
    assert dlg.vad_threshold_slider.minimum() == 0
    assert dlg.vad_threshold_slider.maximum() == 100


def test_reset_defaults_restores_values_but_keeps_server(qapp) -> None:
    dlg = SettingsDialog(options=UiOptions(host="server.example", port=1234))
    dlg.vad_threshold.setValue(0.11)
    dlg.mic_webrtc_ns_cb.setChecked(True)

    dlg._reset_advanced_defaults()

    d = UiOptions()
    assert dlg.vad_threshold.value() == d.vad_threshold
    assert dlg.mic_webrtc_ns_cb.isChecked() == d.mic_webrtc_ns
    # Penyiapan lingkungan tidak ikut direset.
    assert dlg.host_input.text() == "server.example"
    assert dlg.port_input.value() == 1234


def test_every_advanced_control_has_tooltip(qapp) -> None:
    """UI helper: tiap kontrol lanjutan wajib menjelaskan dirinya."""
    dlg = SettingsDialog(options=UiOptions())
    controls = [
        dlg.vad_threshold, dlg.mic_webrtc_ns_cb, dlg.host_input, dlg.port_input,
    ]
    missing = [type(c).__name__ for c in controls if not c.toolTip().strip()]
    assert not missing, f"kontrol tanpa tooltip: {missing}"


def test_noise_reduction_tooltip_is_accurate(qapp) -> None:
    """Label lama menyebut 'WebRTC APM'; implementasinya noise gate di klien."""
    tip = SettingsDialog(options=UiOptions()).mic_webrtc_ns_cb.toolTip()
    assert "WebRTC" not in tip and "APM" not in tip
    assert "mikrofon" in tip.lower()


def test_advanced_labels_avoid_raw_jargon(qapp) -> None:
    """Label harus bahasa pengguna; jargon hanya boleh di dalam tooltip."""
    dlg = SettingsDialog(options=UiOptions())
    labels = [
        w.text() for w in dlg.advanced.findChildren(QtWidgets.QLabel)
        if w.objectName() not in {"GroupHint", "AdvancedNotice", "SliderEnd"}
    ]
    joined = " ".join(labels)
    for jargon in ("RMS", "VAD", "APM", "threshold", "boundary"):
        assert jargon not in joined, f"jargon '{jargon}' masih muncul di label: {joined}"


def test_advanced_shows_guidance_banner(qapp) -> None:
    """Banner penjelas wajib ada dan benar-benar terpasang di tab Advanced."""
    dlg = SettingsDialog(options=UiOptions())
    found = [
        w for w in dlg.advanced.findChildren(QtWidgets.QLabel)
        if w.objectName() == "AdvancedNotice"
    ]
    assert len(found) == 1
    assert "kursor" in found[0].text().lower()
