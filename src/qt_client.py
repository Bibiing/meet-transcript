from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import json

from PySide6 import QtCore, QtGui, QtWidgets

from src.core.engine import (
    archive_transcript_to_history,
    audio_devices_payload,
    audio_levels,
    connection_status,
    list_history,
    mic_error_info,
    outdated_client_info,
    set_mute,
    tls_error_info,
    start_live,
    stop_live,
    transcript_payload,
    DEFAULT_TRANSCRIPT_LOG,
    UiOptions,
)
from src.config import ConfigProvider, QtUserStore
from src.onboarding import ensure_microphone_ready, maybe_run_first_run_wizard
from src.permissions import (
    GUIDANCE_NONE,
    check_microphone,
    guidance_kind,
    guidance_message,
    source_uses_microphone,
)
from src.version import app_version
from src.utils.mode_detector import is_production_build, should_enable_debug_features


def _build_user_store() -> Optional[QtUserStore]:
    """QSettings per-user; None bila tak tersedia (tanpa persistensi)."""
    try:
        return QtUserStore()
    except Exception:
        return None


def _build_config_provider() -> ConfigProvider:
    """Config provider GUI: bundled defaults + override user (QSettings)."""
    store = _build_user_store()
    if store is None:
        # QSettings gagal (mis. env aneh) -> tetap jalan tanpa persistensi.
        return ConfigProvider()
    return ConfigProvider(user_store=store)


def _options_from_config(cfg: ConfigProvider, base: Optional[UiOptions] = None) -> UiOptions:
    """Bangun UiOptions dari config provider (server URL, model, bahasa, domain adaptation)."""
    opts = base or UiOptions()
    opts.host = cfg.server_host()
    opts.port = cfg.server_port()
    opts.use_tls = cfg.use_tls()
    opts.model = cfg.model()
    opts.language = cfg.language()
    opts.initial_prompt = cfg.initial_prompt()
    opts.hotwords = cfg.hotwords()
    return opts


PRIMARY = "#6AD8E5"
PRIMARY_HOVER = "#4BBAC8"
SECONDARY = "#F1F5F9" 
SECONDARY_HOVER = "#E2E8F0"
DANGER = "#E84D5B"
DANGER_HOVER = "#D43C4A"
BG = "#F2F9FA"
TEXT = "#112629"
TEXT_MUTED = "#C0D6D8"
BORDER = "#B6CFD3"

# Label entri perangkat "ikuti default OS" (data=None) pada combo Settings.
DEFAULT_DEVICE_LABEL = "Default sistem"

# Direktori root aplikasi, dihitung dari lokasi file ini (bukan cwd saat dijalankan).
# Memastikan ikon tetap ditemukan meskipun aplikasi dijalankan dari folder lain (mis. via
# `python -m src.qt_client` dari root project vs dijalankan langsung dari folder src).
APP_ROOT = Path(__file__).resolve().parent
ICON_DIR = APP_ROOT / "icon"


def load_icon(filename: str) -> QtGui.QIcon:
    """Muat ikon secara aman berdasarkan path absolut.

    Mengembalikan QIcon kosong (bukan crash) jika file tidak ditemukan,
    sehingga tombol tetap berfungsi dengan label teks meski ikon hilang.
    """
    # 1. Cek di folder icon relatif terhadap file ini (Nuitka temp extraction dir)
    path = ICON_DIR / filename
    
    # 2. Fallback: cari di direktori tempat file executable berada
    if not path.exists():
        exe_dir = Path(sys.argv[0]).resolve().parent
        path = exe_dir / "src" / "icon" / filename
        
    # 3. Fallback: cari di Current Working Directory (CWD)
    if not path.exists():
        path = Path("src") / "icon" / filename

    if not path.exists():
        return QtGui.QIcon()
    return QtGui.QIcon(str(path))



class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, options: Optional[UiOptions] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setWindowIcon(load_icon("apps.png"))
        self.setModal(True)
        self.resize(520, 320)
        self.setSizeGripEnabled(True)
        self.is_production = is_production_build()
        self.debug_enabled = should_enable_debug_features()

        self.tabs = QtWidgets.QTabWidget()
        self.general = QtWidgets.QWidget()
        self.advanced = QtWidgets.QWidget()

        self.tabs.addTab(self.general, "General")
        self.tabs.addTab(self.advanced, "Advanced")

        # General layout - tampilan utama yang sederhana
        g_layout = QtWidgets.QFormLayout()
        self.mic_select = QtWidgets.QComboBox()
        self.spk_select = QtWidgets.QComboBox()
        self.model_select = QtWidgets.QComboBox()
        # Nama model WAJIB cocok dengan daftar tervalidasi server WhisperLive
        # (faster_whisper_backend.model_sizes). "large" polos tidak ada di sana dan
        # gagal dimuat; gunakan nama eksplisit (large-v2/large-v3/...). Varian .en dan
        # distil-* sengaja tidak disertakan karena English-only (aplikasi ini id).
        self.model_select.addItems(
            ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo", "turbo"]
        )
        self.lang_select = QtWidgets.QComboBox()
        self.lang_select.addItems(["id", "en"])
        self.llm_api_key = QtWidgets.QLineEdit()
        self.llm_api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.llm_api_key.setPlaceholderText("OpenAI / LLM API key")

        self.mic_select.setToolTip("Pilih perangkat mikrofon input")
        self.spk_select.setToolTip("Pilih perangkat output/speaker yang direkam")
        self.lang_select.setToolTip("Bahasa transkripsi")
        self.model_select.setToolTip("Model Whisper: makin besar makin akurat namun makin berat")
        self.llm_api_key.setToolTip("API key untuk fitur ringkasan (Summarize). Disimpan hanya di memori, tidak ditulis ke disk")

        g_layout.addRow("Microphone:", self.mic_select)
        g_layout.addRow("Speaker:", self.spk_select)
        g_layout.addRow("Language:", self.lang_select)
        g_layout.addRow("Model:", self.model_select)
        # W4: versi terpasang terlihat pengguna, agar dukungan dapat menanyakannya.
        self.version_label = QtWidgets.QLabel(app_version())
        self.version_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.version_label.setToolTip("Versi aplikasi yang terpasang")
        g_layout.addRow("Version:", self.version_label)
        g_layout.addRow("LLM API Key:", self.llm_api_key)

        self.general.setLayout(g_layout)

        # Advanced settings (technical)
        a_layout = QtWidgets.QFormLayout()

        # Server host & port - dipindah ke Advanced
        self.host_input = QtWidgets.QLineEdit("localhost")
        self.port_input = QtWidgets.QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(9090)
        self.host_input.setToolTip("Alamat server transcription (default: localhost)")
        self.port_input.setToolTip("Port server transcription (1-65535)")
        a_layout.addRow("Server host:", self.host_input)
        a_layout.addRow("Server port:", self.port_input)

        # VAD and normalization
        self.mic_server_vad_cb = QtWidgets.QCheckBox()
        self.mic_server_vad_cb.setChecked(True)
        self.speaker_server_vad_cb = QtWidgets.QCheckBox()
        self.speaker_server_vad_cb.setChecked(False)
        self.mic_webrtc_ns_cb = QtWidgets.QCheckBox("Enable Mic Noise Suppression (WebRTC APM) [Experimental]")
        self.mic_webrtc_ns_cb.setChecked(False)

        self.mic_target_rms = QtWidgets.QDoubleSpinBox()
        self.mic_target_rms.setRange(-60.0, 0.0)
        self.mic_target_rms.setValue(-20.0)
        self.mic_max_gain = QtWidgets.QDoubleSpinBox()
        self.mic_max_gain.setRange(0.0, 60.0)
        self.mic_max_gain.setValue(18.0)

        self.spk_target_rms = QtWidgets.QDoubleSpinBox()
        self.spk_target_rms.setRange(-60.0, 0.0)
        self.spk_target_rms.setValue(-23.0)
        self.spk_max_gain = QtWidgets.QDoubleSpinBox()
        self.spk_max_gain.setRange(0.0, 60.0)
        self.spk_max_gain.setValue(18.0)

        # VAD thresholds
        self.vad_threshold = QtWidgets.QDoubleSpinBox()
        self.vad_threshold.setRange(0.0, 1.0)
        self.vad_threshold.setSingleStep(0.01)
        self.vad_threshold.setValue(0.55)

        self.no_speech_thresh = QtWidgets.QDoubleSpinBox()
        self.no_speech_thresh.setRange(0.0, 1.0)
        self.no_speech_thresh.setSingleStep(0.01)
        self.no_speech_thresh.setValue(0.75)

        self.speech_boundary_silence = QtWidgets.QDoubleSpinBox()
        self.speech_boundary_silence.setRange(0.0, 10.0)
        self.speech_boundary_silence.setValue(0.8)
        self.speech_boundary_max_wait = QtWidgets.QDoubleSpinBox()
        self.speech_boundary_max_wait.setRange(0.0, 30.0)
        self.speech_boundary_max_wait.setValue(5.0)

        # Debug/archive - hanya tampilkan di development
        self.debug_chunk_archive_cb = QtWidgets.QCheckBox()
        self.rolling_audio_archive_cb = QtWidgets.QCheckBox()
        self.rolling_audio_segment = QtWidgets.QSpinBox()
        self.rolling_audio_segment.setRange(1, 3600)
        self.rolling_audio_segment.setValue(60)

        # Log level - hanya tampilkan di development
        self.log_level = QtWidgets.QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])

        # Tooltip untuk parameter teknis
        self.mic_server_vad_cb.setToolTip("Aktifkan Voice Activity Detection di server untuk audio mikrofon")
        self.speaker_server_vad_cb.setToolTip("Aktifkan Voice Activity Detection di server untuk audio speaker")
        self.mic_webrtc_ns_cb.setToolTip(
            "Dapat meredam desis latar belakang (hum/hiss) pada mikrofon, namun berpotensi mengubah karakteristik suara. "
            "Lakukan tes performa jika akurasi transkripsi menurun."
        )
        self.mic_target_rms.setToolTip("Target volume (RMS) mikrofon setelah normalisasi, dalam dB")
        self.mic_max_gain.setToolTip("Batas maksimal penguatan (gain) yang diterapkan pada mikrofon")
        self.spk_target_rms.setToolTip("Target volume (RMS) speaker setelah normalisasi, dalam dB")
        self.spk_max_gain.setToolTip("Batas maksimal penguatan (gain) yang diterapkan pada speaker")
        self.vad_threshold.setToolTip("Ambang sensitivitas deteksi suara (0-1). Semakin tinggi, semakin ketat")
        self.no_speech_thresh.setToolTip("Ambang probabilitas untuk menganggap segmen sebagai 'tidak ada suara'")
        self.speech_boundary_silence.setToolTip("Durasi hening (detik) sebelum segmen ucapan dianggap selesai")
        self.speech_boundary_max_wait.setToolTip("Waktu tunggu maksimum (detik) sebelum segmen dipotong paksa")
        self.debug_chunk_archive_cb.setToolTip("Simpan potongan audio mentah untuk keperluan debugging")
        self.rolling_audio_archive_cb.setToolTip("Simpan rekaman audio berkelanjutan ke disk")
        self.rolling_audio_segment.setToolTip("Panjang tiap segmen arsip audio berkelanjutan, dalam detik")
        self.log_level.setToolTip("Tingkat detail log aplikasi")

        a_layout.addRow("Mic server VAD:", self.mic_server_vad_cb)
        a_layout.addRow("Speaker server VAD:", self.speaker_server_vad_cb)
        a_layout.addRow(self.mic_webrtc_ns_cb)
        a_layout.addRow("Mic target RMS (dB):", self.mic_target_rms)
        a_layout.addRow("Mic max gain (dB):", self.mic_max_gain)
        a_layout.addRow("Speaker target RMS (dB):", self.spk_target_rms)
        a_layout.addRow("Speaker max gain (dB):", self.spk_max_gain)
        a_layout.addRow("VAD threshold:", self.vad_threshold)
        a_layout.addRow("No-speech threshold:", self.no_speech_thresh)
        a_layout.addRow("Speech boundary silence (s):", self.speech_boundary_silence)
        a_layout.addRow("Speech boundary max wait (s):", self.speech_boundary_max_wait)

        # Debug features hanya di development
        if self.debug_enabled:
            a_layout.addRow("Debug chunk archive:", self.debug_chunk_archive_cb)
            a_layout.addRow("Rolling audio archive:", self.rolling_audio_archive_cb)
            a_layout.addRow("Rolling audio segment (s):", self.rolling_audio_segment)
            a_layout.addRow("Log level:", self.log_level)
        
        self.advanced.setLayout(a_layout)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        ok_btn = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Save Settings")
        ok_btn.setObjectName("SaveBtn")
        ok_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        ok_btn.setToolTip("Simpan pengaturan dan tutup dialog")
        ok_btn.setDefault(True)

        cancel_btn = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setObjectName("CancelBtn")
        cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        cancel_btn.setToolTip("Batalkan perubahan (Esc)")

        main = QtWidgets.QVBoxLayout()
        main.addWidget(self.tabs)
        main.addWidget(btns)
        self.setLayout(main)

        self.reload_devices()
        
        if options:
            self.load_options(options)

    def reload_devices(self):
        try:
            payload = audio_devices_payload()
            # Entri "Default sistem" (data=None) WAJIB ada: tanpanya, state "ikuti
            # default OS" tidak dapat direpresentasikan di UI dan akan berubah
            # menjadi device konkret setiap kali Settings disimpan (silent rebinding).
            self.mic_select.clear()
            self.mic_select.addItem(DEFAULT_DEVICE_LABEL, None)
            for d in payload.get("mic", []):
                self.mic_select.addItem(d.get("label") or d.get("name"), d.get("id"))
            self.spk_select.clear()
            self.spk_select.addItem(DEFAULT_DEVICE_LABEL, None)
            for d in payload.get("speaker", []):
                self.spk_select.addItem(d.get("name"), d.get("id"))
        except Exception:
            # best-effort; ignore errors
            pass

    def load_options(self, opts: UiOptions):
        if opts.host is not None:
            self.host_input.setText(opts.host)
        if opts.port is not None:
            self.port_input.setValue(opts.port)
        
        # Pilih berdasarkan data ID, TERMASUK None -> entri "Default sistem",
        # sehingga round-trip None -> UI -> None utuh.
        idx = self.mic_select.findData(opts.mic_device)
        if idx >= 0:
            self.mic_select.setCurrentIndex(idx)
        idx = self.spk_select.findData(opts.speaker_device)
        if idx >= 0:
            self.spk_select.setCurrentIndex(idx)
                
        # Seed server host/port dari opts (sebelumnya selalu "localhost"/9090).
        if getattr(opts, "host", None):
            self.host_input.setText(str(opts.host))
        if getattr(opts, "port", None):
            self.port_input.setValue(int(opts.port))

        idx = self.lang_select.findText(opts.language)
        if idx >= 0:
            self.lang_select.setCurrentIndex(idx)

        idx = self.model_select.findText(opts.model)
        if idx >= 0:
            self.model_select.setCurrentIndex(idx)
            
        self.mic_server_vad_cb.setChecked(opts.mic_server_vad)
        self.speaker_server_vad_cb.setChecked(opts.speaker_server_vad)
        self.mic_webrtc_ns_cb.setChecked(opts.mic_webrtc_ns)
        self.mic_target_rms.setValue(opts.mic_target_rms_db)
        self.mic_max_gain.setValue(opts.mic_max_normalization_gain_db)
        self.spk_target_rms.setValue(opts.speaker_target_rms_db)
        self.spk_max_gain.setValue(opts.speaker_max_normalization_gain_db)
        self.vad_threshold.setValue(opts.vad_threshold)
        self.no_speech_thresh.setValue(opts.no_speech_thresh)
        self.speech_boundary_silence.setValue(opts.speech_boundary_silence_seconds)
        self.speech_boundary_max_wait.setValue(opts.speech_boundary_max_wait_seconds)
        
        # Debug features hanya ada di development mode
        if self.debug_enabled:
            self.debug_chunk_archive_cb.setChecked(opts.debug_chunk_archive)
            self.rolling_audio_archive_cb.setChecked(opts.rolling_audio_archive)
            self.rolling_audio_segment.setValue(int(opts.rolling_audio_segment_seconds))
            idx = self.log_level.findText(opts.log_level)
            if idx >= 0:
                self.log_level.setCurrentIndex(idx)

    def to_options(self) -> UiOptions:
        opts = UiOptions()
        opts.host = self.host_input.text().strip() or "localhost"
        opts.port = int(self.port_input.value())
        opts.mic_device = self.mic_select.currentData()
        opts.speaker_device = self.spk_select.currentData()
        opts.language = str(self.lang_select.currentText())
        opts.model = str(self.model_select.currentText())
        
        opts.mic_server_vad = self.mic_server_vad_cb.isChecked()
        opts.speaker_server_vad = self.speaker_server_vad_cb.isChecked()
        opts.mic_webrtc_ns = self.mic_webrtc_ns_cb.isChecked()
        opts.mic_target_rms_db = self.mic_target_rms.value()
        opts.mic_max_normalization_gain_db = self.mic_max_gain.value()
        opts.speaker_target_rms_db = self.spk_target_rms.value()
        opts.speaker_max_normalization_gain_db = self.spk_max_gain.value()
        opts.vad_threshold = self.vad_threshold.value()
        opts.no_speech_thresh = self.no_speech_thresh.value()
        opts.speech_boundary_silence_seconds = self.speech_boundary_silence.value()
        opts.speech_boundary_max_wait_seconds = self.speech_boundary_max_wait.value()
        
        # Debug dan log level hanya ada di development mode
        if self.debug_enabled:
            opts.debug_chunk_archive = self.debug_chunk_archive_cb.isChecked()
            opts.rolling_audio_archive = self.rolling_audio_archive_cb.isChecked()
            opts.rolling_audio_segment_seconds = float(self.rolling_audio_segment.value())
            opts.log_level = str(self.log_level.currentText())
        else:
            # Production mode: selalu disable debug
            opts.debug_chunk_archive = False
            opts.rolling_audio_archive = False
            opts.rolling_audio_segment_seconds = 60.0
            opts.log_level = "WARNING"
        
        return opts


class SummaryWindow(QtWidgets.QDialog):
    def __init__(self, transcript: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transcript Summary")
        self.setWindowIcon(load_icon("app.ico"))
        self.resize(700, 500)
        self._entries = transcript.get("entries", [])
        layout = QtWidgets.QVBoxLayout()
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        content = "\n".join([e.get("text", "") for e in self._entries])
        self.text.setPlainText(content)
        layout.addWidget(self.text)

        btns = QtWidgets.QHBoxLayout()
        export_pdf = QtWidgets.QPushButton("Export PDF")
        export_pdf.setObjectName("SecondaryBtn")
        export_pdf.clicked.connect(self.export_pdf)
        btns.addStretch(1)
        btns.addWidget(export_pdf)
        layout.addLayout(btns)
        self.setLayout(layout)

    def export_pdf(self):
        if not self._entries:
            QtWidgets.QMessageBox.information(self, "No transcript", "No transcript data to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export transcript", "transcript.pdf", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            _export_transcript_pdf(path, self._entries)
            QtWidgets.QMessageBox.information(self, "Exported", f"Transcript exported to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to export: {exc}")


def _entries_to_html(entries: list) -> str:
    """Render entri transkrip ke HTML dengan label source (Mic/Speaker).

    Dipakai bersama oleh live view DAN tampilan history agar identik.
    """
    rows = []
    for e in entries:
        text = e.get("text", "")
        source = str(e.get("source") or "other")
        if source == "mic":
            label = '<b style="color: #2563EB;">[Me]</b>'
        elif source == "speaker":
            label = '<b style="color: #059669;">[Speaker]</b>'
        else:
            label = f'<b style="color: #64748B;">[{source}]</b>'
        rows.append(f"{label} {text}")
    return "<br>".join(rows)


def _export_transcript_pdf(path: str, entries: list) -> None:
    """Render transkrip ke PDF memakai Qt (QTextDocument -> QPdfWriter)"""
    lines = []
    for e in entries:
        src = e.get("source") or "other"
        text = (e.get("text", "") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f"<p><b>[{src}]</b> {text}</p>")
    doc = QtGui.QTextDocument()
    doc.setHtml("<h2>Transcript</h2>" + "".join(lines))
    writer = QtGui.QPdfWriter(path)
    writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
    doc.print_(writer)


def _humanize_history_time(epoch: float) -> str:
    """Waktu yang enak dibaca: 'Hari ini 14:30', 'Kemarin 09:15', atau tanggal penuh."""
    _BULAN = ["", "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    now = time.localtime()
    t = time.localtime(epoch)
    jam = time.strftime("%H:%M", t)
    today = time.strftime("%Y-%m-%d", now)
    that_day = time.strftime("%Y-%m-%d", t)
    yesterday = time.strftime("%Y-%m-%d", time.localtime(time.mktime(now) - 86400))
    if that_day == today:
        return f"Hari ini {jam}"
    if that_day == yesterday:
        return f"Kemarin {jam}"
    return f"{t.tm_mday} {_BULAN[t.tm_mon]} {t.tm_year} · {jam}"


class _ElidedLabel(QtWidgets.QLabel):
    """QLabel yang meng-elide teks (…) mengikuti lebar tersedia, responsif."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        super().setText(text)

    def resizeEvent(self, event):  # noqa: N802 (API Qt)
        metrics = self.fontMetrics()
        super().setText(metrics.elidedText(self._full, QtCore.Qt.TextElideMode.ElideRight, self.width()))
        super().resizeEvent(event)


class HistoryDialog(QtWidgets.QDialog):
    """Riwayat transkrip dari folder khusus per-user (bukan dialog file kosong)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Riwayat Transkrip")
        self.setWindowIcon(load_icon("app.ico"))
        self.setMinimumSize(560, 460)
        self.setStyleSheet(self._stylesheet())

        items = list_history()

        header = QtWidgets.QLabel("Riwayat Transkrip")
        header.setObjectName("HistTitle")

        separator = QtWidgets.QLabel("|")
        separator.setObjectName("HistSeparator")

        subtitle = QtWidgets.QLabel(
            f"{len(items)} rekaman tersimpan" if items else "Belum ada rekaman tersimpan"
        )
        subtitle.setObjectName("HistSubtitle")

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(header, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addSpacing(8) 
        header_layout.addWidget(separator, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addSpacing(8) 
        header_layout.addWidget(subtitle, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addStretch(1)

        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("HistList")
        self.list.setSpacing(4)
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemDoubleClicked.connect(lambda _it: self._open_selected())
        self.list.itemSelectionChanged.connect(self._sync_open_enabled)
        self._populate(items)

        self.open_btn = QtWidgets.QPushButton("Buka")
        self.open_btn.setObjectName("HistPrimary")
        self.open_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.open_btn.clicked.connect(self._open_selected)
        self.open_btn.setEnabled(False)

        close_btn = QtWidgets.QPushButton("Tutup")
        close_btn.setObjectName("HistGhost")
        close_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        close_btn.clicked.connect(self.reject)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        buttons.addWidget(self.open_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(4)

        layout.addLayout(header_layout)
        layout.addSpacing(10)

        if items:
            layout.addWidget(self.list, 1)
        else:
            layout.addWidget(self._empty_state(), 1)
        layout.addSpacing(6)
        layout.addLayout(buttons)

    def _empty_state(self) -> QtWidgets.QWidget:
        box = QtWidgets.QFrame()
        box.setObjectName("HistEmpty")
        lay = QtWidgets.QVBoxLayout(box)
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        icon = QtWidgets.QLabel("🗂")
        icon.setObjectName("HistEmptyIcon")
        icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        msg = QtWidgets.QLabel(
            "Transkrip akan muncul di sini secara otomatis\n"
            "setiap kali Anda menghentikan rekaman yang berisi teks."
        )
        msg.setObjectName("HistEmptyMsg")
        msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon)
        lay.addWidget(msg)
        return box

    def _populate(self, items: list) -> None:
        for item in items:
            row = self._make_row(item)
            widget_item = QtWidgets.QListWidgetItem(self.list)
            widget_item.setData(QtCore.Qt.ItemDataRole.UserRole, item["path"])
            widget_item.setSizeHint(row.sizeHint())
            self.list.addItem(widget_item)
            self.list.setItemWidget(widget_item, row)

    def _make_row(self, item: dict) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        row.setObjectName("HistRow")
        outer = QtWidgets.QVBoxLayout(row)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        when = QtWidgets.QLabel(_humanize_history_time(item["mtime"]))
        when.setObjectName("HistWhen")
        count = QtWidgets.QLabel(f"{item['entry_count']} baris")
        count.setObjectName("HistCount")
        top.addWidget(when)
        top.addStretch(1)
        top.addWidget(count)

        preview_text = (item.get("preview") or "Tanpa teks").strip()
        preview = _ElidedLabel(preview_text)
        preview.setObjectName("HistPreview")

        outer.addLayout(top)
        outer.addWidget(preview)
        return row

    def _sync_open_enabled(self) -> None:
        self.open_btn.setEnabled(self.list.currentItem() is not None)

    @staticmethod
    def _stylesheet() -> str:
        return f"""
            QDialog {{ background: {BG}; }}
            QLabel#HistTitle {{ color: {TEXT}; font-size: 17px; font-weight: 700; }}
            QLabel#HistSubtitle {{ color: #6B8890; font-size: 12px; }}
            QListWidget#HistList {{
                background: transparent; border: none; outline: none;
            }}
            QListWidget#HistList::item {{
                border: 1px solid {BORDER}; border-radius: 10px;
                background: white; margin: 0px; padding: 0px;
            }}
            QListWidget#HistList::item:selected {{
                border: 1px solid {PRIMARY}; background: #EAF9FB;
            }}
            QListWidget#HistList::item:hover {{ border: 1px solid {PRIMARY}; }}
            QWidget#HistRow {{ background: transparent; }}
            QLabel#HistWhen {{ color: {TEXT}; font-size: 13px; font-weight: 600; }}
            QLabel#HistCount {{
                color: #4B6B72; font-size: 11px; font-weight: 600;
                background: {SECONDARY}; border-radius: 9px; padding: 2px 9px;
            }}
            QLabel#HistPreview {{ color: #64808A; font-size: 12px; }}
            QFrame#HistEmpty {{
                background: white; border: 1px dashed {BORDER}; border-radius: 12px;
            }}
            QLabel#HistEmptyIcon {{ font-size: 40px; }}
            QLabel#HistEmptyMsg {{ color: #6B8890; font-size: 13px; }}
            QPushButton#HistPrimary {{
                background: {PRIMARY}; color: #0A2E33; border: none;
                border-radius: 9px; padding: 9px 18px; font-weight: 700; min-height: 20px;
            }}
            QPushButton#HistPrimary:hover {{ background: {PRIMARY_HOVER}; }}
            QPushButton#HistPrimary:disabled {{ background: #D7E6E8; color: #9BB3B8; }}
            QPushButton#HistGhost {{
                background: transparent; color: {TEXT}; border: 1px solid {BORDER};
                border-radius: 9px; padding: 9px 18px; font-weight: 600;
            }}
            QPushButton#HistGhost:hover {{ background: {SECONDARY}; }}
        """

    def _load_and_show(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "entries" in data:
                entries = data.get("entries", [])
            elif isinstance(data, list):
                entries = data
            else:
                entries = []
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Gagal membuka transkrip: {exc}")
            return
        # Tutup window riwayat, tampilkan di window utama (dengan label source).
        main = self.parent()
        if main is not None and hasattr(main, "show_history_entries"):
            main.show_history_entries(entries)
            self.accept()
        else:  # fallback bila tanpa parent window utama
            SummaryWindow({"entries": entries}, parent=self).exec()

    def _open_selected(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        self._load_and_show(item.data(QtCore.Qt.ItemDataRole.UserRole))


class CompactWidget(QtWidgets.QWidget):
    # Sinyal error dari worker thread (start/stop) -> ditangani di main thread.
    # (msg, revert_recording): revert=True mengembalikan tombol ke keadaan "Start".
    error_occurred = QtCore.Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ListenPLN")
        self.setWindowIcon(load_icon("app.ico"))
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(600, 300) # Tinggi ditambah sedikit untuk menampung bottom bar
        self.recording = False
        # Config provider (SSOT) + opsi awal dari bundled defaults + QSettings user.
        self._config = _build_config_provider()
        self.current_options: Optional[UiOptions] = _options_from_config(self._config)
        self.llm_api_key: Optional[str] = None
        # F3: mtime transcript terakhir agar refresh tidak re-parse/re-render bila tak berubah.
        self._last_transcript_mtime: float | None = None
        # W4/W2: dialog kegagalan permanen hanya sekali per sesi (refresh berjalan 500ms).
        self._outdated_notified = False
        self._tls_notified = False
        self._mic_notified = False
        # W3: readiness mikrofon (health check startup) ditampilkan lewat tombol Mute Mic.
        self._mic_ready = True
        self._mic_not_ready_reason = ""
        # Menampilkan riwayat di window utama (bukan window baru): tangguhkan render live.
        self._viewing_history = False
        # Entri riwayat yang sedang ditampilkan; jadi sumber Export saat mode riwayat
        # (bukan log transkrip live) agar isi yang diekspor = yang tampil di layar.
        self._history_entries: list = []

        self.setObjectName("MainWindow")

        self._build_ui()
        self._apply_styles()

        # F4: error dari thread latar disurunkan ke UI (QueuedConnection lintas-thread).
        self.error_occurred.connect(self._on_error)

        # W3: health check mikrofon dijalankan setelah startup selesai (0 ms = setelah
        # event loop mulai), agar jendela tampil dulu dan device tidak ditahan terbuka.
        QtCore.QTimer.singleShot(0, self.run_mic_health_check)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def _build_ui(self):
        # Layout utama dengan margin 0 agar bar atas/bawah bisa menempel penuh ke tepi (Full Width)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0) # Hilangkan spacing agar antar panel menyatu rapi

        # --- Top Control Bar (Header) ---
        self.top_panel = QtWidgets.QWidget()
        self.top_panel.setObjectName("TopBar") 
        top_layout = QtWidgets.QHBoxLayout(self.top_panel)
        top_layout.setContentsMargins(16, 12, 16, 12) # Padding untuk tombol di dalam bar
        top_layout.setSpacing(16)

        self.record_btn = QtWidgets.QPushButton(" Start Recording")
        self.record_btn.setObjectName("RecordBtn")

        self.record_btn.setIcon(load_icon("play.svg")) # Ikon awal (belum merekam)
        self.record_btn.setIconSize(QtCore.QSize(18, 18))

        self.record_btn.setCheckable(True)
        self.record_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.record_btn.setMinimumSize(150, 42)
        self.record_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.record_btn.clicked.connect(self.toggle_record)
        # Aksesibilitas & UX: tooltip, shortcut keyboard, dan nama untuk screen reader
        self.record_btn.setToolTip("Mulai / hentikan perekaman (Ctrl+R)")
        self.record_btn.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.record_btn.setAccessibleName("Tombol Mulai/Hentikan Rekaman")

        # F1: indikator status koneksi server (dipelihara engine dari sinyal subprocess).
        self.conn_label = QtWidgets.QLabel("● Disconnected")
        self.conn_label.setStyleSheet("color: #94A3B8; font-weight: 600;")
        self.conn_label.setToolTip("Status koneksi server")
        self.conn_label.setAccessibleName("Status koneksi server")

        # Container untuk VU Meter (Ubah menjadi Horizontal Layout)
        vu_layout = QtWidgets.QHBoxLayout()
        vu_layout.setContentsMargins(10, 0, 10, 0) # Beri sedikit jarak kiri-kanan
        vu_layout.setSpacing(8) # Jarak antar elemen di dalam VU layout

        # Indikator Mic
        self.mic_label = QtWidgets.QLabel("🎤 Mic")
        self.mic_label.setToolTip("Level input mikrofon")
        self.mic_vu = QtWidgets.QProgressBar()
        self.mic_vu.setRange(0, 100)
        self.mic_vu.setFixedHeight(12)
        self.mic_vu.setFixedWidth(60) # Gunakan FixedWidth agar ukurannya konsisten dan tidak terlalu panjang
        self.mic_vu.setToolTip("Level input mikrofon")
        self.mic_vu.setAccessibleName("Level mikrofon")

        # Indikator Speaker
        self.spk_label = QtWidgets.QLabel("🔊 Spk")
        self.spk_label.setToolTip("Level audio meeting/speaker")
        self.spk_vu = QtWidgets.QProgressBar()
        self.spk_vu.setRange(0, 100)
        self.spk_vu.setFixedHeight(12)
        self.spk_vu.setFixedWidth(60)
        self.spk_vu.setToolTip("Level audio meeting/speaker")
        self.spk_vu.setAccessibleName("Level speaker")

        # Susun menyamping (kiri ke kanan)
        vu_layout.addWidget(self.mic_label, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        vu_layout.addWidget(self.mic_vu, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        
        vu_layout.addSpacing(16) # Memberikan jarak ekstra pemisah antara grup Mic dan grup Spk
        
        vu_layout.addWidget(self.spk_label, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        vu_layout.addWidget(self.spk_vu, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.mute_btn = QtWidgets.QPushButton(" Mute Mic")
        self.mute_btn.setObjectName("MuteBtn")

        self.mute_btn.setIcon(load_icon("mic.svg"))
        self.mute_btn.setIconSize(QtCore.QSize(18, 18))

        self.mute_btn.setCheckable(True)
        self.mute_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.mute_btn.setMinimumSize(110, 42)
        self.mute_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.mute_btn.clicked.connect(self.toggle_mute)
        self.mute_btn.setToolTip("Bisukan / aktifkan mikrofon (Ctrl+M)")
        self.mute_btn.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        self.mute_btn.setAccessibleName("Tombol Bisukan Mikrofon")
        self.muted = False
        
        # shadow untuk tombol record dan mute
        self._apply_shadow(self.record_btn)
        self._apply_shadow(self.mute_btn)

        # layout atas
        top_layout.addWidget(self.record_btn)
        top_layout.addWidget(self.conn_label, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        top_layout.addLayout(vu_layout)
        top_layout.addStretch(1)
        top_layout.addWidget(self.mute_btn)

        # ==========================================
        # --- Live Preview Area (Tengah) ---
        # ==========================================
        # Kontainer khusus untuk bagian tengah agar memiliki jarak (padding) yang estetik
        self.center_container = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(self.center_container)
        center_layout.setContentsMargins(16, 16, 16, 16)
        
        self.preview = QtWidgets.QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Live transcript will appear here...")
        self.preview.setAccessibleName("Area transkrip langsung")
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.preview)

        # ==========================================
        # --- Bottom Control Bar (Footer) ---
        # ==========================================
        self.bottom_panel = QtWidgets.QWidget()
        self.bottom_panel.setObjectName("BottomBar")
        bottom_layout = QtWidgets.QHBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(16, 12, 16, 12) # Padding untuk tombol di dalam bar
        bottom_layout.setSpacing(10)

        self.history_btn = QtWidgets.QPushButton(" History")
        self.history_btn.setObjectName("SecondaryBtn")
        self.history_btn.setIcon(load_icon("history.svg")) # Ikon SVG
        self.history_btn.setIconSize(QtCore.QSize(18, 18))
        self.history_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.history_btn.setMinimumSize(90, 36)
        self.history_btn.clicked.connect(self.open_history)
        self.history_btn.setToolTip("Buka riwayat transkrip dari file JSON (Ctrl+H)")
        self.history_btn.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        self.history_btn.setAccessibleName("Tombol Buka Riwayat")

        self.resume_btn = QtWidgets.QPushButton(" Summarize")
        self.resume_btn.setObjectName("ActionBtn")
        self.resume_btn.setIcon(load_icon("star.svg")) # Ikon SVG
        self.resume_btn.setIconSize(QtCore.QSize(18, 18))
        self.resume_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.resume_btn.setMinimumSize(110, 36)
        self.resume_btn.clicked.connect(self.resume_summary)
        self.resume_btn.setEnabled(False)
        self.resume_btn.setToolTip("Ringkas transkrip (memerlukan LLM API key di Settings)")
        self.resume_btn.setAccessibleName("Tombol Ringkas Transkrip")

        self.export_btn = QtWidgets.QPushButton(" Export")
        self.export_btn.setObjectName("SecondaryBtn")
        self.export_btn.setIcon(load_icon("download.svg")) # Ikon SVG
        self.export_btn.setIconSize(QtCore.QSize(18, 18))
        self.export_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.export_btn.setMinimumSize(90, 36)
        self.export_btn.clicked.connect(self.export_transcript)
        self.export_btn.setToolTip("Ekspor transkrip ke file .txt atau .md (Ctrl+E)")
        self.export_btn.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        self.export_btn.setAccessibleName("Tombol Ekspor Transkrip")

        self.settings_btn = QtWidgets.QPushButton(" Settings")
        self.settings_btn.setObjectName("SecondaryBtn")
        self.settings_btn.setIcon(load_icon("settings.svg")) # Ikon SVG
        self.settings_btn.setIconSize(QtCore.QSize(18, 18))
        self.settings_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.settings_btn.setMinimumSize(110, 36)
        self.settings_btn.clicked.connect(self.open_settings)
        self.settings_btn.setToolTip("Buka pengaturan aplikasi (Ctrl+,)")
        self.settings_btn.setShortcut(QtGui.QKeySequence("Ctrl+,"))
        self.settings_btn.setAccessibleName("Tombol Buka Pengaturan")

        # SHADOW
        self._apply_shadow(self.history_btn)
        self._apply_shadow(self.resume_btn)
        self._apply_shadow(self.export_btn)
        self._apply_shadow(self.settings_btn)
        
        # Status badge: menampilkan jumlah stable dan candidate
        # self.stability_label = QtWidgets.QLabel("✓ 0 stable  ~ 0 candidate")
        # self.stability_label.setObjectName("StabilityBadge")
        # self.stability_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight)
        # self.stability_label.setToolTip("Jumlah segmen transkripsi: ✓ final (stable) vs ~ hipotesis (candidate)")

        # layout bawah
        bottom_layout.addWidget(self.history_btn)
        bottom_layout.addWidget(self.resume_btn)
        bottom_layout.addWidget(self.export_btn)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.settings_btn)

        # ==========================================
        # --- Susun Semua ke Layout Utama ---
        # ==========================================
        main_layout.addWidget(self.top_panel)
        main_layout.addWidget(self.center_container, 1) # Tambahkan angka 1 agar area teks bisa memuai (expand) maksimal
        main_layout.addWidget(self.bottom_panel)

    def _apply_styles(self):
        self.setStyleSheet(f"""
            #MainWindow {{
                background-color: {BG};
            }}
            
            /* Header Full Width */
            QWidget#TopBar {{
                background-color: white;
                border-bottom: 1px solid {BORDER};
            }}

            /* Footer Full Width */
            QWidget#BottomBar {{
                background-color: white;
                border-top: 1px solid {BORDER};
            }}

            /* Dialog Settings & Summary - pastikan tetap tema terang, tidak ikut dark mode OS */
            QDialog {{
                background-color: {BG};
                color: {TEXT};
            }}

            QTabWidget::pane {{
                background-color: white;
                border: 1px solid {BORDER};
                border-radius: 6px;
                top: -1px;
            }}
            QTabBar::tab {{
                background-color: {SECONDARY};
                color: {TEXT};
                padding: 6px 14px;
                border: 1px solid {BORDER};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background-color: white;
                font-weight: 600;
            }}
            QTabBar::tab:hover {{
                background-color: {SECONDARY_HOVER};
            }}

            /* Field input pada form Settings (sebelumnya tidak di-style -> ikut palet gelap OS) */
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: white;
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 4px 6px;
                selection-background-color: {PRIMARY};
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {PRIMARY};
            }}
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background-color: #E2E8F0;
                color: #94A3B8;
            }}
            QComboBox QAbstractItemView {{
                background-color: white;
                color: {TEXT};
                border: 1px solid {BORDER};
                selection-background-color: {PRIMARY};
                selection-color: {TEXT};
            }}
            QCheckBox {{
                color: {TEXT};
                font-weight: 500;
                spacing: 6px;
            }}

            QLabel {{
                color: {TEXT};
                font-weight: 600;
                font-size: 12px;
            }}
            
            /* General Button Styling */
            QPushButton {{
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid {BORDER};
                padding: 6px 16px;
            }}
            
            QPushButton:disabled,
            QPushButton#RecordBtn:disabled,
            QPushButton#MuteBtn:disabled,
            QPushButton#ActionBtn:disabled,
            QPushButton#SecondaryBtn:disabled {{
                background-color: #E2E8F0;
                color: #94A3B8;
                border: 1px solid #CBD5E1;
            }}

            /* Record Button (Primary) */
            QPushButton#RecordBtn {{
                background-color: {PRIMARY};
                color: #0A2E33;
                border: none;
            }}
            QPushButton#RecordBtn:hover {{
                background-color: {PRIMARY_HOVER};
            }}
            QPushButton#RecordBtn:checked {{
                background-color: {DANGER};
                color: white;
            }}

            /* Mute Button (Warning/Toggle) */
            QPushButton#MuteBtn {{
                background-color: white;
                color: {TEXT};
            }}
            QPushButton#MuteBtn:checked {{
                background-color: #FDEDEF;
                color: {DANGER};
                border: 1px solid {DANGER};
            }}
            /* Mikrofon tidak siap: tombol yang sama dipakai sebagai indikator,
               tanpa menambah widget baru. */
            QPushButton#MuteBtn[micState="unavailable"] {{
                background-color: #FEF3C7;
                color: #92400E;
                border: 1px solid #F59E0B;
            }}

            /* Action Button */
            QPushButton#ActionBtn {{
                background-color: white;
                color: #6AD8E5;
            }}
            QPushButton#ActionBtn:hover {{
                background-color: #EEF2FF;
            }}

            /* Secondary Buttons */
            QPushButton#SecondaryBtn {{
                background-color: white;
                color: {TEXT};
            }}
            QPushButton#SecondaryBtn:hover {{
                background-color: #E3FAFC;
                color: {TEXT};
            }}

            /* Text Area (Tengah) */
            QPlainTextEdit {{
                background-color: white;
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.5;
                color: #334155;
            }}

            /* VU Meter */
            QProgressBar {{
                background-color: transparent;
                border: none;
                text-align: center;
                color: transparent;
                min-width: 60px;
            }}
            QProgressBar::chunk {{
                background-color: #3A8EEB;
                width: 4px;
                margin-right: 2px;
                border-radius: 2px;
            }}

            QTextEdit {{
                background-color: white;
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.5;
                color: #334155;
            }}
            
            /* Styling untuk tombol Save */
            QPushButton#SaveBtn {{
                background-color: {PRIMARY};
                color: #0A2E33;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 16px;
            }}
            QPushButton#SaveBtn:hover {{
                background-color: {PRIMARY_HOVER};
            }}
            
            /* Styling untuk tombol Cancel */
            QPushButton#CancelBtn {{
                background-color: white;
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 16px;
            }}
            QPushButton#CancelBtn:hover {{
                background-color: #FEF2F2;
                color: {DANGER};
                border-color: {DANGER};
            }}
        """)

    def open_settings(self):
        opts = self.current_options or UiOptions()
        # Rekam binding perangkat SEBELUM dialog, untuk membedakan "config server
        # berubah" dari "perangkat audio berubah" (lihat _apply_settings_result).
        prev_devices = (getattr(opts, "mic_device", None), getattr(opts, "speaker_device", None))
        dlg = SettingsDialog(self, options=opts)
        
        # Populate the LLM API key field if previously set
        if getattr(self, 'llm_api_key', None):
            dlg.llm_api_key.setText(self.llm_api_key)
            
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.current_options = dlg.to_options()
            # Domain adaptation tidak di-edit di UI: bawa dari config provider.
            self.current_options.initial_prompt = self._config.initial_prompt()
            self.current_options.hotwords = self._config.hotwords()
            # W2: TLS tidak di-edit di UI. Tanpa baris ini, to_options() (yang membuat
            # UiOptions baru) akan MEMATIKAN TLS diam-diam setelah Settings dibuka.
            self.current_options.use_tls = self._config.use_tls()
            # Persist pengaturan user ke QSettings (persisten lintas restart).
            try:
                self._config.set_user("server_host", self.current_options.host)
                self._config.set_user("server_port", self.current_options.port)
                self._config.set_user("model", self.current_options.model)
                self._config.set_user("language", self.current_options.language)
            except Exception as exc:
                print("failed to persist settings:", exc)
            try:
                self._preferred_source = getattr(self.current_options, "source", "both") or "both"
            except Exception:
                self._preferred_source = "both"
            
            self.llm_api_key = dlg.llm_api_key.text().strip() or None
            self.resume_btn.setEnabled(bool(self.llm_api_key))
            # Validasi audio HANYA bila pilihan perangkat benar-benar berubah.
            # Mengubah host/port/model/bahasa tidak boleh menyentuh subsistem audio.
            new_devices = (
                getattr(self.current_options, "mic_device", None),
                getattr(self.current_options, "speaker_device", None),
            )
            if new_devices != prev_devices:
                self.run_mic_health_check()

    def toggle_record(self, checked: bool):
        # W3: gate permission sebelum sesi dijalankan, agar mikrofon yang diblokir
        # menghasilkan panduan yang jelas — bukan sesi yang berjalan senyap.
        if checked and not self._ensure_microphone_permission():
            self.record_btn.setChecked(False)  # `clicked` -> setChecked tidak re-emit
            self._apply_record_button_state(recording=False)
            return
        self._apply_record_button_state(recording=checked)
        if checked:
            # Mulai rekaman -> keluar dari mode lihat-riwayat, kembali ke live view.
            self._viewing_history = False
            self._last_transcript_mtime = None
            self._start_live()
        else:
            self._stop_live()

    def show_history_entries(self, entries: list) -> None:
        """Tampilkan riwayat terpilih di window utama (bukan window baru).

        Menahan render live agar tidak menimpa; format (label Mic/Speaker) sama
        dengan live view. Mulai rekaman baru mengembalikan tampilan ke live.
        """
        self._viewing_history = True
        self._history_entries = list(entries)
        self.preview.setHtml(_entries_to_html(entries))
        scrollbar = self.preview.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(0)

    def _abort_session_and_warn(self, title: str, body: str, *, link: str = "") -> None:
        """Kegagalan permanen sesi: hentikan rekaman lalu jelaskan sekali.

        Dipakai jalur yang mencoba ulang tidak akan pernah berhasil sampai ada
        tindakan di luar aplikasi (versi usang W4, verifikasi TLS gagal W2).
        """
        if self.record_btn.isChecked():
            self.record_btn.setChecked(False)
            self._apply_record_button_state(recording=False)
            self._stop_live()

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        if link:
            box.setTextFormat(QtCore.Qt.TextFormat.RichText)
            box.setText(body.replace("\n", "<br>") + f'<br><br><a href="{link}">{link}</a>')
            box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        else:
            box.setText(body)
        box.exec()

    def _check_outdated_client(self) -> None:
        """W4: server menolak versi ini -> hentikan sesi dan arahkan pengguna mengunduh."""
        info = outdated_client_info()
        if info is None or self._outdated_notified:
            return
        self._outdated_notified = True
        minimum = info.get("min_version") or "-"
        self._abort_session_and_warn(
            "Pembaruan Diperlukan",
            f"Versi aplikasi Anda ({info.get('client_version')}) sudah tidak didukung server. "
            f"Versi minimum yang diperlukan: {minimum}.\n\n"
            "Rekaman tidak dapat dijalankan sampai aplikasi diperbarui.\n\nUnduh versi terbaru:",
            link=self._config.download_url(),
        )

    def _check_tls_error(self) -> None:
        """W2: koneksi aman gagal -> hentikan sesi dan jelaskan sebabnya.

        Tanpa ini pengguna hanya melihat indikator "Error" tanpa tahu bahwa
        masalahnya ada pada sertifikat server (tindakan ada di sisi IT, bukan pengguna).
        """
        info = tls_error_info()
        if info is None or self._tls_notified:
            return
        self._tls_notified = True
        reason = info.get("reason") or "verifikasi sertifikat gagal"
        self._abort_session_and_warn(
            "Koneksi Aman Gagal",
            f"Aplikasi tidak dapat membuat koneksi aman ke server ({info.get('url') or '-'}).\n\n"
            f"Penyebab: {reason}.\n\n"
            "Rekaman dihentikan demi menjaga kerahasiaan rapat. Hubungi tim IT/administrator "
            "server Anda — sertifikat server perlu diperbaiki."
        )

    def _check_mic_error(self) -> None:
        """W3: subprocess gagal membuka mikrofon -> hentikan sesi dan beri panduan.

        Windows TIDAK menampilkan dialog izin untuk aplikasi desktop Win32; satu-satunya
        jalan bagi pengguna adalah Setelan Privasi. Karena itu kegagalan ini harus
        dijelaskan, bukan dibiarkan senyap.
        """
        info = mic_error_info()
        if info is None or self._mic_notified:
            return
        self._mic_notified = True
        access = check_microphone(getattr(self.current_options or UiOptions(), "mic_device", None))
        detail = guidance_message(access) or (
            "Aplikasi tidak dapat membuka mikrofon.\n\nPenyebab: "
            f"{info.get('reason') or 'tidak diketahui'}."
        )
        self._abort_session_and_warn("Mikrofon Tidak Dapat Digunakan", detail)
        self.run_mic_health_check()

    def _ensure_microphone_permission(self) -> bool:
        """True bila rekaman boleh jalan (source tanpa mic, atau mikrofon siap)."""
        opts = self.current_options or UiOptions()
        if not source_uses_microphone(getattr(opts, "source", "both")):
            return True
        return ensure_microphone_ready(self, device=getattr(opts, "mic_device", None))

    def _apply_record_button_state(self, recording: bool) -> None:
        if recording:
            self.record_btn.setText(" Stop Recording")
            self.record_btn.setIcon(load_icon("stop.svg"))
            self.record_btn.setToolTip("Hentikan perekaman (Ctrl+R)")
        else:
            self.record_btn.setText(" Start Recording")
            self.record_btn.setIcon(load_icon("play.svg"))
            self.record_btn.setToolTip("Mulai perekaman (Ctrl+R)")

    def _on_error(self, message: str, revert_recording: bool) -> None:
        # F4: berjalan di main thread (via Signal). Tampilkan error + pulihkan tombol
        # agar UI tidak menampilkan "Stop Recording" padahal sesi gagal berjalan.
        if revert_recording:
            self.record_btn.setChecked(False)
            self._apply_record_button_state(recording=False)
        QtWidgets.QMessageBox.warning(self, "Error", message)

    def _update_connection_indicator(self) -> None:
        # F1: konsumsi status koneksi yang dipelihara engine.
        status = connection_status()
        color, text = {
            "CONNECTED": ("#059669", "Connected"),
            "CONNECTING": ("#F59E0B", "Connecting…"),
            "ERROR": ("#E84D5B", "Error"),
            "DISCONNECTED": ("#94A3B8", "Disconnected"),
        }.get(status, ("#94A3B8", status))
        self.conn_label.setText(f"● {text}")
        self.conn_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.conn_label.setToolTip(f"Status koneksi server: {status}")

    @staticmethod
    def _db_to_pct(rms_db: Optional[float]) -> int:
        # Petakan level RMS -60..0 dB -> 0..100%. None (tak ada level) -> 0.
        if rms_db is None:
            return 0
        pct = (rms_db + 60.0) / 60.0 * 100.0
        return max(0, min(100, int(pct)))

    def _update_audio_meters(self) -> None:
        # Indikator Mic/Speaker dari level audio NYATA (RMS input subprocess),
        # bukan proxy stabilitas transkrip. Kosong saat tidak merekam -> 0%.
        levels = audio_levels()
        mic_pct = self._db_to_pct(levels.get("mic"))
        spk_pct = self._db_to_pct(levels.get("speaker"))
        self.mic_vu.setValue(mic_pct)
        self.spk_vu.setValue(spk_pct)
        self.mic_vu.setToolTip(f"Level mikrofon: {mic_pct}%")
        self.spk_vu.setToolTip(f"Level speaker: {spk_pct}%")

    def run_mic_health_check(self) -> None:
        """Health check mikrofon sekali jalan; device dibuka lalu SEGERA ditutup.

        Windows tidak menyediakan dialog izin untuk aplikasi desktop Win32, jadi
        readiness harus dideteksi sendiri. Hasilnya ditampilkan lewat tombol Mute Mic
        (tanpa menambah indikator UI baru), bukan dengan menahan device tetap terbuka.
        """
        opts = self.current_options or UiOptions()
        if not source_uses_microphone(getattr(opts, "source", "both")):
            self._mic_ready = True
            self._apply_mute_button_state()
            return
        access = check_microphone(getattr(opts, "mic_device", None))
        self._mic_ready = guidance_kind(access) == GUIDANCE_NONE
        self._mic_not_ready_reason = guidance_message(access)
        self._apply_mute_button_state()

    def _apply_mute_button_state(self) -> None:
        """Tampilkan state mute ATAU ketidaksiapan mikrofon pada satu tombol yang sama."""
        if not self._mic_ready:
            # Status DITAMPILKAN, kontrol TIDAK diubah: label dan arti klik tetap
            # "mute", agar satu tombol tidak berganti makna diam-diam. Jalur
            # perbaikan disediakan di Start Recording (validasi otoritatif).
            self.mute_btn.setText(" Muted" if self.muted else " Mute Mic")
            self.mute_btn.setIcon(load_icon("micoff.svg" if self.muted else "mic.svg"))
            self.mute_btn.setToolTip(
                "Mikrofon belum dapat digunakan.\n\n"
                + (self._mic_not_ready_reason or "Penyebab tidak diketahui.")
                + "\n\nTekan Start Recording untuk melihat langkah perbaikannya."
            )
            self.mute_btn.setProperty("micState", "unavailable")
        elif self.muted:
            self.mute_btn.setText(" Muted")
            self.mute_btn.setIcon(load_icon("micoff.svg"))
            self.mute_btn.setToolTip("Aktifkan kembali mikrofon (Ctrl+M)")
            self.mute_btn.setProperty("micState", "muted")
        else:
            self.mute_btn.setText(" Mute Mic")
            self.mute_btn.setIcon(load_icon("mic.svg"))
            self.mute_btn.setToolTip("Bisukan mikrofon (Ctrl+M)")
            self.mute_btn.setProperty("micState", "ready")
        # Refresh QSS agar perubahan property ikut ter-render.
        self.mute_btn.style().unpolish(self.mute_btn)
        self.mute_btn.style().polish(self.mute_btn)

    def toggle_mute(self, checked: bool):
        self.muted = bool(checked)
        self._apply_mute_button_state()

        # True mute: terapkan ke sesi yang sedang berjalan tanpa restart koneksi.
        if self.record_btn.isChecked():
            self._apply_mute()

    def _apply_mute(self):
        muted = self.muted

        def _runner():
            try:
                if not set_mute("mic", muted):
                    self.error_occurred.emit("Gagal menerapkan mute: sesi live tidak aktif.", False)
            except Exception as exc:
                self.error_occurred.emit(f"Gagal menerapkan mute: {exc}", False)

        threading.Thread(target=_runner, daemon=True).start()

    def _start_live(self):
        # Sesi baru -> penolakan versi berikutnya harus tetap diberitahukan
        # (engine juga me-reset detailnya saat start).
        self._outdated_notified = False
        self._tls_notified = False
        self._mic_notified = False
        opts = self.current_options or UiOptions()
        opts.host = getattr(opts, "host", "localhost")
        opts.port = getattr(opts, "port", 9090)

        def _runner():
            try:
                start_live(opts)
            except Exception as exc:
                self.error_occurred.emit(f"Gagal memulai sesi: {exc}", True)
                return
            # Sesi dimulai dalam keadaan mute: terapkan lewat kanal kontrol, bukan
            # dengan mengubah source (yang dulu memaksa restart).
            if self.muted:
                set_mute("mic", True)

        threading.Thread(target=_runner, daemon=True).start()

    def _stop_live(self):
        def _runner():
            try:
                stop_live()
                # Sesi selesai & transkrip sudah ter-drain -> arsipkan ke history
                # (satu berkas per rapat). Transkrip kosong tidak diarsipkan.
                try:
                    archive_transcript_to_history()
                except Exception as exc:
                    print("failed to archive transcript to history:", exc)
            except Exception as exc:
                self.error_occurred.emit(f"Gagal menghentikan sesi: {exc}", False)

        threading.Thread(target=_runner, daemon=True).start()
    
    def _apply_shadow(self, widget):
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)       # Seberapa blur bayangannya (semakin besar semakin halus)
        shadow.setXOffset(0)           # Geser bayangan horizontal (0 = di tengah)
        shadow.setYOffset(3)           # Geser bayangan vertikal (bayangan jatuh ke bawah)
        
        # Mengatur warna hitam transparan (R, G, B, Alpha/Opacity)
        shadow.setColor(QtGui.QColor(0, 0, 0, 25)) 
        
        # Menerapkan efek ke widget
        widget.setGraphicsEffect(shadow)

    def open_history(self):
        # Riwayat dibaca dari folder khusus per-user (bukan dialog file kosong).
        HistoryDialog(self).exec()

    def resume_summary(self):
        if not self.llm_api_key:
            QtWidgets.QMessageBox.information(self, "Resume disabled", "Provide an LLM API key in Settings to enable remote summarization.")
            return
        try:
            data = transcript_payload()
            entries = data.get("entries", [])
            texts = [e.get("text", "") for e in entries if e.get("text")]
            sample = texts[-20:]
            if not sample:
                QtWidgets.QMessageBox.information(self, "No transcript", "No transcript data to summarize.")
                return
            summary_text = "\n".join(sample)
            summary_note = "(Local heuristic summary — LLM backend integration not implemented)\n\n"
            SummaryWindow({"entries": [{"text": summary_note + summary_text}]}, parent=self).exec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to produce summary: {exc}")

    def export_transcript(self):
        try:
            # Mode riwayat: ekspor entri yang sedang ditampilkan, bukan log live
            # (yang mungkin kosong atau berisi sesi lain).
            if self._viewing_history:
                entries = self._history_entries
            else:
                entries = transcript_payload().get("entries", [])
            if not entries:
                QtWidgets.QMessageBox.information(self, "No transcript", "No transcript data to export.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export transcript", "transcript.pdf", "PDF (*.pdf)",
            )
            if not path:
                return
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            _export_transcript_pdf(path, entries)
            QtWidgets.QMessageBox.information(self, "Exported", f"Transcript exported to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to export: {exc}")

    def refresh(self):
        # F1: status koneksi + indikator level audio diperbarui tiap tick (murah),
        # lepas dari transcript (level bergerak lebih cepat dari file transkrip).
        try:
            self._update_connection_indicator()
            self._update_audio_meters()
            self._check_outdated_client()
            self._check_tls_error()
            self._check_mic_error()
        except Exception:
            pass

        # F3: lewati re-read/re-render transcript bila file tidak berubah sejak tick lalu.
        try:
            mtime = os.path.getmtime(DEFAULT_TRANSCRIPT_LOG) if os.path.exists(DEFAULT_TRANSCRIPT_LOG) else None
        except OSError:
            mtime = None
        # Sedang menampilkan riwayat di window utama -> jangan ditimpa transkrip live.
        if self._viewing_history:
            return
        if mtime is not None and mtime == self._last_transcript_mtime:
            return
        self._last_transcript_mtime = mtime

        try:
            data = transcript_payload()
            entries = data.get("entries", [])

            # Hitung jumlah stable dan candidate dari audit_entries (semua entri)
            audit_entries = data.get("audit_entries", entries)
            stable_count = sum(1 for e in audit_entries if str(e.get("stability") or "") == "stable")
            candidate_count = sum(1 for e in audit_entries if str(e.get("stability") or "") == "candidate")
            
            # Amankan posisi scroll jika pengguna sedang scroll ke atas
            scrollbar = self.preview.verticalScrollBar()
            was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10 if scrollbar else True

            # Render HTML (label source bersama dengan tampilan history)
            self.preview.setHtml(_entries_to_html(entries))
            
            if was_at_bottom and scrollbar:
                scrollbar.setValue(scrollbar.maximum())

            # Indikator Mic/Speaker kini digerakkan level audio nyata di
            # _update_audio_meters() (tiap tick), bukan dari rasio transkrip.

            # Update badge stable/candidate (commented out)
            # badge_text = f"✓ {stable_count} stable  ~ {candidate_count} candidate"
            # self.stability_label.setText(badge_text)
            # if stable_count == 0 and candidate_count == 0:
            #     self.stability_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
            # elif candidate_count > stable_count * 2:
            #     self.stability_label.setStyleSheet("color: #F59E0B; font-size: 11px; font-weight: bold;")
            # else:
            #     self.stability_label.setStyleSheet("color: #059669; font-size: 11px;")

        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Menghitung ukuran font responsif berdasarkan lebar jendela, agar teks
        # transkrip tetap nyaman dibaca baik di jendela sempit maupun lebar.
        base_width = 600
        base_font = 12
        extra_width = max(0, self.width() - base_width)
        new_size = base_font + int(extra_width / 50)
        new_size = min(new_size, 21)  # Batasi ukuran maksimal font agar tidak terlalu raksasa

        # Hindari pemanggilan setStyleSheet berulang saat ukurannya tidak berubah
        # (setStyleSheet relatif berat & bisa memicu flicker saat resize di-drag terus-menerus).
        if getattr(self, "_last_preview_font_size", None) == new_size:
            return
        self._last_preview_font_size = new_size
        self.preview.setStyleSheet(f"font-size: {new_size}px;")


def run_gui() -> int:
    # --- Dukungan High-DPI ---
    # Di Qt6, High-DPI scaling sudah AKTIF SECARA OTOMATIS (berbeda dari Qt5 yang
    # memerlukan AA_EnableHighDpiScaling/AA_UseHighDpiPixmaps secara eksplisit -
    # atribut tersebut sudah dihapus di PyQt6 dan akan error bila dipanggil).
    # Yang masih relevan diatur adalah kebijakan pembulatan faktor skala, agar UI
    # tetap tajam & proporsional di layar dengan scaling non-integer (mis. 125%, 150%).
    if hasattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy"):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QtWidgets.QApplication(sys.argv)

    # Menetapkan ikon aplikasi secara global agar muncul di Taskbar & semua Window
    app_icon = load_icon("app.ico")
    app.setWindowIcon(app_icon)

    # Khusus Windows: Supaya taskbar Windows tidak mengelompokkan aplikasi ke dalam grup generic 'Python'
    # dan menampilkan logo custom kita secara konsisten pada taskbar.
    if sys.platform == "win32":
        import ctypes
        myappid = "pln.listenpln.client.1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Paksa skema warna TERANG secara eksplisit. Sejak Qt 6.5, Qt secara otomatis
    # mengikuti tema gelap/terang sistem operasi (mis. dark mode Windows) untuk
    # palet default widget yang belum di-style secara eksplisit (QLineEdit, QComboBox,
    # QSpinBox, dsb di dialog Settings). Tanpa baris ini, aplikasi bisa tiba-tiba
    # terlihat gelap di komputer yang OS-nya diset dark mode, padahal palet warna
    # aplikasi ini memang dirancang untuk tema terang.
    if hasattr(app, "styleHints") and hasattr(QtCore.Qt, "ColorScheme"):
        try:
            app.styleHints().setColorScheme(QtCore.Qt.ColorScheme.Light)
        except Exception:
            pass

    # Style dasar "Fusion" dipilih karena konsisten lintas platform (Windows/macOS/Linux)
    # dan merender QSS custom (warna, border-radius, dsb.) jauh lebih akurat
    # dibanding style native masing-masing OS. Warna & layout aplikasi tidak berubah,
    # ini hanya mempengaruhi rendering dasar widget yang belum di-style eksplisit.
    app.setStyle("Fusion")

    # W3: onboarding first-run (Welcome -> Izin Mikrofon -> Selesai) sebelum jendela
    # utama tampil. Ditutup pengguna -> keluar dengan aman, tanpa jendela utama.
    if not maybe_run_first_run_wizard(_build_user_store()):
        return 0

    w = CompactWidget()
    w.show()
    return app.exec()


if __name__ == "__main__":
    # Tetap lewat dispatcher tunggal agar jalur startup konsisten.
    from src.app import main as _dispatch

    raise SystemExit(_dispatch())
