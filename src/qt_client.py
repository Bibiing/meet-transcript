from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional
import json

from PyQt6 import QtCore, QtGui, QtWidgets

from src.ui.server import (
    audio_devices_payload,
    start_live,
    stop_live,
    transcript_payload,
    UiOptions,
)


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
    path = ICON_DIR / filename
    if not path.exists():
        return QtGui.QIcon()
    return QtGui.QIcon(str(path))



class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(520, 320)
        self.setSizeGripEnabled(True)  # Memudahkan pengguna memperbesar dialog bila daftar Advanced panjang

        self.tabs = QtWidgets.QTabWidget()
        self.general = QtWidgets.QWidget()
        self.advanced = QtWidgets.QWidget()

        self.tabs.addTab(self.general, "General")
        self.tabs.addTab(self.advanced, "Advanced")

        # General layout
        g_layout = QtWidgets.QFormLayout()
        self.host_input = QtWidgets.QLineEdit("localhost")
        self.port_input = QtWidgets.QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(9090)
        self.mic_select = QtWidgets.QComboBox()
        self.spk_select = QtWidgets.QComboBox()
        self.model_select = QtWidgets.QComboBox()
        self.model_select.addItems(["tiny", "base", "small", "medium", "large"])
        self.lang_select = QtWidgets.QComboBox()
        self.lang_select.addItems(["id", "en"])
        # LLM API key for resume/summary feature (stored in memory only)
        self.llm_api_key = QtWidgets.QLineEdit()
        self.llm_api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.llm_api_key.setPlaceholderText("OpenAI / LLM API key")

        # Tooltip singkat agar pengguna paham fungsi tiap field tanpa dokumentasi terpisah
        self.host_input.setToolTip("Alamat server transcription (default: localhost)")
        self.port_input.setToolTip("Port server transcription (1-65535)")
        self.mic_select.setToolTip("Pilih perangkat mikrofon input")
        self.spk_select.setToolTip("Pilih perangkat output/speaker yang direkam")
        self.lang_select.setToolTip("Bahasa transkripsi")
        self.model_select.setToolTip("Model Whisper: makin besar makin akurat namun makin berat")
        self.llm_api_key.setToolTip("API key untuk fitur ringkasan (Summarize). Disimpan hanya di memori, tidak ditulis ke disk")

        g_layout.addRow("Server host:", self.host_input)
        g_layout.addRow("Server port:", self.port_input)
        g_layout.addRow("Microphone:", self.mic_select)
        g_layout.addRow("Speaker:", self.spk_select)
        g_layout.addRow("Language:", self.lang_select)
        g_layout.addRow("Model:", self.model_select)
        g_layout.addRow("LLM API Key:", self.llm_api_key)

        self.general.setLayout(g_layout)

        # Advanced settings (technical)
        a_layout = QtWidgets.QFormLayout()

        # VAD and normalization
        self.mic_server_vad_cb = QtWidgets.QCheckBox()
        self.mic_server_vad_cb.setChecked(True)
        self.speaker_server_vad_cb = QtWidgets.QCheckBox()
        self.speaker_server_vad_cb.setChecked(False)

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

        # Bukan pilihan (wajib)
        # Reliability / reconnect
        # self.auto_reconnect_cb = QtWidgets.QCheckBox()
        # self.auto_reconnect_cb.setChecked(True)

        # Local agreement / dynamic prompt
        # self.local_agreement_cb = QtWidgets.QCheckBox()
        # self.local_agreement_cb.setChecked(True)
        # self.dynamic_prompt_cb = QtWidgets.QCheckBox()
        # self.dynamic_prompt_cb.setChecked(True)

        # Speech boundary
        # self.speech_boundary_cb = QtWidgets.QCheckBox()
        # self.speech_boundary_cb.setChecked(True)

        self.speech_boundary_silence = QtWidgets.QDoubleSpinBox()
        self.speech_boundary_silence.setRange(0.0, 10.0)
        self.speech_boundary_silence.setValue(0.8)
        self.speech_boundary_max_wait = QtWidgets.QDoubleSpinBox()
        self.speech_boundary_max_wait.setRange(0.0, 30.0)
        self.speech_boundary_max_wait.setValue(5.0)

        # Debug/archive
        self.debug_chunk_archive_cb = QtWidgets.QCheckBox()
        self.rolling_audio_archive_cb = QtWidgets.QCheckBox()
        self.rolling_audio_segment = QtWidgets.QSpinBox()
        self.rolling_audio_segment.setRange(1, 3600)
        self.rolling_audio_segment.setValue(60)

        # Log level
        self.log_level = QtWidgets.QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])

        # Tooltip untuk parameter teknis agar pengguna non-teknis tetap paham fungsinya
        self.mic_server_vad_cb.setToolTip("Aktifkan Voice Activity Detection di server untuk audio mikrofon")
        self.speaker_server_vad_cb.setToolTip("Aktifkan Voice Activity Detection di server untuk audio speaker")
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
        a_layout.addRow("Mic target RMS (dB):", self.mic_target_rms)
        a_layout.addRow("Mic max gain (dB):", self.mic_max_gain)
        a_layout.addRow("Speaker target RMS (dB):", self.spk_target_rms)
        a_layout.addRow("Speaker max gain (dB):", self.spk_max_gain)
        a_layout.addRow("VAD threshold:", self.vad_threshold)
        a_layout.addRow("No-speech threshold:", self.no_speech_thresh)
        # a_layout.addRow("Auto reconnect:", self.auto_reconnect_cb)
        # a_layout.addRow("Local agreement:", self.local_agreement_cb)
        # a_layout.addRow("Dynamic prompt:", self.dynamic_prompt_cb)
        # a_layout.addRow("Speech boundary detection:", self.speech_boundary_cb)
        a_layout.addRow("Speech boundary silence (s):", self.speech_boundary_silence)
        a_layout.addRow("Speech boundary max wait (s):", self.speech_boundary_max_wait)
        a_layout.addRow("Debug chunk archive:", self.debug_chunk_archive_cb)
        a_layout.addRow("Rolling audio archive:", self.rolling_audio_archive_cb)
        a_layout.addRow("Rolling audio segment (s):", self.rolling_audio_segment)
        a_layout.addRow("Log level:", self.log_level)

        self.advanced.setLayout(a_layout)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Beri label dan ID khusus untuk di-styling nanti
        ok_btn = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Save Settings")
        ok_btn.setObjectName("SaveBtn")
        ok_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        ok_btn.setToolTip("Simpan pengaturan dan tutup dialog")
        ok_btn.setDefault(True)  # Enter langsung menyimpan, sesuai konvensi dialog standar

        cancel_btn = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setObjectName("CancelBtn")
        cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        cancel_btn.setToolTip("Batalkan perubahan (Esc)")

        main = QtWidgets.QVBoxLayout()
        main.addWidget(self.tabs)
        main.addWidget(btns)
        self.setLayout(main)

        self.reload_devices()

    def reload_devices(self):
        try:
            payload = audio_devices_payload()
            self.mic_select.clear()
            for d in payload.get("mic", []):
                self.mic_select.addItem(d.get("label") or d.get("name"), d.get("id"))
            self.spk_select.clear()
            for d in payload.get("speaker", []):
                self.spk_select.addItem(d.get("name"), d.get("id"))
        except Exception:
            # best-effort; ignore errors
            pass

    def to_options(self) -> UiOptions:
        opts = UiOptions()
        opts.host = self.host_input.text().strip() or "localhost"
        opts.port = int(self.port_input.value())
        mic = self.mic_select.currentData()
        spk = self.spk_select.currentData()
        opts.mic_device = mic
        opts.speaker_device = spk
        opts.language = str(self.lang_select.currentText())
        # model & LLM key
        opts.model = str(self.model_select.currentText())
        return opts


class SummaryWindow(QtWidgets.QDialog):
    def __init__(self, transcript: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transcript Summary")
        self.resize(700, 500)
        layout = QtWidgets.QVBoxLayout()
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        content = "\n".join([e.get("text", "") for e in transcript.get("entries", [])])
        self.text.setPlainText(content)
        layout.addWidget(self.text)

        btns = QtWidgets.QHBoxLayout()
        export_txt = QtWidgets.QPushButton("Export .txt")
        export_txt.clicked.connect(self.export_txt)
        btns.addStretch(1)
        btns.addWidget(export_txt)
        layout.addLayout(btns)
        self.setLayout(layout)

    def export_txt(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save transcript", "transcript.txt", "Text files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text.toPlainText())


class CompactWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLN Meeting Transcriber")
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(600, 300) # Tinggi ditambah sedikit untuk menampung bottom bar
        self.recording = False
        self.current_options: Optional[UiOptions] = None
        self.llm_api_key: Optional[str] = None

        self.setObjectName("MainWindow")

        self._build_ui()
        self._apply_styles()

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
                min-width: 110px;
                min-height: 36px;
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
                min-width: 90px;
                min-height: 36px;
            }}
            QPushButton#CancelBtn:hover {{
                background-color: #FEF2F2;
                color: {DANGER};
                border-color: {DANGER};
            }}
        """)

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.current_options = dlg.to_options()
            try:
                self._preferred_source = getattr(self.current_options, "source", "both") or "both"
            except Exception:
                self._preferred_source = "both"
            
            self.llm_api_key = dlg.llm_api_key.text().strip() or None
            self.resume_btn.setEnabled(bool(self.llm_api_key))

            try:
                self.current_options.model = str(dlg.model_select.currentText())
            except Exception:
                pass

    def toggle_record(self, checked: bool):
        if checked:
            # Saat sedang merekam (Tombol aktif)
            self.record_btn.setText(" Stop Recording") # Hapus emoji ⏹
            self.record_btn.setIcon(load_icon("stop.svg")) # Ganti ke ikon stop
            self.record_btn.setToolTip("Hentikan perekaman (Ctrl+R)")
            self._start_live()
        else:
            # Saat berhenti merekam (Tombol kembali normal)
            self.record_btn.setText(" Start Recording") # Hapus emoji ⏺
            self.record_btn.setIcon(load_icon("play.svg")) # Ganti ke ikon record
            self.record_btn.setToolTip("Mulai perekaman (Ctrl+R)")
            self._stop_live()

    def toggle_mute(self, checked: bool):
        self.muted = bool(checked)
        
        if self.muted:
            self.mute_btn.setText(" Muted")
            self.mute_btn.setIcon(load_icon("micoff.svg")) 
            self.mute_btn.setToolTip("Aktifkan kembali mikrofon (Ctrl+M)")
        else:
            self.mute_btn.setText(" Mute Mic")
            self.mute_btn.setIcon(load_icon("mic.svg"))  
            self.mute_btn.setToolTip("Bisukan mikrofon (Ctrl+M)")
            
        # Restart stream jika sedang merekam
        if self.record_btn.isChecked():
            self._restart_with_options()

    def _restart_with_options(self):
        opts = self.current_options or UiOptions()
        preferred = getattr(self, "_preferred_source", getattr(opts, "source", "both") or "both")
        opts.source = "speaker" if self.muted else preferred

        def _runner():
            try:
                stop_live()
            except Exception:
                pass
            time.sleep(0.15)
            try:
                start_live(opts)
            except Exception as exc:
                print("restart_with_options failed:", exc)

        threading.Thread(target=_runner, daemon=True).start()

    def _start_live(self):
        opts = self.current_options or UiOptions()
        opts.host = getattr(opts, "host", "localhost")
        opts.port = getattr(opts, "port", 9090)
        if self.muted:
            opts.source = "speaker"

        def _runner():
            try:
                start_live(opts)
            except Exception as exc:
                print("start_live failed:", exc)

        threading.Thread(target=_runner, daemon=True).start()

    def _stop_live(self):
        def _runner():
            try:
                stop_live()
            except Exception as exc:
                print("stop_live failed:", exc)

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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open transcript (JSON)", str(), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "entries" in data:
                entries = data.get("entries", [])
            elif isinstance(data, list):
                entries = data
            else:
                entries = []
            SummaryWindow({"entries": entries}, parent=self).exec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to open transcript: {exc}")

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
            data = transcript_payload()
            entries = data.get("entries", [])
            if not entries:
                QtWidgets.QMessageBox.information(self, "No transcript", "No transcript data to export.")
                return
            path, filt = QtWidgets.QFileDialog.getSaveFileName(self, "Export transcript", "transcript.txt", "Text files (*.txt);;Markdown (*.md)")
            if not path:
                return
            if path.endswith(".md") or filt.startswith("Markdown"):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("# Transcript\n\n")
                    for e in entries:
                        src = e.get("source") or "other"
                        text = e.get("text", "")
                        f.write(f"- **{src}**: {text}\n")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    for e in entries:
                        src = e.get("source") or "other"
                        text = e.get("text", "")
                        f.write(f"[{src}] {text}\n")
            QtWidgets.QMessageBox.information(self, "Exported", f"Transcript exported to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to export: {exc}")

    def refresh(self):
        try:
            data = transcript_payload()
            entries = data.get("entries", [])
            last = entries[-4:]
            display_html = [] # Kita gunakan list penyimpan format HTML
            
            for e in last:
                stability = e.get("stability") or ("stable" if e.get("completed", True) else "candidate")
                text = e.get("text", "")
                source = str(e.get("source") or "other")
                
                # Format label dengan Bold dan Warna berbeda
                if source == "mic":
                    # Warna Biru untuk Mic (Me)
                    label_html = '<b style="color: #2563EB;">[Me]</b>'
                elif source == "speaker":
                    # Warna Hijau untuk Speaker
                    label_html = '<b style="color: #059669;">[Speaker]</b>'
                else:
                    # Warna Abu-abu jika sumber tidak diketahui
                    label_html = f'<b style="color: #64748B;">[{source}]</b>'

                # Menyusun baris transkrip
                if stability == "stable":
                    display_html.append(f"{label_html} {text}")
                else:
                    display_html.append(f"{label_html} {text} <i style='color: #94A3B8;'>(candidate)</i>")
            
            # Gunakan setHtml dan gabungkan list menggunakan tag <br> untuk baris baru
            self.preview.setHtml("<br>".join(display_html))

            # VU meters logic (Tetap sama seperti sebelumnya)
            quality = data.get("quality", {})
            stable = quality.get("stable", 0)
            candidate = quality.get("candidate", 0)
            total = stable + candidate or 1
            level = min(100, int(candidate / total * 100) + (5 if entries else 0))
            self.mic_vu.setValue(level)
            self.spk_vu.setValue(min(100, 100 - level))
            # Tooltip dinamis agar pengguna bisa melihat angka persis, tidak hanya bar visual
            self.mic_vu.setToolTip(f"Level mikrofon: {level}%")
            self.spk_vu.setToolTip(f"Level speaker: {min(100, 100 - level)}%")
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


def main():
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

    w = CompactWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()