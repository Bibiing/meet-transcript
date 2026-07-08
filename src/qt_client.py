from __future__ import annotations

import sys
import threading
import time
from typing import Optional
import json

from PyQt5 import QtCore, QtGui, QtWidgets

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



class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(520, 320)

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
        self.llm_api_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.llm_api_key.setPlaceholderText("OpenAI / LLM API key")

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

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Beri label dan ID khusus untuk di-styling nanti
        ok_btn = btns.button(QtWidgets.QDialogButtonBox.Ok)
        ok_btn.setText("Save Settings")
        ok_btn.setObjectName("SaveBtn")
        ok_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        cancel_btn = btns.button(QtWidgets.QDialogButtonBox.Cancel)
        cancel_btn.setObjectName("CancelBtn")
        cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

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
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
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

        self.record_btn.setIcon(QtGui.QIcon("src/icon/play.svg")) # Ikon awal (belum merekam)
        self.record_btn.setIconSize(QtCore.QSize(18, 18))

        self.record_btn.setCheckable(True)
        self.record_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.record_btn.setMinimumSize(150, 42)
        self.record_btn.clicked.connect(self.toggle_record)

        # Container untuk VU Meter (Ubah menjadi Horizontal Layout)
        vu_layout = QtWidgets.QHBoxLayout()
        vu_layout.setContentsMargins(10, 0, 10, 0) # Beri sedikit jarak kiri-kanan
        vu_layout.setSpacing(8) # Jarak antar elemen di dalam VU layout

        # Indikator Mic
        self.mic_label = QtWidgets.QLabel("🎤 Mic")
        self.mic_vu = QtWidgets.QProgressBar()
        self.mic_vu.setRange(0, 100)
        self.mic_vu.setFixedHeight(12)
        self.mic_vu.setFixedWidth(60) # Gunakan FixedWidth agar ukurannya konsisten dan tidak terlalu panjang

        # Indikator Speaker
        self.spk_label = QtWidgets.QLabel("🔊 Spk")
        self.spk_vu = QtWidgets.QProgressBar()
        self.spk_vu.setRange(0, 100)
        self.spk_vu.setFixedHeight(12)
        self.spk_vu.setFixedWidth(60)

        # Susun menyamping (kiri ke kanan)
        vu_layout.addWidget(self.mic_label, alignment=QtCore.Qt.AlignVCenter)
        vu_layout.addWidget(self.mic_vu, alignment=QtCore.Qt.AlignVCenter)
        
        vu_layout.addSpacing(16) # Memberikan jarak ekstra pemisah antara grup Mic dan grup Spk
        
        vu_layout.addWidget(self.spk_label, alignment=QtCore.Qt.AlignVCenter)
        vu_layout.addWidget(self.spk_vu, alignment=QtCore.Qt.AlignVCenter)

        self.mute_btn = QtWidgets.QPushButton(" Mute Mic")
        self.mute_btn.setObjectName("MuteBtn")

        self.mute_btn.setIcon(QtGui.QIcon("src/icon/mic.svg"))
        self.mute_btn.setIconSize(QtCore.QSize(18, 18))

        self.mute_btn.setCheckable(True)
        self.mute_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mute_btn.setMinimumSize(110, 42)
        self.mute_btn.clicked.connect(self.toggle_mute)
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
        self.history_btn.setIcon(QtGui.QIcon("src/icon/history.svg")) # Ikon SVG
        self.history_btn.setIconSize(QtCore.QSize(18, 18))
        self.history_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.history_btn.setMinimumSize(90, 36)
        self.history_btn.clicked.connect(self.open_history)

        self.resume_btn = QtWidgets.QPushButton(" Summarize")
        self.resume_btn.setObjectName("ActionBtn")
        self.resume_btn.setIcon(QtGui.QIcon("src/icon/star.svg")) # Ikon SVG
        self.resume_btn.setIconSize(QtCore.QSize(18, 18))
        self.resume_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.resume_btn.setMinimumSize(110, 36)
        self.resume_btn.clicked.connect(self.resume_summary)
        self.resume_btn.setEnabled(False)

        self.export_btn = QtWidgets.QPushButton(" Export")
        self.export_btn.setObjectName("SecondaryBtn")
        self.export_btn.setIcon(QtGui.QIcon("src/icon/download.svg")) # Ikon SVG
        self.export_btn.setIconSize(QtCore.QSize(18, 18))
        self.export_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.export_btn.setMinimumSize(90, 36)
        self.export_btn.clicked.connect(self.export_transcript)

        self.settings_btn = QtWidgets.QPushButton(" Settings")
        self.settings_btn.setObjectName("SecondaryBtn")
        self.settings_btn.setIcon(QtGui.QIcon("src/icon/settings.svg")) # Ikon SVG
        self.settings_btn.setIconSize(QtCore.QSize(18, 18))
        self.settings_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.settings_btn.setMinimumSize(110, 36)
        self.settings_btn.clicked.connect(self.open_settings)

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

            QLabel {{
                color: {TEXT};
                font-weight: 600;
                font-size: 12px;
            }}
            
            /* General Button Styling */
            QPushButton {{
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
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
                font-size: 13px;
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
                font-size: 13px;
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
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
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
            self.record_btn.setIcon(QtGui.QIcon("src/icon/stop.svg")) # Ganti ke ikon stop
            self._start_live()
        else:
            # Saat berhenti merekam (Tombol kembali normal)
            self.record_btn.setText(" Start Recording") # Hapus emoji ⏺
            self.record_btn.setIcon(QtGui.QIcon("src/icon/play.svg")) # Ganti ke ikon record
            self._stop_live()

    def toggle_mute(self, checked: bool):
        self.muted = bool(checked)
        
        if self.muted:
            self.mute_btn.setText(" Muted")
            self.mute_btn.setIcon(QtGui.QIcon("src/icon/micoff.svg")) 
        else:
            self.mute_btn.setText(" Mute Mic")
            self.mute_btn.setIcon(QtGui.QIcon("src/icon/mic.svg"))  
            
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
            SummaryWindow({"entries": entries}, parent=self).exec_()
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
            SummaryWindow({"entries": [{"text": summary_note + summary_text}]}, parent=self).exec_()
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
        except Exception:
            pass
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = CompactWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
