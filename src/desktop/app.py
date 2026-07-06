"""Aplikasi desktop PyQt6 untuk PLN Meeting Transcriber.

File ini membangun UI native: kontrol koneksi, pemilihan device mic/speaker,
start/stop sesi, tampilan transcript, metrik sesi, dan log client.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QTabWidget,
)

from src.capture.mic_stream import list_input_devices
from src.capture.win_loopback import list_soundcard_loopback_devices
from src.engine.whisperlive_client import WhisperLiveProfile
from src.engine.whisperlive_session import WhisperLiveSessionConfig
from src.desktop.session_worker import SessionWorker, check_server_health


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRANSCRIPT_LOG = ROOT_DIR / "audio" / "transcript_log.json"

# ──────────────────────────────────────────────────────────────────────────────
#  Colour palette & Stylesheet
# ──────────────────────────────────────────────────────────────────────────────
ACCENT = "#0f766e"
ACCENT_STRONG = "#0b5f59"
DANGER = "#b42318"
WARN = "#b45309"
OK_BG = "#e6f4f1"
WARN_BG = "#fff3df"
DANGER_BG = "#fde7e7"

GLOBAL_STYLESHEET = f"""
QMainWindow {{
    background-color: #f6f7f9;
}}
QGroupBox {{
    font-weight: 600;
    font-size: 13px;
    color: #344054;
    border: 1px solid #d9dee7;
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    background: #ffffff;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
}}
QTabWidget::pane {{
    border: 1px solid #d9dee7;
    border-radius: 8px;
    background: #ffffff;
    top: -1px;
}}
QTabBar::tab {{
    background: #f6f7f9;
    border: 1px solid #d9dee7;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #647084;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background: #ffffff;
    border-bottom-color: #ffffff;
    color: {ACCENT};
}}
QLabel {{
    color: #647084;
    font-size: 12px;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: 30px;
    border: 1px solid #d9dee7;
    border-radius: 6px;
    padding: 4px 8px;
    background: #ffffff;
    color: #1d2430;
    font-size: 13px;
}}
QPushButton {{
    min-height: 34px;
    border: 1px solid #d9dee7;
    border-radius: 6px;
    background: #ffffff;
    color: #1d2430;
    font-weight: 600;
    font-size: 13px;
    padding: 0 14px;
}}
QPushButton:hover {{
    background: #f0f1f3;
}}
QPushButton:disabled {{
    opacity: 0.55;
    color: #999;
}}
QPushButton#startBtn {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
}}
QPushButton#startBtn:hover {{
    background: {ACCENT_STRONG};
}}
QPushButton#stopBtn, QPushButton#forceBtn {{
    color: {DANGER};
    border-color: #f0b8b2;
    background: #fffafa;
}}
QPushButton#summaryBtn {{
    background: #0284c7;
    border-color: #0284c7;
    color: #ffffff;
}}
QTableWidget {{
    border: none;
    background: #ffffff;
    gridline-color: #eef1f5;
    font-size: 13px;
    selection-background-color: {OK_BG};
}}
QTableWidget::item {{
    padding: 6px 8px;
}}
QHeaderView::section {{
    background: #f9fafb;
    border: none;
    border-bottom: 1px solid #d9dee7;
    font-weight: 600;
    font-size: 12px;
    color: #647084;
    padding: 6px 8px;
}}
QPlainTextEdit#logPanel {{
    background: #111827;
    color: #d1d5db;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 11px;
    border: none;
    border-radius: 0 0 8px 8px;
}}
"""


class MetricCard(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "MetricCard { background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; }"
        )
        self.setMinimumHeight(70)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        self._title = QLabel(title)
        self._title.setStyleSheet("font-size: 11px; color: #647084;")
        self._value = QLabel("0")
        self._value.setStyleSheet("font-size: 22px; font-weight: 700; color: #1d2430;")
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class StatusBadge(QLabel):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(26)
        self.setContentsMargins(10, 0, 10, 0)
        self.set_state("neutral")

    def set_state(self, state: str) -> None:
        colours = {
            "ok": f"background: {OK_BG}; color: {ACCENT_STRONG}; border: 1px solid #a7d8cf;",
            "warn": f"background: {WARN_BG}; color: {WARN}; border: 1px solid #f5d59e;",
            "danger": f"background: {DANGER_BG}; color: {DANGER}; border: 1px solid #f1b3ae;",
            "neutral": "background: #ffffff; color: #647084; border: 1px solid #d9dee7;",
        }
        base = colours.get(state, colours["neutral"])
        self.setStyleSheet(
            f"StatusBadge {{ {base} border-radius: 6px; font-size: 12px; padding: 2px 10px; font-weight: 600; }}"
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PLN Meeting Transcriber")
        self.resize(1200, 800)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self._worker: SessionWorker | None = None
        self._session_start: float | None = None
        self._transcript_entries: list[dict] = []

        self._build_ui()
        self._connect_signals()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1500)
        self._tick()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(14)

        # ── Top status bar ───────────────────────────────────────────
        self._server_badge = StatusBadge("Server checking…")
        self._session_badge = StatusBadge("Client idle")

        top_bar = QHBoxLayout()
        title = QLabel("PLN Meeting Transcriber")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #1d2430;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addWidget(self._server_badge)
        top_bar.addWidget(self._session_badge)

        # ── Left panel (Controls) ────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(12)

        self.tabs = QTabWidget()
        
        # TAB 1: UMUM (General)
        tab_umum = QWidget()
        umum_layout = QVBoxLayout(tab_umum)
        umum_layout.setSpacing(12)
        
        self._source = QComboBox()
        self._source.addItems(["both", "mic", "speaker"])
        self._model = QComboBox()
        self._model.addItems(["small", "medium", "large-v3-turbo", "base"])
        self._mic_device = QComboBox()
        self._speaker_device = QComboBox()
        self._load_audio_devices()
        self._language = QLineEdit("id")

        umum_layout.addWidget(QLabel("Sumber Suara (Source)"))
        umum_layout.addWidget(self._source)
        umum_layout.addWidget(QLabel("Model AI"))
        umum_layout.addWidget(self._model)
        umum_layout.addWidget(QLabel("Perangkat Mikrofon"))
        umum_layout.addWidget(self._mic_device)
        umum_layout.addWidget(QLabel("Perangkat Speaker"))
        umum_layout.addWidget(self._speaker_device)
        umum_layout.addWidget(QLabel("Bahasa (Kode, contoh: id)"))
        umum_layout.addWidget(self._language)
        umum_layout.addStretch()
        
        # TAB 2: LANJUTAN (Advanced/Technical)
        tab_lanjutan = QWidget()
        lanjutan_layout = QVBoxLayout(tab_lanjutan)
        
        scroll_area = QWidget()
        cg = QGridLayout(scroll_area)
        cg.setVerticalSpacing(8)
        cg.setHorizontalSpacing(10)
        
        row = 0
        def _add_pair(label_a, widget_a, label_b, widget_b):
            nonlocal row
            cg.addWidget(QLabel(label_a), row, 0)
            cg.addWidget(QLabel(label_b), row, 2)
            row += 1
            cg.addWidget(widget_a, row, 0, 1, 2)
            cg.addWidget(widget_b, row, 2, 1, 2)
            row += 1

        self._host = QLineEdit("localhost")
        self._port = QSpinBox(); self._port.setRange(1, 65535); self._port.setValue(9090)
        _add_pair("Server Host", self._host, "WS Port", self._port)

        self._vad = QDoubleSpinBox(); self._vad.setRange(0.1, 0.95); self._vad.setSingleStep(0.01); self._vad.setValue(0.55)
        self._no_speech = QDoubleSpinBox(); self._no_speech.setRange(0.1, 0.95); self._no_speech.setSingleStep(0.01); self._no_speech.setValue(0.75)
        _add_pair("VAD Threshold", self._vad, "No Speech Filter", self._no_speech)

        self._ready_timeout = QDoubleSpinBox(); self._ready_timeout.setRange(30, 3600); self._ready_timeout.setSingleStep(10); self._ready_timeout.setValue(300); self._ready_timeout.setDecimals(0)
        self._final_drain = QDoubleSpinBox(); self._final_drain.setRange(0, 60); self._final_drain.setSingleStep(1); self._final_drain.setValue(10); self._final_drain.setDecimals(0)
        _add_pair("Ready Timeout", self._ready_timeout, "Final Drain", self._final_drain)

        self._agreement_window = QDoubleSpinBox(); self._agreement_window.setRange(3, 60); self._agreement_window.setValue(20); self._agreement_window.setDecimals(0)
        self._agreement_hop = QDoubleSpinBox(); self._agreement_hop.setRange(0.2, 10); self._agreement_hop.setSingleStep(0.1); self._agreement_hop.setValue(3.0)
        _add_pair("Agreement Window", self._agreement_window, "Agreement Hop", self._agreement_hop)

        self._chunk_seconds = QDoubleSpinBox(); self._chunk_seconds.setRange(0.1, 5); self._chunk_seconds.setSingleStep(0.1); self._chunk_seconds.setValue(0.5)
        self._boundary_silence = QDoubleSpinBox(); self._boundary_silence.setRange(0.1, 3); self._boundary_silence.setSingleStep(0.1); self._boundary_silence.setValue(0.8)
        _add_pair("Chunk Seconds", self._chunk_seconds, "Boundary Silence", self._boundary_silence)

        self._boundary_max_wait = QDoubleSpinBox(); self._boundary_max_wait.setRange(1, 15); self._boundary_max_wait.setSingleStep(0.5); self._boundary_max_wait.setValue(5.0)
        self._log_level = QComboBox()
        self._log_level.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        _add_pair("Boundary Max Wait", self._boundary_max_wait, "Log Level", self._log_level)

        self._local_agreement = QCheckBox("Local agreement"); self._local_agreement.setChecked(True)
        self._dynamic_prompt = QCheckBox("Dynamic prompt"); self._dynamic_prompt.setChecked(True)
        self._speech_boundary = QCheckBox("Speech boundary detection"); self._speech_boundary.setChecked(True)
        self._mic_client_vad = QCheckBox("Mic client VAD"); self._mic_client_vad.setChecked(True)
        self._speaker_client_vad = QCheckBox("Speaker client VAD"); self._speaker_client_vad.setChecked(False)
        self._hide_partials = QCheckBox("Hide unstable partials"); self._hide_partials.setChecked(True)
        self._debug_chunk = QCheckBox("Debug chunk archive"); self._debug_chunk.setChecked(False)
        self._reset_transcript = QCheckBox("Archive old transcript on start"); self._reset_transcript.setChecked(True)

        for cb in (self._local_agreement, self._dynamic_prompt, self._speech_boundary,
                    self._mic_client_vad, self._speaker_client_vad,
                    self._hide_partials, self._debug_chunk, self._reset_transcript):
            cg.addWidget(cb, row, 0, 1, 4)
            row += 1

        lanjutan_layout.addWidget(scroll_area)
        lanjutan_layout.addStretch()

        self.tabs.addTab(tab_umum, "Umum")
        self.tabs.addTab(tab_lanjutan, "Lanjutan (Teknis)")
        left.addWidget(self.tabs)

        # ── Buttons Action Area (Tetap terlihat) ──
        btn_group = QGroupBox("Kontrol Sesi")
        btn_layout = QVBoxLayout(btn_group)
        
        btn_row1 = QHBoxLayout()
        self._start_btn = QPushButton("▶ Mulai (Start)")
        self._start_btn.setObjectName("startBtn")
        self._stop_btn = QPushButton("■ Berhenti (Stop)")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        btn_row1.addWidget(self._start_btn)
        btn_row1.addWidget(self._stop_btn)

        btn_row2 = QHBoxLayout()
        self._archive_btn = QPushButton("Arsip Transkrip")
        self._force_btn = QPushButton("Berhenti Paksa")
        self._force_btn.setObjectName("forceBtn")
        btn_row2.addWidget(self._archive_btn)
        btn_row2.addWidget(self._force_btn)

        btn_layout.addLayout(btn_row1)
        btn_layout.addLayout(btn_row2)
        
        left.addWidget(btn_group)
        
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(360)

        # ── Right panel (Workspace) ──────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        # Metrics row
        metrics = QHBoxLayout()
        self._metric_total = MetricCard("Total Transkrip")
        self._metric_me = MetricCard("Saya (Me)")
        self._metric_meeting = MetricCard("Rapat (Meeting)")
        self._metric_elapsed = MetricCard("Waktu Berjalan")
        for card in (self._metric_total, self._metric_me, self._metric_meeting, self._metric_elapsed):
            metrics.addWidget(card)

        # Transcript table
        transcript_group = QGroupBox("Transkrip Sesi")
        tg = QVBoxLayout(transcript_group)

        filter_row = QHBoxLayout()
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["Semua Sumber", "Saya", "Rapat"])
        self._filter_combo.setFixedWidth(160)
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Cari kata dalam transkrip...")
        
        self._summary_btn = QPushButton("Generate Summary")
        self._summary_btn.setObjectName("summaryBtn")
        
        filter_row.addWidget(self._filter_combo)
        filter_row.addWidget(self._search_box)
        filter_row.addWidget(self._summary_btn)
        tg.addLayout(filter_row)

        self._transcript_table = QTableWidget(0, 3)
        self._transcript_table.setHorizontalHeaderLabels(["Waktu", "Pembicara", "Teks"])
        header = self._transcript_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._transcript_table.verticalHeader().setVisible(False)
        self._transcript_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._transcript_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._transcript_table.setAlternatingRowColors(True)
        tg.addWidget(self._transcript_table)

        # Log panel
        log_group = QGroupBox("Log Klien Sistem")
        lg = QVBoxLayout(log_group)
        lg.setContentsMargins(0, 16, 0, 0)
        self._log_text = QPlainTextEdit()
        self._log_text.setObjectName("logPanel")
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(2000)
        self._log_text.setMinimumHeight(140)
        lg.addWidget(self._log_text)

        right.addLayout(top_bar)
        right.addLayout(metrics)
        right.addWidget(transcript_group, stretch=3)
        right.addWidget(log_group, stretch=1)

        right_widget = QWidget()
        right_widget.setLayout(right)

        root.addWidget(left_widget)
        root.addWidget(right_widget, stretch=1)

    def _connect_signals(self) -> None:
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._force_btn.clicked.connect(self._on_force_stop)
        self._archive_btn.clicked.connect(self._on_archive)
        self._filter_combo.currentIndexChanged.connect(self._render_transcript)
        self._search_box.textChanged.connect(self._render_transcript)
        self._summary_btn.clicked.connect(self._on_summary_clicked)

    def _on_summary_clicked(self) -> None:
        # Placeholder untuk fungsi summary yang Anda butuhkan
        QMessageBox.information(self, "Ringkasan", "Fitur generate summary akan memproses transkrip Anda di sini.")

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        if self._reset_transcript.isChecked():
            self._archive_transcript()

        self._transcript_entries.clear()
        self._render_transcript()
        self._log_text.clear()

        profile = WhisperLiveProfile(
            language=self._language.text().strip() or "id",
            model=self._model.currentText(),
            use_vad=False,
            vad_threshold=self._vad.value(),
            no_speech_thresh=self._no_speech.value(),
            local_agreement=self._local_agreement.isChecked(),
            local_agreement_window_seconds=self._agreement_window.value(),
            local_agreement_hop_seconds=self._agreement_hop.value(),
            dynamic_prompt=self._dynamic_prompt.isChecked(),
            speech_boundary_detection=self._speech_boundary.isChecked(),
            speech_boundary_silence_seconds=self._boundary_silence.value(),
            speech_boundary_max_wait_seconds=self._boundary_max_wait.value(),
        )

        config = WhisperLiveSessionConfig(
            server_host=self._host.text().strip() or "localhost",
            server_port=self._port.value(),
            chunk_seconds=self._chunk_seconds.value(),
            source=self._source.currentText(),
            mic_device=self._combo_device(self._mic_device),
            speaker_device=self._combo_device(self._speaker_device),
            mic_client_vad=self._mic_client_vad.isChecked(),
            speaker_client_vad=self._speaker_client_vad.isChecked(),
            auto_reconnect=True,
            reconnect_initial_backoff_seconds=1.0,
            reconnect_max_backoff_seconds=30.0,
            reconnect_buffer_seconds=30.0,
            ready_timeout=self._ready_timeout.value(),
            final_drain_seconds=self._final_drain.value(),
            show_partials=not self._hide_partials.isChecked(),
            log_transcript=DEFAULT_TRANSCRIPT_LOG,
            chunk_archive_dir=ROOT_DIR / "audio" / "chunks" if self._debug_chunk.isChecked() else None,
            profile=profile,
        )

        self._worker = SessionWorker(config, parent=self)
        self._worker.transcript_received.connect(self._on_transcript)
        self._worker.log_line.connect(self._on_log)
        self._worker.status_changed.connect(self._on_status)
        self._worker.error.connect(self._on_error)
        self._worker.session_finished.connect(self._on_finished)

        self._session_start = perf_counter()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._worker.start()

    def _load_audio_devices(self) -> None:
        self._mic_device.addItem("Default microphone", None)
        self._speaker_device.addItem("Default speaker loopback", None)
        try:
            for device in list_input_devices(concise=True):
                index = device.get("index")
                name = str(device.get("label") or device.get("name", f"device-{index}"))
                channels = int(device.get("max_input_channels", 0))
                self._mic_device.addItem(f"{name} ({channels} ch)", index)
        except Exception as exc:
            self._mic_device.addItem(f"Device probe failed: {exc}", None)
        try:
            for device in list_soundcard_loopback_devices():
                self._speaker_device.addItem(f"{device.name} ({device.channels} ch)", device.index)
        except Exception as exc:
            self._speaker_device.addItem(f"Device probe failed: {exc}", None)

    @staticmethod
    def _combo_device(combo: QComboBox) -> int | str | None:
        value = combo.currentData()
        return value if value not in ("", None) else None

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()

    def _on_force_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.terminate()
            self._on_finished()

    def _on_archive(self) -> None:
        archived = self._archive_transcript()
        if archived:
            QMessageBox.information(self, "Archived", f"Transcript archived to:\n{archived}")
        else:
            QMessageBox.information(self, "Archive", "No transcript file to archive.")

    def _on_transcript(self, entry: dict) -> None:
        self._transcript_entries.append(entry)
        self._render_transcript()
        total = len(self._transcript_entries)
        mic = sum(1 for e in self._transcript_entries if e.get("source") == "mic")
        spk = sum(1 for e in self._transcript_entries if e.get("source") == "speaker")
        self._metric_total.set_value(str(total))
        self._metric_me.set_value(str(mic))
        self._metric_meeting.set_value(str(spk))

    def _on_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.appendPlainText(f"{ts} {text}")

    def _on_status(self, status: str) -> None:
        labels = {
            "connecting": "Client connecting…",
            "running": "Client running",
            "stopping": "Client stopping…",
            "stopped": "Client idle",
        }
        self._session_badge.setText(labels.get(status, status))
        if status == "running":
            self._session_badge.set_state("ok")
        elif status == "stopping":
            self._session_badge.set_state("warn")
        elif status == "stopped":
            self._session_badge.set_state("neutral")
        else:
            self._session_badge.set_state("warn")

    def _on_error(self, message: str) -> None:
        self._on_log(f"[ERROR] {message}")

    def _on_finished(self) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._session_badge.setText("Client idle")
        self._session_badge.set_state("neutral")
        self._session_start = None

    def _tick(self) -> None:
        host = self._host.text().strip() or "localhost"
        port = self._port.value()
        healthy = check_server_health(host, port)
        self._server_badge.setText("Server healthy" if healthy else "Server offline")
        self._server_badge.set_state("ok" if healthy else "danger")

        if self._session_start is not None:
            elapsed = perf_counter() - self._session_start
            mins, secs = divmod(int(elapsed), 60)
            self._metric_elapsed.set_value(f"{mins}m {secs}s" if mins else f"{secs}s")

    def _render_transcript(self) -> None:
        filter_idx = self._filter_combo.currentIndex()
        source_filter = {0: None, 1: "mic", 2: "speaker"}.get(filter_idx)
        query = self._search_box.text().strip().lower()

        filtered = []
        for entry in self._transcript_entries:
            if source_filter and entry.get("source") != source_filter:
                continue
            if query and query not in (entry.get("text") or "").lower():
                continue
            filtered.append(entry)

        self._transcript_table.setRowCount(len(filtered))
        for row, entry in enumerate(filtered):
            start = _format_ts(entry.get("start_seconds", 0))
            end = _format_ts(entry.get("end_seconds", 0))
            time_item = QTableWidgetItem(f"{start} – {end}")
            time_item.setForeground(QColor("#647084"))

            source = entry.get("source", "")
            label = entry.get("label", source.upper())
            speaker_item = QTableWidgetItem(label)
            if source == "mic":
                speaker_item.setBackground(QColor(OK_BG))
                speaker_item.setForeground(QColor(ACCENT_STRONG))
            else:
                speaker_item.setBackground(QColor("#eef7ff"))
                speaker_item.setForeground(QColor("#075985"))
            font = speaker_item.font()
            font.setBold(True)
            speaker_item.setFont(font)

            text_item = QTableWidgetItem(entry.get("text", ""))

            self._transcript_table.setItem(row, 0, time_item)
            self._transcript_table.setItem(row, 1, speaker_item)
            self._transcript_table.setItem(row, 2, text_item)

        if filtered:
            self._transcript_table.scrollToBottom()

    def _archive_transcript(self) -> Path | None:
        path = DEFAULT_TRANSCRIPT_LOG.resolve()
        if not path.exists():
            return None
        archive_dir = path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = archive_dir / f"{path.stem}.{stamp}{path.suffix}"
        shutil.move(str(path), str(target))
        return target

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(5000)
        event.accept()

def _format_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    mins, sec = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{sec:02d}"
    return f"{mins:02d}:{sec:02d}"
