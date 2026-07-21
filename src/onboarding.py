"""First-run wizard & dialog panduan permission mikrofon (W3).

Alur wizard: **Welcome -> Izin Mikrofon -> Selesai**. Konfigurasi server TIDAK ada
di wizard (tetap lewat dialog Settings), dan tidak ada uji koneksi.

Deteksi permission memakai `src.permissions`: buka device = source of truth,
registry = indikator penyebab. Dialog panduan dipakai ulang oleh wizard maupun gate
saat pengguna menekan Start Recording.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

from PySide6 import QtWidgets

from src.permissions import (
    GUIDANCE_DEVICE,
    GUIDANCE_NONE,
    GUIDANCE_PRIVACY,
    MicrophoneAccess,
    check_microphone,
    guidance_kind,
    guidance_message,
    open_privacy_settings,
)

_log = logging.getLogger(__name__)

FIRST_RUN_KEY = "app/first_run_completed"

_READY_TEXT = "Mikrofon siap digunakan."

# Judul mengikuti PENYEBAB. Menyebut "Izin Mikrofon" untuk device yang tidak ada
# atau sedang dipakai aplikasi lain mengarahkan pengguna ke tempat yang salah.
_TITLE_BY_KIND = {
    GUIDANCE_PRIVACY: "Izin Mikrofon",
    GUIDANCE_DEVICE: "Mikrofon Tidak Tersedia",
}
_TITLE_FALLBACK = "Mikrofon Bermasalah"
_NOT_READY_SUFFIX = (
    "\n\nAnda tetap dapat melanjutkan penyiapan, namun rekaman baru dapat dimulai "
    "setelah mikrofon dapat digunakan."
)

Checker = Callable[[Any], MicrophoneAccess]


class FlagStore(Protocol):
    """Subset UserStore (QSettings) yang dipakai untuk flag first-run."""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...


def is_first_run(store: FlagStore) -> bool:
    """True bila wizard belum pernah diselesaikan pengguna ini."""
    try:
        value = store.get(FIRST_RUN_KEY)
    except Exception as exc:  # store bermasalah -> jangan halangi startup
        _log.warning("failed to read first-run flag: %s", exc)
        return False
    return str(value).strip().lower() not in {"1", "true", "yes", "on"}


def mark_first_run_completed(store: FlagStore) -> None:
    try:
        store.set(FIRST_RUN_KEY, "true")
    except Exception as exc:  # gagal persist -> wizard muncul lagi, bukan crash
        _log.warning("failed to persist first-run flag: %s", exc)


class MicrophoneGuidanceDialog(QtWidgets.QDialog):
    """Jelaskan penyebab, arahkan ke Setelan Privasi, izinkan retry, tutup dengan aman."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        access: MicrophoneAccess,
        *,
        device: Any = None,
        checker: Checker = check_microphone,
    ) -> None:
        super().__init__(parent)
        self._device = device
        self._checker = checker
        self.setWindowTitle(_TITLE_BY_KIND.get(guidance_kind(access), _TITLE_FALLBACK))
        self.setMinimumWidth(460)

        self._message = QtWidgets.QLabel(guidance_message(access))
        self._message.setWordWrap(True)

        self._settings_btn = QtWidgets.QPushButton("Buka Setelan Privasi")
        self._settings_btn.clicked.connect(self._open_settings)
        self._retry_btn = QtWidgets.QPushButton("Coba lagi")
        self._retry_btn.clicked.connect(self._retry)
        self._close_btn = QtWidgets.QPushButton("Tutup")
        self._close_btn.clicked.connect(self.reject)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._settings_btn)
        buttons.addStretch(1)
        buttons.addWidget(self._retry_btn)
        buttons.addWidget(self._close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._message)
        layout.addLayout(buttons)

    def _open_settings(self) -> None:
        if not open_privacy_settings():
            self._message.setText(
                "Setelan Privasi tidak dapat dibuka otomatis. Buka Settings > Privacy & "
                "security > Microphone secara manual, lalu pilih Coba lagi."
            )

    def _retry(self) -> None:
        access = self._checker(self._device)
        if guidance_kind(access) == GUIDANCE_NONE:
            self.accept()
            return
        self.setWindowTitle(_TITLE_BY_KIND.get(guidance_kind(access), _TITLE_FALLBACK))
        self._message.setText(guidance_message(access))


def ensure_microphone_ready(
    parent: QtWidgets.QWidget | None,
    *,
    device: Any = None,
    checker: Checker = check_microphone,
) -> bool:
    """Gate reusable: True bila mikrofon siap (setelah panduan/retry bila perlu)."""
    access = checker(device)
    if guidance_kind(access) == GUIDANCE_NONE:
        return True
    dialog = MicrophoneGuidanceDialog(parent, access, device=device, checker=checker)
    return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted


class _WelcomePage(QtWidgets.QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Selamat datang")
        self.setSubTitle("Penyiapan singkat sebelum Anda mulai merekam rapat.")
        text = QtWidgets.QLabel(
            "Aplikasi ini merekam rapat dan mengubahnya menjadi transkrip.\n\n"
            "Langkah berikutnya memastikan aplikasi dapat menggunakan mikrofon Anda. "
            "Alamat server dapat diubah kapan saja melalui tombol Settings."
        )
        text.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(text)


class _PermissionPage(QtWidgets.QWizardPage):
    def __init__(self, *, device: Any = None, checker: Checker = check_microphone) -> None:
        super().__init__()
        self._device = device
        self._checker = checker
        self._ready = False
        self.setTitle("Izin Mikrofon")
        self.setSubTitle("Aplikasi memeriksa apakah mikrofon dapat digunakan.")

        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._settings_btn = QtWidgets.QPushButton("Buka Setelan Privasi")
        self._settings_btn.clicked.connect(self._open_settings)
        self._recheck_btn = QtWidgets.QPushButton("Coba lagi")
        self._recheck_btn.clicked.connect(self._refresh)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._settings_btn)
        buttons.addWidget(self._recheck_btn)
        buttons.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addLayout(buttons)

    def initializePage(self) -> None:  # noqa: N802 (API Qt)
        self._refresh()

    def _open_settings(self) -> None:
        if not open_privacy_settings():
            self._status.setText(
                "Setelan Privasi tidak dapat dibuka otomatis. Buka Settings > Privacy & "
                "security > Microphone secara manual, lalu pilih Coba lagi."
            )

    @property
    def ready(self) -> bool:
        return self._ready

    def _refresh(self) -> None:
        access = self._checker(self._device)
        self._ready = guidance_kind(access) == GUIDANCE_NONE
        self._status.setText(_READY_TEXT if self._ready else guidance_message(access) + _NOT_READY_SUFFIX)
        self._settings_btn.setVisible(not self._ready)
        self._recheck_btn.setVisible(not self._ready)
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802 (API Qt)
        # Wizard adalah onboarding, bukan titik penegakan: mikrofon yang belum siap
        # tidak mengunci pengguna di luar aplikasi (mis. mikrofon sedang dipakai aplikasi
        # lain — mereka tetap perlu akses Settings). Penegakan dilakukan gate saat Start
        # Recording (`ensure_microphone_ready`), yang sadar-source.
        return True


_FINISH_READY = "Mikrofon siap. Tekan Start Recording untuk mulai merekam."
_FINISH_NOT_READY = (
    "Penyiapan selesai, namun mikrofon belum dapat digunakan. Aplikasi akan menampilkan "
    "panduan yang sama saat Anda menekan Start Recording."
)
_FINISH_FOOTER = "\n\nAlamat server dan bahasa dapat diubah melalui tombol Settings."


class _FinishPage(QtWidgets.QWizardPage):
    def __init__(self, ready_probe: Callable[[], bool]) -> None:
        super().__init__()
        self._ready_probe = ready_probe
        self.setTitle("Selesai")
        self.setSubTitle("Penyiapan selesai.")
        self._text = QtWidgets.QLabel("")
        self._text.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._text)

    def initializePage(self) -> None:  # noqa: N802 (API Qt)
        # Teks mengikuti keadaan NYATA mikrofon: jangan menjanjikan "siap" bila tidak.
        ready = self._ready_probe()
        self._text.setText((_FINISH_READY if ready else _FINISH_NOT_READY) + _FINISH_FOOTER)


class FirstRunWizard(QtWidgets.QWizard):
    """Welcome -> Izin Mikrofon -> Selesai."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        device: Any = None,
        checker: Checker = check_microphone,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Penyiapan PLN Meeting Transcriber")
        self.setWizardStyle(QtWidgets.QWizard.WizardStyle.ModernStyle)
        self._permission_page = _PermissionPage(device=device, checker=checker)
        self.addPage(_WelcomePage())
        self.addPage(self._permission_page)
        self.addPage(_FinishPage(lambda: self._permission_page.ready))


def maybe_run_first_run_wizard(
    store: Optional[FlagStore],
    *,
    parent: QtWidgets.QWidget | None = None,
    device: Any = None,
    checker: Checker = check_microphone,
    wizard_factory: Callable[..., QtWidgets.QWizard] | None = None,
) -> bool:
    """Jalankan wizard bila ini first-run.

    Kembalikan True bila aplikasi boleh lanjut (bukan first-run, atau wizard selesai);
    False bila pengguna menutup wizard -> pemanggil keluar dengan aman.
    """
    if store is None or not is_first_run(store):
        return True
    factory = wizard_factory or FirstRunWizard
    wizard = factory(parent, device=device, checker=checker)
    if wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        _log.info("first-run wizard cancelled by user; exiting")
        return False
    mark_first_run_completed(store)
    return True
