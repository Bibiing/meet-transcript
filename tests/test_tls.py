"""W2 — TLS (wss://): wiring GUI->subprocess, verifikasi sertifikat, state terminal."""
from __future__ import annotations

import ssl

import numpy as np
import pytest

from src.config import ConfigProvider
from src.core import engine
from src.core.engine import UiOptions, build_live_command
from src.preprocessing.core import PreprocessedAudioChunk
from src.utils.status_ipc import format_status_line
from src.whisper.client_base import (
    TlsVerificationError,
    WhisperLiveConnectionConfig,
    WhisperLiveProfile,
    WhisperLiveStreamClient,
    _tls_failure_reason,
)
from src.whisper.client_reconnect import _BufferedReconnectClient
from src.whisper.models import ReconnectPolicy


def _chunk() -> PreprocessedAudioChunk:
    return PreprocessedAudioChunk(
        source="mic",
        samples=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        start_seconds=0.0,
        duration_seconds=0.1,
        rms_db=-12.0,
    )


def _cert_error(verify_code: int, message: str) -> ssl.SSLCertVerificationError:
    exc = ssl.SSLCertVerificationError(message)
    exc.verify_code = verify_code
    exc.verify_message = message
    exc.reason = "CERTIFICATE_VERIFY_FAILED"
    return exc


# Wiring GUI -> subprocess ------------------------------------------------
def test_build_live_command_omits_wss_by_default() -> None:
    assert "--server-wss" not in build_live_command(UiOptions())


def test_build_live_command_passes_wss_when_tls_enabled() -> None:
    # Regresi: sebelum W2, GUI TIDAK PERNAH dapat memakai wss:// (flag hanya di CLI).
    assert "--server-wss" in build_live_command(UiOptions(use_tls=True))


def test_options_from_config_carries_use_tls(tmp_path) -> None:
    from src.qt_client import _options_from_config

    path = tmp_path / "app_config.json"
    path.write_text('{"version": 1, "server": {"use_tls": true}}', encoding="utf-8")
    opts = _options_from_config(ConfigProvider(bundled_path=path, env={}))
    assert opts.use_tls is True


def test_use_tls_survives_settings_dialog_roundtrip(tmp_path) -> None:
    # Regresi: SettingsDialog.to_options() membuat UiOptions BARU; tanpa penerapan
    # ulang dari config, TLS mati diam-diam setelah pengguna membuka Settings.
    pytest.importorskip("PySide6")
    import inspect

    from src import qt_client

    source = inspect.getsource(qt_client.CompactWidget.open_settings)
    assert "use_tls = self._config.use_tls()" in source


# Klasifikasi kegagalan TLS ----------------------------------------------
def test_hostname_mismatch_is_reported_as_name_mismatch() -> None:
    reason = _tls_failure_reason(_cert_error(62, "Hostname mismatch, certificate is not valid for 'x'"))
    assert "does not match the server name" in reason


def test_untrusted_ca_is_reported_as_not_trusted() -> None:
    reason = _tls_failure_reason(_cert_error(19, "self signed certificate in certificate chain"))
    assert "not trusted" in reason


def test_expired_certificate_is_reported_as_expired() -> None:
    reason = _tls_failure_reason(_cert_error(10, "certificate has expired"))
    assert "expired" in reason


def test_generic_tls_failure_still_reported() -> None:
    exc = ssl.SSLError("handshake failure")
    exc.reason = "SSLV3_ALERT_HANDSHAKE_FAILURE"
    assert "TLS handshake failed" in _tls_failure_reason(exc)


# Perilaku client ---------------------------------------------------------
def test_tls_failure_raises_and_emits_actionable_status() -> None:
    statuses: list[dict] = []

    def failing_factory(url, timeout, header):
        raise _cert_error(19, "self signed certificate in certificate chain")

    client = WhisperLiveStreamClient(
        "mic",
        WhisperLiveConnectionConfig(host="server.example", port=9090, use_wss=True, profile=WhisperLiveProfile()),
        websocket_factory=failing_factory,
        on_status=lambda source, message: statuses.append(message),
        uid="mic-test",
    )
    with pytest.raises(TlsVerificationError):
        client.connect()

    tls = [m for m in statuses if m["status"] == "CLIENT_TLS_ERROR"]
    assert len(tls) == 1
    assert "not trusted" in tls[0]["reason"]
    assert tls[0]["url"] == "wss://server.example:9090"


def test_tls_failure_stops_reconnect_attempts(monkeypatch) -> None:
    # Sertifikat tak dipercaya tidak akan sembuh dengan mencoba ulang.
    import src.whisper.client_reconnect as reconnect_module

    attempts: list[int] = []

    def failing_factory(url, timeout, header):
        attempts.append(1)
        raise _cert_error(19, "self signed certificate in certificate chain")

    original = reconnect_module.WhisperLiveStreamClient

    class TlsFailingClient(original):
        def __init__(self, *args, **kwargs):
            kwargs["websocket_factory"] = failing_factory
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(reconnect_module, "WhisperLiveStreamClient", TlsFailingClient)

    wrapper = _BufferedReconnectClient(
        "mic",
        WhisperLiveConnectionConfig(use_wss=True, profile=WhisperLiveProfile()),
        policy=ReconnectPolicy(enabled=True, initial_backoff_seconds=0.0, buffer_seconds=5.0),
        on_transcript=lambda source, segments, message: None,
        on_status=lambda source, message: None,
    )
    wrapper.connect_initial()
    assert wrapper._fatal_status == "TLS_ERROR"
    for _ in range(5):
        wrapper.send(_chunk())
    assert attempts == [1]


# Jalur engine ------------------------------------------------------------
def test_engine_records_tls_error_details_and_error_state() -> None:
    engine.STATE.tls_error = None
    engine._handle_monitor_line(
        format_status_line("mic", "CLIENT_TLS_ERROR", reason="server certificate is not trusted by this computer",
                           url="wss://server.example:9090")
    )
    info = engine.tls_error_info()
    assert info == {
        "reason": "server certificate is not trusted by this computer",
        "url": "wss://server.example:9090",
    }
    assert engine.connection_status() == "ERROR"
    engine.STATE.tls_error = None


def test_engine_tls_info_none_by_default() -> None:
    engine.STATE.tls_error = None
    assert engine.tls_error_info() is None


def test_no_option_disables_certificate_verification() -> None:
    # Tanpa auth, TLS adalah satu-satunya pelindung: verifikasi tidak boleh dapat dimatikan.
    from pathlib import Path

    for path in (Path("src/whisper/client_base.py"), Path("src/whisper/models.py"), Path("src/config.py")):
        text = path.read_text(encoding="utf-8")
        assert "CERT_NONE" not in text
        assert "check_hostname" not in text or "client_base" in str(path)
        assert "verify_mode" not in text
