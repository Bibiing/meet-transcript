from src.utils.os_detector import (
    AudioBackend,
    OperatingSystem,
    detect_os,
    get_audio_backend,
    is_linux,
    is_macos,
    is_supported,
    is_windows,
)


def test_windows_maps_to_wasapi_loopback_backend() -> None:
    assert detect_os("Windows") is OperatingSystem.WINDOWS
    assert get_audio_backend("Windows") is AudioBackend.WASAPI_LOOPBACK


def test_darwin_maps_to_screencapturekit_backend() -> None:
    assert detect_os("Darwin") is OperatingSystem.MACOS
    assert get_audio_backend("Darwin") is AudioBackend.SCREENCAPTUREKIT


def test_linux_maps_to_sounddevice_input_backend() -> None:
    assert detect_os("Linux") is OperatingSystem.LINUX
    assert get_audio_backend("Linux") is AudioBackend.SOUNDDEVICE_INPUT


def test_unknown_os_is_unsupported() -> None:
    assert detect_os("FreeBSD") is OperatingSystem.UNSUPPORTED
    assert get_audio_backend("FreeBSD") is AudioBackend.UNSUPPORTED


def test_boolean_helpers_use_platform_system(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    assert is_windows()
    assert not is_macos()
    assert not is_linux()
    assert is_supported()


def test_boolean_helpers_accept_explicit_system_name() -> None:
    assert is_macos("Darwin")
    assert not is_supported("Solaris")
