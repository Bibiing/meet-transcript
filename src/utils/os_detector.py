"""Operating system detection helpers for selecting audio backends."""

from __future__ import annotations

from enum import Enum
import platform


class OperatingSystem(str, Enum):
    """Stable OS labels used by the application."""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNSUPPORTED = "unsupported"


class AudioBackend(str, Enum):
    """Stable backend labels used by the application."""

    WASAPI_LOOPBACK = "wasapi_loopback"
    SCREENCAPTUREKIT = "screencapturekit"
    SOUNDDEVICE_INPUT = "sounddevice_input"
    UNSUPPORTED = "unsupported"


def detect_os(system_name: str | None = None) -> OperatingSystem:
    """Detect the current operating system.

    Args:
        system_name: Optional platform name for tests. When omitted,
            ``platform.system()`` is used.
    """

    name = (system_name if system_name is not None else platform.system()).strip().lower()

    if name == "windows":
        return OperatingSystem.WINDOWS
    if name == "darwin":
        return OperatingSystem.MACOS
    if name == "linux":
        return OperatingSystem.LINUX
    return OperatingSystem.UNSUPPORTED


def get_audio_backend(
    operating_system: OperatingSystem | str | None = None,
) -> AudioBackend:
    """Return the planned audio backend for an operating system."""

    os_value = _coerce_operating_system(operating_system)

    if os_value is OperatingSystem.WINDOWS:
        return AudioBackend.WASAPI_LOOPBACK
    if os_value is OperatingSystem.MACOS:
        return AudioBackend.SCREENCAPTUREKIT
    if os_value is OperatingSystem.LINUX:
        return AudioBackend.SOUNDDEVICE_INPUT
    return AudioBackend.UNSUPPORTED


def is_windows(system_name: str | None = None) -> bool:
    """Return True when the current or provided OS is Windows."""

    return detect_os(system_name) is OperatingSystem.WINDOWS


def is_macos(system_name: str | None = None) -> bool:
    """Return True when the current or provided OS is macOS."""

    return detect_os(system_name) is OperatingSystem.MACOS


def is_linux(system_name: str | None = None) -> bool:
    """Return True when the current or provided OS is Linux."""

    return detect_os(system_name) is OperatingSystem.LINUX


def is_supported(system_name: str | None = None) -> bool:
    """Return True when phase 1 has a non-unsupported OS label."""

    return detect_os(system_name) is not OperatingSystem.UNSUPPORTED


def _coerce_operating_system(
    operating_system: OperatingSystem | str | None,
) -> OperatingSystem:
    if operating_system is None:
        return detect_os()
    if isinstance(operating_system, OperatingSystem):
        return operating_system

    normalized = operating_system.strip().lower()
    try:
        return OperatingSystem(normalized)
    except ValueError:
        return detect_os(normalized)
