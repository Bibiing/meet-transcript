"""Command-line entrypoint for the phase 1 smoke run."""

from __future__ import annotations

from src.utils.os_detector import detect_os, get_audio_backend


def main() -> int:
    """Print the selected platform/backend and exit without capturing audio."""

    operating_system = detect_os()
    audio_backend = get_audio_backend(operating_system)

    print(f"Detected OS: {operating_system.value}")
    print(f"Audio backend: {audio_backend.value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
