"""Capture-specific exceptions."""


class CaptureError(RuntimeError):
    """Base exception for audio capture failures."""


class CaptureNotSupportedError(CaptureError):
    """Raised when the requested capture mode is unavailable on this host."""
