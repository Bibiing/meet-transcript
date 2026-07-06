"""Audio capture package."""

from src.capture.models import (
    AudioFrame,
    MacSystemAudioConfig,
    MicrophoneConfig,
    WindowsLoopbackConfig,
)
from src.capture.errors import CaptureError, CaptureNotSupportedError
from src.capture.mac_sys_audio import MacSystemAudioStream
from src.capture.mic_stream import MicrophoneStream
from src.capture.win_loopback import WindowsLoopbackStream

__all__ = [
    "AudioFrame",
    "CaptureError",
    "CaptureNotSupportedError",
    "MacSystemAudioConfig",
    "MacSystemAudioStream",
    "MicrophoneConfig",
    "MicrophoneStream",
    "WindowsLoopbackConfig",
    "WindowsLoopbackStream",
]
