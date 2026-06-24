"""Audio capture package."""

from src.capture.audio_frame import AudioFrame
from src.capture.errors import CaptureError, CaptureNotSupportedError
from src.capture.mac_sys_audio import MacSystemAudioConfig, MacSystemAudioStream
from src.capture.mic_stream import MicrophoneConfig, MicrophoneStream
from src.capture.win_loopback import WindowsLoopbackConfig, WindowsLoopbackStream

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
