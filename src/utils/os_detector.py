from enum import Enum
import platform

class AudioBackend(str, Enum):
    WASAPI_LOOPBACK = "wasapi_loopback"     # Windows
    SCREENCAPTUREKIT = "screencapturekit"   # macOS
    SOUNDDEVICE_INPUT = "sounddevice_input" # Linux
    UNSUPPORTED = "unsupported"

def get_audio_backend(operating_system: str | None = None) -> AudioBackend:
    name = (operating_system or platform.system()).strip().lower()
    print(f"Detected OS: {name}")

    if name == "windows":
        return AudioBackend.WASAPI_LOOPBACK
    if name == "darwin":
        return AudioBackend.SCREENCAPTUREKIT
    if name == "linux":
        return AudioBackend.SOUNDDEVICE_INPUT
    return AudioBackend.UNSUPPORTED
