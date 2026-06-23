from __future__ import annotations

from typing import Protocol

from rttranscriber.model.audio import AudioFrame


class AudioSource(Protocol):
    """Kontrak source audio agar coordinator tidak terikat ke view atau device."""

    def diagnostics(self) -> str: ...

    def start(self) -> None: ...

    def read_frame(self, timeout: float = 1.0) -> AudioFrame | None: ...

    def stop(self) -> None: ...
