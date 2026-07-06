from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from time import monotonic, sleep
from typing import Literal

import numpy as np

from src.capture.audio_frame import AudioFrame
from src.capture.errors import CaptureError
from src.capture.mac_sys_audio import MacSystemAudioConfig, MacSystemAudioStream
from src.capture.mic_stream import MicrophoneConfig, MicrophoneStream
from src.capture.wav_sink import write_frames_to_wav
from src.capture.win_loopback import WindowsLoopbackConfig, WindowsLoopbackStream
from src.utils.os_detector import AudioBackend, get_audio_backend

_log = logging.getLogger(__name__)

CaptureSource = Literal["mic", "speaker", "both"]

@dataclass(frozen=True, slots=True)
class CaptureOptions:
    seconds: float = 3.0                            # durasi perekaman dalam detik
    output_dir: Path = Path("audio")                # direktori output untuk menyimpan file WAV
    source: CaptureSource = "both"                  # sumber audio yang akan direkam: "mic", "speaker", atau "both"
    sample_rate: int | None = None                  # sample rate dalam Hz
    block_size: int = 1_024                         # ukuran blok audio dalam sampel
    queue_size: int = 64                            # ukuran antrian untuk buffer audio
    mic_channels: int | None = None                 # jumlah saluran untuk mikrofon
    speaker_channels: int | None = None             # jumlah saluran untuk speaker
    mic_device: int | str | None = None             # perangkat mikrofon
    speaker_device: int | str | None = None         # perangkat speaker


@dataclass(frozen=True, slots=True)
class CaptureResult:
    source: str                                     # sumber audio yang direkam: "mic" atau "speaker"
    path: Path | None = None                        # path ke file WAV yang dihasilkan
    frame_count: int = 0                            # jumlah frame yang direkam
    duration_seconds: float = 0.0                   # durasi perekaman dalam detik
    sample_rate: int = 0                            # sample rate dalam Hz
    channels: int = 0                               # jumlah saluran audio
    warning: str = ""                               # peringatan jika ada masalah selama perekaman

    @property
    def ok(self) -> bool: return self.path is not None and not self.warning # mengembalikan True jika perekaman berhasil dan tidak ada peringatan

# rekam audio dari stream selama beberapa detik dan kembalikan daftar frame audio
def record_stream(stream: object, seconds: float) -> list[AudioFrame]:
    frames: list[AudioFrame] = []
    stream.start()
    try:
        deadline = monotonic() + seconds + 3.0 # tambahkan 3 detik untuk toleransi
        captured_seconds = 0.0  # jumlah detik yang telah direkam

        # loop untuk membaca frame dari stream hingga durasi yang diminta tercapai atau waktu habis
        while captured_seconds < seconds and monotonic() < deadline:    
            frame = stream.read_frame(timeout=0.25)
            if frame is not None:
                frames.append(frame)
                captured_seconds += frame.duration_seconds
    finally:
        stream.stop()
    return frames


def run_capture(options: CaptureOptions) -> list[CaptureResult]:
    if options.seconds <= 0:
        raise ValueError("invalid seconds: must be positive")
    if options.sample_rate is not None and options.sample_rate <= 0:
        raise ValueError("invalid sample_rate: must be positive")
    if options.block_size <= 0:
        raise ValueError("invalid block_size: must be positive")
    if options.queue_size <= 0:
        raise ValueError("invalid queue_size: must be positive")
    if options.mic_channels is not None and options.mic_channels <= 0:
        raise ValueError("invalid mic_channels: must be positive")
    if options.speaker_channels is not None and options.speaker_channels <= 0:
        raise ValueError("invalid speaker_channels: must be positive")

    options.output_dir.mkdir(parents=True, exist_ok=True) 
    results: list[CaptureResult] = []

    if options.source in ("mic", "both"):
        results.append(_record_microphone(options)) 

    if options.source in ("speaker", "both"):
        results.append(_record_speaker(options, get_audio_backend()))

    # tunggu untuk memastikan semua frame audio telah ditulis ke file WAV sebelum keluar
    sleep(0.05)
    return results

def _record_microphone(options: CaptureOptions) -> CaptureResult:
    try:
        stream = MicrophoneStream(
            MicrophoneConfig(
                sample_rate=options.sample_rate,
                channels=options.mic_channels,
                block_size=options.block_size,
                device=options.mic_device,
                queue_size=options.queue_size,
            )
        )
        frames = record_stream(stream, options.seconds)                     # rekam audio dari stream mikrofon
        return _write_result("mic", options.output_dir / "mic.wav", frames) # tulis hasil rekaman ke file WAV dan kembalikan hasilnya
    except CaptureError as exc:
        return CaptureResult(source="mic", warning=str(exc))


def _record_speaker(options: CaptureOptions, audio_backend: AudioBackend) -> CaptureResult:
    try:
        if audio_backend is AudioBackend.WASAPI_LOOPBACK:
            stream = WindowsLoopbackStream(
                WindowsLoopbackConfig(
                    sample_rate=options.sample_rate,
                    channels=options.speaker_channels,
                    block_size=options.block_size,
                    device=options.speaker_device,
                    queue_size=options.queue_size,
                )
            )
        elif audio_backend is AudioBackend.SCREENCAPTUREKIT:
            stream = MacSystemAudioStream(
                MacSystemAudioConfig(
                    sample_rate=options.sample_rate,
                    channels=options.speaker_channels,
                    block_size=options.block_size,
                    queue_size=options.queue_size,
                )
            )
        else:
            return CaptureResult(source="speaker", warning=f"speaker capture is unsupported on {audio_backend.value}")

        frames = record_stream(stream, options.seconds)
        return _write_result("speaker", options.output_dir / "speaker.wav", frames)
    except CaptureError as exc:
        return CaptureResult(source="speaker", warning=str(exc))


def _write_result(source: str, output_path: Path, frames: list[AudioFrame]) -> CaptureResult:
    if not frames:
        return CaptureResult(source=source, warning="capture produced no frames")
    failure = next((frame.status for frame in frames if frame.frame_count == 0 and frame.status), "")
    if failure:
        return CaptureResult(source=source, warning=failure)

    path = write_frames_to_wav(output_path, frames)
    frame_count = sum(frame.frame_count for frame in frames)
    sample_rate = frames[0].sample_rate
    channels = frames[0].channels
    duration_seconds = frame_count / sample_rate if sample_rate else 0.0

    # RC#5: Zero-signal detection -- warn if captured audio is pure silence
    all_samples = np.concatenate(
        [frame.samples.flatten() for frame in frames if frame.frame_count > 0]
    )
    rms = float(np.sqrt(np.mean(np.square(all_samples.astype(np.float32))))) if all_samples.size else 0.0
    if rms < 1e-8:
        _log.warning(
            "capture source=%s produced all-zeros audio (RMS=0) -- "
            "loopback device may not be receiving any audio signal",
            source,
        )
        return CaptureResult(
            source=source,
            path=path,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            warning=(
                f"captured audio is silent (RMS=0); "
                f"check that audio is playing and the loopback device is active"
            ),
        )

    _log.debug("capture source=%s rms=%.5f duration=%.2fs", source, rms, duration_seconds)
    return CaptureResult(
        source=source,
        path=path,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Record mic/speaker WAV files.")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("audio"))
    parser.add_argument("--source", choices=("mic", "speaker", "both"), default="both")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--block-size", type=int, default=1_024)
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--mic-channels", type=int, default=1)
    parser.add_argument("--speaker-channels", type=int, default=2)
    args = parser.parse_args()

    results = run_capture(
        CaptureOptions(
            seconds=args.seconds,
            output_dir=args.output_dir,
            source=args.source,
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            queue_size=args.queue_size,
            mic_channels=args.mic_channels,
            speaker_channels=args.speaker_channels,
        )
    )

    for result in results:
        if result.path is None:
            print(f"{result.source}: warning: {result.warning}")
            continue
        print(
            f"{result.source}: saved={result.path} "
            f"frames={result.frame_count} duration={result.duration_seconds:.2f}s "
            f"sample_rate={result.sample_rate} channels={result.channels}"
        )
    return 0 if any(result.path is not None for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
