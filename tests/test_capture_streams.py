from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import pytest

from src.capture.errors import CaptureNotSupportedError
from src.capture.mac_sys_audio import MacSystemAudioConfig, MacSystemAudioStream
from src.capture.mic_stream import MicrophoneConfig, MicrophoneStream
from src.capture.recorder import CaptureOptions, CaptureResult, run_capture
from src.capture.wav_sink import write_frames_to_wav
from src.capture.win_loopback import (
    list_wasapi_output_devices,
    select_soundcard_loopback_device,
    select_wasapi_output_device,
    WindowsLoopbackConfig,
    WindowsLoopbackStream,
)
from src.utils.os_detector import AudioBackend


class FakeStream:
    def __init__(self, callback, samples: np.ndarray | None = None) -> None:
        self.callback = callback
        self.samples = samples
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        if self.samples is not None:
            self.callback(self.samples, self.samples.shape[0], None, "")

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_microphone_stream_queues_copied_audio_frame() -> None:
    captured_kwargs = {}
    samples = np.ones((4, 1), dtype=np.float32)

    def factory(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeStream(kwargs["callback"], samples=samples)

    stream = MicrophoneStream(
        MicrophoneConfig(sample_rate=16_000, channels=1, block_size=4),
        stream_factory=factory,
    )

    stream.start()
    frame = stream.read_frame(timeout=0.01)
    stream.stop()

    assert captured_kwargs["samplerate"] == 16_000
    assert captured_kwargs["channels"] == 1
    assert captured_kwargs["blocksize"] == 4
    assert frame is not None
    assert frame.source == "mic"
    assert frame.sample_rate == 16_000
    assert frame.samples.shape == (4, 1)

    samples[0, 0] = 9.0
    assert frame.samples[0, 0] == 1.0


def test_microphone_queue_keeps_latest_frame_when_full() -> None:
    callbacks = []

    def factory(**kwargs):
        callbacks.append(kwargs["callback"])
        return FakeStream(kwargs["callback"])

    stream = MicrophoneStream(
        MicrophoneConfig(sample_rate=16_000, channels=1, block_size=2, queue_size=1),
        stream_factory=factory,
    )
    stream.start()

    callbacks[0](np.ones((2, 1), dtype=np.float32), 2, None, "")
    callbacks[0](np.full((2, 1), 2.0, dtype=np.float32), 2, None, "")
    frame = stream.read_frame(timeout=0.01)

    assert frame is not None
    assert np.all(frame.samples == 2.0)
    assert stream.frames_dropped == 2


class FakeWasapiSettings:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeSoundDevice:
    WasapiSettings = FakeWasapiSettings

    @staticmethod
    def query_hostapis():
        return (
            {"name": "MME", "default_output_device": 0},
            {"name": "Windows WASAPI", "default_output_device": 2},
        )

    @staticmethod
    def query_devices():
        return (
            {
                "name": "Mic",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
            {
                "name": "Other Speakers",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44_100,
            },
            {
                "name": "Speakers",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000,
            },
        )


class FakeSoundcardLoopback:
    isloopback = True
    name = "Speakers (Realtek) Loopback"
    channels = 2

    def recorder(self, samplerate, channels=None, blocksize=None):
        return FakeSoundcardRecorder(samplerate=samplerate, channels=channels, blocksize=blocksize)


class FakeSoundcardMic:
    isloopback = False
    name = "Microphone"
    channels = 1


class FakeSoundcardSpeaker:
    name = "Speakers (Realtek)"


class FakeSoundcardRecorder:
    def __init__(self, samplerate, channels=None, blocksize=None) -> None:
        self.samplerate = samplerate
        self.channels = channels or 2
        self.blocksize = blocksize or 4
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def record(self, numframes=None):
        self.read_count += 1
        frames = numframes or self.blocksize
        return np.full((frames, self.channels), self.read_count, dtype=np.float32)


class FakeSoundcard:
    @staticmethod
    def all_microphones(include_loopback=False):
        devices = [FakeSoundcardLoopback(), FakeSoundcardMic()]
        return devices if include_loopback else [device for device in devices if not device.isloopback]

    @staticmethod
    def default_speaker():
        return FakeSoundcardSpeaker()


def test_select_wasapi_output_device_uses_default_output() -> None:
    device = select_wasapi_output_device(sd_module=FakeSoundDevice)

    assert device.index == 2
    assert device.name == "Speakers"
    assert device.sample_rate == 48_000
    assert device.channels == 2


def test_list_wasapi_output_devices_filters_output_devices() -> None:
    devices = list_wasapi_output_devices(sd_module=FakeSoundDevice)

    assert [device.name for device in devices] == ["Speakers"]


def test_select_soundcard_loopback_device_uses_default_speaker_loopback() -> None:
    device = select_soundcard_loopback_device(soundcard_module=FakeSoundcard)

    assert device.index == 0
    assert device.name == "Speakers (Realtek) Loopback"
    assert device.channels == 2


def test_windows_loopback_stream_reads_soundcard_loopback_frames() -> None:
    stream = WindowsLoopbackStream(
        WindowsLoopbackConfig(sample_rate=16_000, channels=2, block_size=4),
        soundcard_module=FakeSoundcard,
        system_name="Windows",
    )

    stream.start()
    frame = stream.read_frame(timeout=1.0)
    stream.stop()

    assert frame is not None
    assert frame.source == "speaker"
    assert frame.sample_rate == 16_000
    assert frame.channels == 2
    assert frame.samples.shape == (4, 2)


def test_windows_loopback_rejects_non_windows_host() -> None:
    stream = WindowsLoopbackStream(soundcard_module=FakeSoundcard, system_name="Linux")

    with pytest.raises(CaptureNotSupportedError):
        stream.start()


class FakeNativeStream:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeScreenCaptureKitProvider:
    def __init__(self) -> None:
        self.native = FakeNativeStream()

    def start(self, callback):
        callback(np.zeros((3, 2), dtype=np.float32), "ok")
        return self.native


def test_mac_system_audio_stream_queues_provider_audio() -> None:
    provider = FakeScreenCaptureKitProvider()
    stream = MacSystemAudioStream(
        MacSystemAudioConfig(sample_rate=48_000, channels=2),
        provider=provider,
        system_name="Darwin",
        clock=lambda: 123.0,
    )

    stream.start()
    frame = stream.read_frame(timeout=0.01)
    stream.stop()

    assert provider.native.started
    assert provider.native.stopped
    assert provider.native.closed
    assert frame is not None
    assert frame.source == "speaker"
    assert frame.timestamp_seconds == 123.0


def test_write_frames_to_wav() -> None:
    stream = MicrophoneStream(MicrophoneConfig(sample_rate=8_000, channels=1), stream_factory=lambda **kwargs: FakeStream(kwargs["callback"]))
    stream._on_audio(np.array([[0.0], [0.5], [-0.5]], dtype=np.float32), 3, None, "")
    frame = stream.read_frame(timeout=0.01)
    assert frame is not None

    output = write_frames_to_wav(Path("tmp") / "tests" / "capture.wav", [frame])

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 8_000
        assert wav_file.getnframes() == 3


def test_run_capture_passes_detected_backend_to_speaker_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_backend():
        return AudioBackend.WASAPI_LOOPBACK

    def fake_record_speaker(options, audio_backend):
        captured["options"] = options
        captured["audio_backend"] = audio_backend
        return CaptureResult(source="speaker", warning="skipped")

    monkeypatch.setattr("src.capture.recorder.get_audio_backend", fake_backend)
    monkeypatch.setattr("src.capture.recorder._record_speaker", fake_record_speaker)

    result = run_capture(CaptureOptions(source="speaker", seconds=0.1))

    assert len(result) == 1
    assert captured["audio_backend"] is AudioBackend.WASAPI_LOOPBACK
    assert captured["options"].source == "speaker"
