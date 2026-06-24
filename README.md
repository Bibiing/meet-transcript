# Multiplatform Real-Time Transcriber

Fondasi aplikasi transkripsi rapat real-time multiplatform berbasis Python.
Fase 1 menyiapkan struktur project dan deteksi OS. Fase 2 menambahkan engine
capture mic/speaker berbasis queue serta writer WAV untuk validasi manual.

## Setup

```powershell
uv sync
```

## Run

```powershell
uv run python -m src.main
```

Contoh output:

```text
Detected OS: windows
Audio backend: wasapi_loopback
Smoke run completed.
```

## Phase 2 Capture

Record short WAV files from `main.py`:

```powershell
uv run python -m src.main --record --seconds 3
```

By default, results are saved to `audio/mic.wav` and `audio/speaker.wav`
when the selected source is available. Useful test parameters:

```powershell
uv run python -m src.main --record --seconds 5 --source both --output-dir audio
uv run python -m src.main --record --seconds 2 --source mic --sample-rate 48000 --block-size 1024
```

The capture engine keeps audio callbacks non-blocking and writes queued
`AudioFrame` blocks to 16-bit PCM WAV for manual validation. Temporary test
artifacts belong under `tmp/`; recordings belong under `audio/`.

On Windows, microphone capture uses `sounddevice`, while speaker capture uses
`soundcard` loopback devices so it does not try to open an output-only WASAPI
endpoint as an input stream.

## Test

```powershell
uv run pytest -q
```

## Platform Backend

| OS | Backend fase 1 |
| --- | --- |
| Windows | `wasapi_loopback` |
| macOS | `screencapturekit` |
| Linux | `sounddevice_input` |
| Lainnya | `unsupported` |
