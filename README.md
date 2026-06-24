# Multiplatform Real-Time Transcriber

Fondasi aplikasi transkripsi rapat real-time multiplatform berbasis Python.
Fase 1 menyiapkan struktur project, manajemen dependensi, deteksi OS, dan
entrypoint smoke-run. Belum ada audio capture sungguhan pada fase ini.

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
Phase 1 smoke run completed.
```

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
