# Realtime Transcriber

Workspace ini memakai backend Python penuh dengan arsitektur `MVVM`. Jalur primary saat ini sudah mencakup audio pipeline dan realtime transcription yang tervalidasi.

## Primary Workspace

- `rttranscriber/model`
- `rttranscriber/services`
- `rttranscriber/viewmodels`
- `rttranscriber/views`
- `run_realtime_transcriber.py`
- `test`

## Secondary Workspace

- `run_audio_chunk_debug.py`
- `rttranscriber/audio_chunk_debug_session.py`
- `vendor/pysoundio`
- `vendor/libsoundio`

## Workflow

```powershell
uv sync
uv run pytest test -q
uv run python run_realtime_transcriber.py
```

Jika cache `uv` ke profil user dibatasi:

```powershell
$env:UV_CACHE_DIR="$PWD/.uv-cache"
uv sync
uv run pytest test -q
uv run python run_realtime_transcriber.py
```

## Next Step

- tambah VAD
- tambah transcript stabilization lintas overlap
- tambah backpressure handling
- tambah product layer di atas ViewModel
