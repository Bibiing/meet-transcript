# Realtime Transcriber

Workspace ini mengikuti `PLAN.md` dengan backend Python penuh dan arsitektur `MVVM`. Jalur primary sekarang sudah mencakup Phase 1, Phase 2, dan Phase 3.

## Primary Workspace

- `rttranscriber/model`
- `rttranscriber/services`
- `rttranscriber/viewmodels`
- `rttranscriber/views`
- `run_realtime_transcriber.py`
- `test`
- `docs`

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

## Dokumen

- Arsitektur: `docs/architecture.md`
- Phase 1: `docs/phase-1-feasibility.md`
- Phase 2: `docs/phase-2-audio-pipeline.md`
- Phase 3: `docs/phase-3-realtime-transcription.md`
- Runbook: `docs/runbook-python-windows.md`

## Next Step

- tambah VAD
- tambah transcript stabilization lintas overlap
- tambah backpressure handling
- tambah product layer di atas ViewModel
