# Realtime Transcriber

Workspace ini mengikuti clean architecture untuk realtime transcriber lintas platform yang dijelaskan di `PLAN.md`. Fokus awalnya adalah membuat boundary yang bersih antara domain transcription, orchestration/use case, adapter platform audio, dan bootstrap app.

## Struktur

- `apps/cli`: entrypoint sederhana untuk wiring dependency dan smoke test pipeline
- `src/domain`: entity, value object, dan port yang stabil terhadap perubahan framework
- `src/application`: use case dan coordinator pipeline
- `src/infrastructure`: adapter logging, transcript sink, stub engine, dan placeholder audio source
- `src/infrastructure/platform`: target adapter konkret untuk Windows dan macOS
- `tests`: smoke test level workspace

## Build

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

`libsoundio` sekarang di-vendor pada `vendor/libsoundio` dan dibangun otomatis lewat CMake root.

## Next Step

- tambah adapter `libsoundio` di layer infrastructure/platform
- tambah worker `whisper.cpp`
- sambungkan VAD, sliding window chunker, dan transcript merger sebagai service application
