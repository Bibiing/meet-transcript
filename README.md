# PLN Meeting Transcriber

Aplikasi client-server untuk transkripsi online meeting. Folder `WhisperLive`
adalah server ASR yang dideploy dengan Docker. Folder root adalah client Python
untuk menangkap audio mic dan speaker, melakukan preprocessing ke 16 kHz mono
float32 PCM, lalu mengirim dua stream terpisah ke server.

Output live diberi label:

| Source | Label |
| --- | --- |
| `mic` | `Me` |
| `speaker` | `Meeting` |

## Quick Start

### 1. Install client

```powershell
cd D:\PLN
uv sync
```

Salin konfigurasi contoh jika ingin menyimpan nilai default lokal:

```powershell
Copy-Item .env.example .env
```

Semua command client `uv run python -m src.main ...` dijalankan dari root
project `D:\PLN`, bukan dari folder `D:\PLN\WhisperLive`.

### 2. Deploy server WhisperLive

```powershell
cd D:\PLN\WhisperLive
docker compose --env-file ..\.env up --build
```

Jika belum membuat `.env`, gunakan contoh bawaan:

```powershell
docker compose --env-file ..\.env.example up --build
```

Alternatif dari root project:

```powershell
cd D:\PLN
docker compose --env-file .env -f WhisperLive\docker-compose.yml up --build
```

Default server:

| Item | Default |
| --- | --- |
| WebSocket | `localhost:9090` |
| Metrics | `localhost:9091` |
| Backend | `faster_whisper` |
| Model client | `large-v3-turbo` |
| Bahasa | `id` |
| VAD threshold | `0.55` |
| Max clients | `4` |
| Max connection time | `600s` |

Server menjalankan anti-hallucination filter secara default. Untuk debugging:

```powershell
docker compose run --rm whisperlive python run_server.py --disable_segment_filter
```

### 3. Jalankan live meeting client

Jalankan ini saat Zoom, Teams, atau Google Meet sudah aktif:

```powershell
cd D:\PLN
uv run python -m src.main --live
```

Contoh output:

```text
Live transcription aktif via WhisperLive. Tekan Ctrl+C untuk berhenti.
Server: localhost:9090  |  Model: large-v3-turbo  |  VAD: 0.55
Sumber: MIC, SPEAKER
[00:04 - 00:07] [Me] Selamat pagi semua, kita mulai meeting.
[00:08 - 00:11] [Meeting] Baik, agenda hari ini deployment billing.
```

Opsi umum:

```powershell
# Server remote
uv run python -m src.main --live --server-host 192.168.1.20 --server-port 9090

# Simpan transcript backup JSON
uv run python -m src.main --live --transcript-log audio\meeting.json

# Hanya mic
uv run python -m src.main --live --source mic

# Ganti model
uv run python -m src.main --live --whisper-model large-v3

# Tuning VAD jika kata pendek terpotong
uv run python -m src.main --live --vad-threshold 0.50
```

Tekan **Ctrl+C** untuk menghentikan sesi.

### 4. Jalankan UI lokal

Jika server WhisperLive Docker sudah selalu berjalan, gunakan dashboard lokal
untuk start/stop client, melihat status server, transcript, dan log:

```powershell
cd D:\PLN
uv run python -m src.ui.server
```

Buka:

```text
http://127.0.0.1:8787
```

UI menjalankan command client yang sama dengan CLI, lalu membaca hasil dari
`audio/transcript_log.json`. Tombol **Stop** mengirim stop signal yang graceful
agar client masih sempat mengirim chunk tersisa dan menunggu hasil akhir sesuai
nilai `Final Drain`.

## Manual Smoke Test

Gunakan replay WAV untuk memastikan koneksi client-server berjalan tanpa harus
masuk online meeting.

```powershell
cd D:\PLN
uv run python -m src.main --replay-file audio\sample.wav --replay-source mic
```

Pada first run, Docker server bisa perlu beberapa menit untuk download/load model.
Jika replay timeout saat menunggu `SERVER_READY`, ulangi dengan timeout lebih panjang:

```powershell
uv run python -m src.main --replay-file audio\sample.wav --replay-source mic --server-ready-timeout 300
```

Jika ingin mengirim audio mengikuti durasi asli:

```powershell
uv run python -m src.main --replay-file audio\sample.wav --replay-source speaker --replay-realtime
```

Untuk cek konfigurasi Docker tanpa menjalankan model:

```powershell
docker compose -f WhisperLive\docker-compose.yml config
```

Untuk cek CLI client:

```powershell
uv run python -m src.main --help
```

## Speaker Capture

`speaker` menangkap suara peserta lain yang keluar dari speaker/headset melalui
WASAPI loopback di Windows. Jika tidak ada meeting aktif atau tidak ada audio
yang diputar, stream `speaker` akan kosong dan itu normal.

Pastikan aplikasi meeting sudah terbuka dan audio peserta lain terdengar sebelum
menjalankan `uv run python -m src.main --live`.

---

## Mode Batch (Analisis Rekaman)

Untuk analisis rekaman yang sudah ada atau debugging, gunakan pipeline batch tiga-tahap:

### Tahap 1: Rekam Audio

```powershell
# Rekam 30 detik dari mic + speaker
uv run python -m src.main --record --seconds 30 --source both

# Rekam mic saja
uv run python -m src.main --record --seconds 30 --source mic
```

### Tahap 2: Preprocessing

```powershell
uv run python -m src.main --preprocess
```

Menghasilkan `audio/mic.preprocessed.wav` dan `audio/speaker.preprocessed.wav`
(mono, 16 kHz, ternormalisasi, sudah di-VAD-filter).

### Tahap 3: Transkripsi

```powershell
uv run python -m src.main --transcribe
```

Hasil disimpan di `audio/transcript.phase4.json` dan ditampilkan di terminal.

### Batch Sekaligus

```powershell
uv run python -m src.main --record --preprocess --transcribe --seconds 30
```

---

## CLI Parameters

### Mode eksekusi

| Parameter            | Keterangan |
|----------------------|------------|
| `--live`             | Capture mic/speaker dan stream ke WhisperLive server. |
| `--replay-file`      | Stream file WAV ke WhisperLive untuk smoke test. |
| `--record`           | Rekam audio ke WAV (batch mode). |
| `--preprocess`       | Preprocessing WAV yang sudah direkam (batch mode). |
| `--transcribe`       | Transkripsi WAV lokal untuk debugging/offline analysis. |

### Parameter capture (berlaku untuk `--live` dan `--record`)

| Parameter              | Default          | Keterangan |
|------------------------|------------------|------------|
| `--source`             | `both`           | Sumber audio: `mic`, `speaker`, atau `both`. |
| `--sample-rate`        | bawaan perangkat | Override sample rate. Biarkan kosong untuk auto-detect. |
| `--block-size`         | `1024`           | Ukuran blok audio per callback (frame). |
| `--queue-size`         | `64`             | Maks blok audio yang diantrekan per stream. |
| `--mic-channels`       | `1`              | Channel mikrofon (mono = 1). |
| `--output-dir`         | `audio`          | Direktori output WAV (hanya batch mode). |
| `--seconds`            | `3.0`            | Durasi rekaman (hanya `--record`). |

### Parameter WhisperLive

| Parameter | Default | Keterangan |
| --- | --- | --- |
| `--server-host` | `localhost` | Host server WhisperLive. |
| `--server-port` | `9090` | Port WebSocket server. |
| `--server-wss` | off | Gunakan `wss://`. |
| `--server-api-key` | kosong | API key jika server memakai auth. |
| `--server-ready-timeout` | `300` | Batas tunggu client sampai server mengirim `SERVER_READY`. Naikkan pada first run/model warmup. |
| `--whisper-model` | `large-v3-turbo` untuk live/replay | Model Faster-Whisper. Hindari model `.en` untuk meeting Indonesia-Inggris. |
| `--whisper-language` | `id` | Bahasa utama meeting. |
| `--vad-threshold` | `0.55` | VAD server. Naikkan jika silence masih menghasilkan teks, turunkan jika kata pendek terpotong. |
| `--whisperlive-no-speech-thresh` | `0.45` | Filter `no_speech_prob` segment di WhisperLive. |
| `--live-chunk-seconds` | `1.0` | Durasi kirim audio dari client. Konteks tetap dijaga server lewat sliding window. |
| `--local-agreement` / `--no-local-agreement` | on | Server memakai sliding window dan hanya mengunci teks yang stabil di dua hipotesis berurutan. |
| `--local-agreement-window-seconds` | `15` | Durasi maksimum buffer konteks server. |
| `--local-agreement-hop-seconds` | `2` | Interval minimum antar transkripsi window. Lebih kecil = lebih realtime, lebih berat. |
| `--dynamic-prompt` / `--no-dynamic-prompt` | on | Transcript final terbaru dikirim balik sebagai konteks decode Whisper. |
| `--hide-partials` / `--no-hide-partials` | on | Sembunyikan partial mentah yang belum lolos Local Agreement. |
| `--initial-prompt` | prompt EYD/PUEBI konservatif | Instruksi decode Bahasa Indonesia: gunakan ejaan baku hanya untuk kata yang jelas dan jangan menambah isi. |
| `--hotwords` | glossary PLN/teknis | Istilah yang perlu dipertahankan seperti `API`, `database`, `PLN`, `meteran`, `token listrik`. |

### Parameter lokal/batch

| Parameter | Default | Keterangan |
| --- | --- | --- |
| `--asr-backend local` | off | Pakai OpenAI Whisper lokal untuk debugging tanpa server. |
| `--whisper-device` | auto | Override device lokal: `cpu` atau `cuda`. |
| `--whisper-fp16` / `--no-whisper-fp16` | auto | Paksa FP16 aktif/nonaktif untuk mode lokal. |
| `--whisper-min-logprob` | `-1.0` | Quality gate lokal. |
| `--whisper-max-no-speech` | `0.60` | Quality gate lokal. |
| `--whisper-max-compression` | `2.2` | Quality gate lokal. |
| `--transcript-output` | `audio/transcript.phase4.json` | File JSON output batch lokal. |

### Parameter logging & backup

| Parameter                  | Default                      | Keterangan |
|----------------------------|------------------------------|------------|
| `--transcript-log`         | `audio/transcript_log.json`  | Backup transkrip real-time, diperbarui setiap hasil baru. |
| `--resume-transcript-log`  | off                          | Append ke log yang sudah ada (default: mulai sesi baru). |
| `--log-level`              | `INFO`                       | Verbositas log: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `--log-file`               | `logs/transcriber.log`       | Path file log diagnostik. |

## Lokasi Output

| Mode | File default | Catatan |
| --- | --- | --- |
| Live CLI / UI | `audio/transcript_log.json` | JSON append-style, disimpan setiap hasil final diterima. |
| Batch phase 4 | `audio/transcript.phase4.json` | Output lengkap mode `--transcribe`. |
| Diagnostik client | `logs/transcriber.log` | Detail koneksi, chunk audio, VAD/preprocess, dan status WhisperLive. |

Gunakan `--transcript-log audio\nama_file.json` untuk menyimpan live transcript
ke file berbeda.

## Kualitas Transkrip

Transkrip meeting sebaiknya diperlakukan sebagai catatan bantu, bukan bukti
tunggal keputusan. Untuk mengurangi halusinasi dan salah dengar:

Live mode WhisperLive memakai **Sliding Window + Local Agreement** secara default.
Client mengirim audio tiap 1 detik. Server membaca buffer konteks sampai 15 detik, memprosesnya tiap 2 detik, lalu
hanya menandai teks sebagai final jika prefix kata stabil pada dua hipotesis
berturut-turut. Transcript final terbaru juga dipakai sebagai dynamic prompt
agar potongan audio berikutnya tetap memiliki konteks pembicaraan.

Partial mentah disembunyikan secara default karena belum stabil dan dapat berisi
tebakan. Aktifkan `--no-hide-partials` hanya untuk debugging kualitas audio/ASR.

Default prompt live juga mengarahkan Faster-Whisper untuk memakai ejaan baku
Bahasa Indonesia (EYD/PUEBI) hanya pada kata yang terdengar jelas. Prompt ini
sengaja konservatif: tidak boleh menambah informasi, mengganti makna, atau
menebak kreatif saat audio ambigu. Ubah `WHISPERLIVE_INITIAL_PROMPT` dan
`WHISPERLIVE_HOTWORDS` di `.env` jika glossary domain perlu ditambah.

| Masalah | Tindakan |
| --- | --- |
| Suara terlalu kecil | Dekatkan mic, naikkan input gain Windows, atau gunakan headset. Noise yang dinormalisasi bisa ikut terdengar kuat oleh ASR. |
| Banyak teks saat hening | Naikkan `--vad-threshold` ke `0.60`-`0.65` atau `--whisperlive-no-speech-thresh` ke `0.50`. |
| Kata pendek terpotong | Turunkan `--vad-threshold` ke `0.50`. |
| Semua transcript masuk `[Me]` | Pakai headset agar speaker meeting tidak bocor ke mic, atau jalankan `--source speaker` untuk isolasi suara meeting. |
| Akurasi `small` kurang | Coba `medium` sebagai kompromi, lalu `large-v3-turbo` jika GPU cukup. |
| Review keputusan penting | Simpan log JSON, cek `logs/transcriber.log`, dan verifikasi bagian ambigu dengan rekaman/sumber asli. |

---

## Performa & Ekspektasi

| Profile | Model | Device | Catatan |
| --- | --- | --- | --- |
| Balanced | `large-v3-turbo` | GPU | Default live/replay. Cocok untuk Indonesia-Inggris dengan latency lebih rendah dari `large-v3`. |
| Quality | `large-v3` | GPU | Akurasi lebih tinggi, resource lebih besar. |
| Low resource | `medium` / `small` | CPU/GPU | Gunakan jika server ringan atau tanpa GPU. |

Target awal:

| Metrik | Target |
| --- | ---: |
| End-to-end latency | 1.5-3 detik |
| Empty/silence transcript | mendekati 0 |
| Queue backlog | tidak terus naik |
| Dropped frames | < 1% |

---

## Test

```powershell
uv run python -m pytest tests\test_main.py tests\test_preprocessing.py tests\test_capture_streams.py tests\test_whisperlive_client.py tests\test_transcript_merger.py tests\test_whisperlive_replay.py tests\test_formatter_and_transcript_log.py -q
```

Untuk server post-processing:

```powershell
$env:PYTHONPATH="D:\PLN\WhisperLive"
uv run python -m pytest WhisperLive\tests\test_postprocessing.py WhisperLive\tests\test_base_backend.py -q
```

---

## Arsitektur Pipeline

### Live Mode (`--live`)

```
Mic stream     ──→ preprocess 16 kHz mono ──→ WebSocket source=mic ─────┐
Speaker stream ──→ preprocess 16 kHz mono ──→ WebSocket source=speaker ─┤
                                                                         ↓
WhisperLive server → Faster-Whisper + VAD + anti-hallucination filter → client transcript merger
```

### Batch Mode (`--record --preprocess --transcribe`)

```
--record     → mic.wav + speaker.wav
--preprocess → mic.preprocessed.wav + speaker.preprocessed.wav (VAD + normalisasi)
--transcribe → transcript.phase4.json + terminal output
```

---

## Platform Backend

| OS      | Backend capture speaker |
|---------|------------------------|
| Windows | WASAPI Loopback (`soundcard`) |
| macOS   | ScreenCaptureKit |
| Linux   | sounddevice (mic only) |

---

## Troubleshooting

### `couldn't find env file`
Path `--env-file` selalu relatif terhadap folder tempat command dijalankan.

```powershell
# Dari D:\PLN\WhisperLive
docker compose --env-file ..\.env up --build

# Dari D:\PLN
docker compose --env-file .env -f WhisperLive\docker-compose.yml up --build
```

Jika `.env` belum ada:

```powershell
cd D:\PLN
Copy-Item .env.example .env
```

### `ModuleNotFoundError: No module named 'src'`
Client harus dijalankan dari root project:

```powershell
cd D:\PLN
uv run python -m src.main --live
```

Jika terminal sedang berada di `D:\PLN\WhisperLive`, jalankan `cd ..` terlebih dahulu.

### `WhisperLive stream ... was not ready`
Server sedang download/load model atau GPU sedang warmup. Default client menunggu
`300` detik. Untuk first run model besar, command ini lebih aman:

```powershell
uv run python -m src.main --live --server-ready-timeout 600
```

Pantau log server:

```powershell
docker compose -f WhisperLive\docker-compose.yml logs --tail 80 whisperlive
```

### Live sudah aktif lalu muncul `Connection timed out`
Jika banner live sudah tampil dan server log menunjukkan `Processing audio`, ini
bukan timeout model. Artinya client tidak menerima transcript selama beberapa
detik, biasanya karena audio diam atau VAD membuang chunk, lalu receiver lama
menganggap koneksi idle sebagai error. Client saat ini menggunakan blocking
receive setelah handshake, jadi sesi tidak terputus saat server belum mengirim
segment.

### Docker build `TLS handshake timeout`
Ini masalah koneksi Docker ke Docker Hub saat menarik base image. Jika image
lokal sudah pernah berhasil dibuat, jalankan tanpa `--no-cache`:

```powershell
cd D:\PLN\WhisperLive
docker compose --env-file ..\.env up --build
```

Jika tetap perlu rebuild bersih, ulangi saat koneksi Docker Hub stabil.

### `[SPEAKER] preprocess skipped: no speech chunk passed VAD`
Ini **normal** jika tidak ada meeting atau audio yang sedang diputar saat recording.
Saat online meeting aktif, suara peserta lain akan tertangkap secara otomatis.

### Client gagal connect ke server
Pastikan server Docker sudah jalan dan port `9090` terbuka:

```powershell
docker compose -f WhisperLive\docker-compose.yml ps
docker compose -f WhisperLive\docker-compose.yml logs --tail 80 whisperlive
```

### Transkripsi tidak muncul / semua di-reject
- Pastikan server sudah selesai load model.
- Pastikan meeting/audio memang terdengar di perangkat output.
- Pastikan bahasa sudah benar: `--whisper-language id`
- Turunkan VAD jika ucapan pendek terpotong: `--vad-threshold 0.50`
- Naikkan VAD jika silence masih menghasilkan teks: `--vad-threshold 0.65`
- Periksa log: `--log-level DEBUG`

### Metrics tidak muncul
Compose expose metrics di `localhost:9091`. Jika endpoint kosong, rebuild image
agar dependency `prometheus-client` ikut terpasang:

```powershell
cd D:\PLN\WhisperLive
docker compose --env-file ..\.env build --no-cache
```
