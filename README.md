# PLN Meeting Transcriber

Aplikasi client-server untuk transkripsi online meeting. Folder `WhisperLive`
adalah server ASR yang dideploy dengan Docker. Folder root adalah client Python
untuk menangkap audio mic dan speaker, melakukan preprocessing ke 16 kHz mono
PCM16, lalu mengirim dua stream terpisah ke server.

Output live diberi label:

| Source    | Label     |
| --------- | --------- |
| `mic`     | `Me`      |
| `speaker` | `Meeting` |

## Workspace Architecture

Workspace dipisahkan menjadi client application code, runtime artifacts, dan
server ASR vendor:

| Path                 | Tanggung jawab                                                                     |
| -------------------- | ---------------------------------------------------------------------------------- |
| `src/main.py`        | CLI composition root: parsing argumen, memilih mode, dan menghubungkan layer.      |
| `src/capture/`       | Adapter input audio untuk mic, Windows loopback, macOS system audio, dan WAV sink. |
| `src/preprocessing/` | Audio preprocessing pipeline dan transformasi menjadi PCM16 mono.                  |
| `src/utils/`         | Utility helpers, environment loading, dan logging support.                         |
| `src/whisper/`       | WhisperLive client/session layer dan transcription orchestration.                  |
| `src/core/`          | Desktop core engine untuk manajemen subprocess live session.                       |
| `tests/`             | Unit/integration tests untuk client code.                                          |
| `WhisperLive/`       | Server ASR Docker/vendor; jangan dijalankan sebagai client root.                   |

`audio/`, `logs/`, dan `tmp/` adalah output runtime dan di-ignore oleh Git.
Gunakan `.env.example` sebagai template; file `.env` lokal tidak disimpan di
repository.

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

`pln-whisperlive:gpu` bukan image publik yang perlu di-pull manual. Nama itu
adalah tag image lokal dari `WHISPERLIVE_IMAGE` yang dibuat Docker Compose dari
`WhisperLive/docker/Dockerfile.gpu`. Command `up --build` akan build image
tersebut jika belum ada atau jika Dockerfile berubah.

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

Jika server tidak memiliki NVIDIA GPU, gunakan compose CPU:

```bash
cd ~/meet-transcript
cp .env.example .env
```

Set `.env` untuk CPU:

```env
WHISPERLIVE_CPU_IMAGE=pln-whisperlive:cpu
WHISPERLIVE_DEFAULT_MODEL=small
WHISPERLIVE_CPU_MAX_CLIENTS=2
WHISPERLIVE_CPU_OMP_NUM_THREADS=4
```

Lalu jalankan:

```bash
docker compose --env-file .env -f WhisperLive/docker-compose.cpu.yml up -d --build
```

Jika VM CPU kecil dan model `small` terlalu lambat, gunakan
`WHISPERLIVE_DEFAULT_MODEL=base`. Detail Azure VM CPU/GPU ada di
`docs/PLAN.md`.

Default server:

| Item                | Default          |
| ------------------- | ---------------- |
| WebSocket           | `localhost:9090` |
| Backend             | `faster_whisper` |
| Model client        | `small`          |
| Bahasa              | `id`             |
| VAD threshold       | `0.55`           |
| Max clients         | `4`              |
| Max connection time | `600s`           |

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
Server: localhost:9090  |  Model: small  |  VAD: 0.55
Sumber: MIC, SPEAKER
[00:04 - 00:07] [Me] Selamat pagi semua, kita mulai meeting.
[00:08 - 00:11] [Meeting] Baik, agenda hari ini deployment billing.
```

Opsi umum:

```powershell
# Server remote
uv run python -m src.main --live --server-host 192.168.1.20 --server-port 9090

# Server Azure VM public IP
uv run python -m src.main --live --server-host 20.189.120.244 --server-port 9090 --server-ready-timeout 600

# Simpan transcript backup JSON
uv run python -m src.main --live --transcript-log audio\meeting.json

# Hanya mic
uv run python -m src.main --live --source mic

# Ganti model jika butuh kualitas lebih tinggi dan GPU cukup
uv run python -m src.main --live --whisper-model large-v3-turbo

# Tuning VAD jika kata pendek terpotong
uv run python -m src.main --live --vad-threshold 0.50
```

Tekan **Ctrl+C** untuk menghentikan sesi.

Untuk deployment server di Azure VM, lihat bagian **Deployment Azure VM** di
`docs/PLAN.md`. Ringkasnya: jalankan Docker Compose di VM, pastikan port `9090`
terbuka di Azure NSG dan firewall OS, lalu arahkan client lokal ke
`--server-host 20.189.120.244 --server-port 9090`.

### 4. Jalankan aplikasi desktop

Jika server WhisperLive Docker sudah berjalan, jalankan aplikasi desktop untuk
mengontrol client live transcription secara langsung:

```powershell
cd D:\PLN
uv run python -m src.qt_client
```

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

Saat server Docker start, Faster-Whisper model default `small` di-preload dalam
shared single-model mode. First run tetap bisa perlu beberapa menit untuk
download model ke volume cache, tetapi setelah server healthy client tidak perlu
menunggu load model per koneksi. Jika replay timeout saat menunggu `SERVER_READY`,
ulang dengan timeout lebih panjang:

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

Untuk melihat image lokal yang sudah dibuat:

```powershell
docker images pln-whisperlive
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

| Parameter       | Keterangan                                              |
| --------------- | ------------------------------------------------------- |
| `--live`        | Capture mic/speaker dan stream ke WhisperLive server.   |
| `--replay-file` | Stream file WAV ke WhisperLive untuk smoke test.        |
| `--record`      | Rekam audio ke WAV (batch mode).                        |
| `--preprocess`  | Preprocessing WAV yang sudah direkam (batch mode).      |
| `--transcribe`  | Transkripsi WAV lokal untuk debugging/offline analysis. |

### Parameter capture (berlaku untuk `--live` dan `--record`)

| Parameter                                          | Default          | Keterangan                                                                                                                                                           |
| -------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--source`                                         | `both`           | Sumber audio: `mic`, `speaker`, atau `both`.                                                                                                                         |
| `--sample-rate`                                    | bawaan perangkat | Override sample rate. Biarkan kosong untuk auto-detect.                                                                                                              |
| `--block-size`                                     | `1024`           | Ukuran blok audio per callback (frame).                                                                                                                              |
| `--queue-size`                                     | `64`             | Maks blok audio yang diantrekan per stream.                                                                                                                          |
| `--mic-channels`                                   | `1`              | Channel mikrofon (mono = 1).                                                                                                                                         |
| `--mic-device`                                     | default OS       | Index/nama input microphone. Pakai ini untuk memilih mic headset atau mic laptop.                                                                                    |
| `--speaker-device`                                 | default speaker  | Index/nama speaker loopback. Pakai ini untuk memilih output headset atau speaker laptop yang sedang dipakai meeting.                                                 |
| `--mic-client-vad` / `--no-mic-client-vad`         | on               | VAD ringan di client untuk mic. Matikan sementara jika suara mic terbukti hilang di `logs/process.log`.                                                              |
| `--speaker-client-vad` / `--no-speaker-client-vad` | off              | VAD ringan di client untuk speaker/system audio. Default off karena volume loopback aplikasi meeting/YouTube sering rendah dan berisiko membuang ucapan sebelum ASR. |
| `--output-dir`                                     | `audio`          | Direktori output WAV (hanya batch mode).                                                                                                                             |
| `--seconds`                                        | `3.0`            | Durasi rekaman (hanya `--record`).                                                                                                                                   |

Contoh memilih headset:

```powershell
uv run python -m src.main --live --mic-device "Headset" --speaker-device "Headset"
```

Nilai device bisa berupa index atau potongan nama perangkat. Web UI dan desktop
UI menampilkan daftar device yang sudah diringkas sehingga pengguna bisa memilih
mic headset, mic laptop, speaker headset, atau speaker laptop tanpa melihat
duplikasi internal Windows seperti MME/DirectSound/WDM-KS, Sound Mapper, Stereo
Mix, atau PC Speaker. Endpoint speaker tetap dipilih dari daftar loopback
terpisah.

### Parameter WhisperLive

| Parameter                                                        | Default                      | Keterangan                                                                                                                                                                          |
| ---------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--server-host`                                                  | `localhost`                  | Host server WhisperLive.                                                                                                                                                            |
| `--server-port`                                                  | `9090`                       | Port WebSocket server.                                                                                                                                                              |
| `--server-wss`                                                   | off                          | Gunakan `wss://`.                                                                                                                                                                   |
| `--server-api-key`                                               | kosong                       | API key jika server memakai auth.                                                                                                                                                   |
| `--server-ready-timeout`                                         | `300`                        | Batas tunggu client sampai server mengirim `SERVER_READY`. Naikkan pada first run/model warmup.                                                                                     |
| `--whisper-model`                                                | `small` untuk live/replay    | Model Faster-Whisper. Hindari model `.en` untuk meeting Indonesia-Inggris.                                                                                                          |
| `--whisper-language`                                             | `id`                         | Bahasa utama meeting.                                                                                                                                                               |
| `--vad-threshold`                                                | `0.55`                       | VAD server. Naikkan jika silence masih menghasilkan teks, turunkan jika kata pendek terpotong.                                                                                      |
| `--server-vad` / `--no-server-vad`                               | off                          | VAD Faster-Whisper di server. Default off karena audio sudah dipreprocess/VAD di client dan transcript final lebih penting dari caption realtime.                                   |
| `--whisperlive-no-speech-thresh`                                 | `0.75`                       | Filter `no_speech_prob` segment di WhisperLive. Lebih longgar agar kandidat transcript tidak habis dibuang.                                                                         |
| `--live-chunk-seconds`                                           | `0.5`                        | Durasi kirim audio dari client. 500 ms menekan overhead paket tanpa membuat transcript terasa lambat.                                                                               |
| `--audio-format`                                                 | `int16`                      | Format payload audio ke WhisperLive. PCM16 lebih hemat bandwidth dibanding `float32`.                                                                                               |
| `--auto-reconnect` / `--no-auto-reconnect`                       | on                           | Reconnect otomatis jika WebSocket putus. Saat aktif, sesi tidak langsung berhenti ketika server/network transient error.                                                            |
| `--reconnect-initial-backoff-seconds`                            | `1`                          | Delay awal sebelum reconnect ulang setelah disconnect.                                                                                                                              |
| `--reconnect-max-backoff-seconds`                                | `30`                         | Batas maksimum exponential backoff reconnect.                                                                                                                                       |
| `--reconnect-buffer-seconds`                                     | `30`                         | Durasi maksimum buffer chunk audio lokal per source saat reconnect. Jika buffer penuh, chunk tertua dibuang agar audio terbaru tetap diprioritaskan.                                |
| `--local-agreement` / `--no-local-agreement`                     | on                           | Server memakai sliding window dan hanya mengunci teks yang stabil di dua hipotesis berurutan.                                                                                       |
| `--local-agreement-window-seconds`                               | `20`                         | Durasi maksimum buffer konteks server. Cukup untuk konteks meeting tanpa membuat inferensi terlalu berat.                                                                           |
| `--local-agreement-hop-seconds`                                  | `3`                          | Interval minimum antar transkripsi window jika ada audio baru. Lebih kecil = lebih realtime, lebih berat.                                                                           |
| `--dynamic-prompt` / `--no-dynamic-prompt`                       | on                           | Transcript final terbaru dikirim balik sebagai konteks decode Whisper.                                                                                                              |
| `--speech-boundary-detection` / `--no-speech-boundary-detection` | on                           | Tunda ASR sampai tidak ada chunk speech baru, dengan max-wait fallback.                                                                                                             |
| `--speech-boundary-silence-seconds`                              | `0.8`                        | Durasi tanpa chunk speech baru sebelum ASR dianggap berada di akhir ucapan.                                                                                                         |
| `--speech-boundary-max-wait-seconds`                             | `5`                          | Batas tunggu saat pembicaraan terus berlangsung agar ASR tetap berjalan.                                                                                                            |
| `--hide-partials` / `--no-hide-partials`                         | on                           | Sembunyikan partial mentah yang belum lolos Local Agreement.                                                                                                                        |
| `--debug-chunk-archive` / `--no-debug-chunk-archive`             | off                          | Simpan setiap chunk sebagai WAV hanya untuk debugging. Default off agar disk I/O rendah.                                                                                            |
| `--chunk-archive-dir`                                            | `audio/chunks`               | Direktori chunk debug saat `--debug-chunk-archive` aktif.                                                                                                                           |
| `--initial-prompt`                                               | prompt EYD/PUEBI konservatif | Instruksi decode Bahasa Indonesia: gunakan ejaan baku hanya untuk kata yang jelas dan jangan menambah isi.                                                                          |
| `--hotwords`                                                     | kosong                       | Istilah bantu decode. Default kosong karena hotwords dapat bocor menjadi transcript saat audio lemah/noisy. Aktifkan hanya untuk sesi/domain yang benar-benar membutuhkan glossary. |

### Parameter lokal/batch

| Parameter                              | Default                        | Keterangan                                               |
| -------------------------------------- | ------------------------------ | -------------------------------------------------------- |
| `--asr-backend local`                  | off                            | Pakai OpenAI Whisper lokal untuk debugging tanpa server. |
| `--whisper-device`                     | auto                           | Override device lokal: `cpu` atau `cuda`.                |
| `--whisper-fp16` / `--no-whisper-fp16` | auto                           | Paksa FP16 aktif/nonaktif untuk mode lokal.              |
| `--whisper-min-logprob`                | `-1.0`                         | Quality gate lokal.                                      |
| `--whisper-max-no-speech`              | `0.60`                         | Quality gate lokal.                                      |
| `--whisper-max-compression`            | `2.2`                          | Quality gate lokal.                                      |
| `--transcript-output`                  | `audio/transcript.phase4.json` | File JSON output batch lokal.                            |

### Parameter logging & backup

| Parameter                 | Default                     | Keterangan                                                |
| ------------------------- | --------------------------- | --------------------------------------------------------- |
| `--transcript-log`        | `audio/transcript_log.json` | Backup transkrip real-time, diperbarui setiap hasil baru. |
| `--resume-transcript-log` | off                         | Append ke log yang sudah ada (default: mulai sesi baru).  |
| `--log-level`             | `INFO`                      | Verbositas log: `DEBUG`, `INFO`, `WARNING`, `ERROR`.      |
| `--log-file`              | `logs/transcriber.log`      | Path file log diagnostik.                                 |

## Lokasi Output

| Mode                     | File default                            | Catatan                                                                                                                                                                                 |
| ------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Live CLI / UI            | `audio/transcript_log.json`             | JSON append-style. Entry `stability=candidate` adalah live hypothesis/review; entry `stability=stable` adalah hasil completed Local Agreement.                                          |
| Live chunk archive debug | `audio/chunks/<session>/<source>/*.wav` | Hanya dibuat saat `--debug-chunk-archive` aktif. Default produksi tidak menulis ribuan file kecil.                                                                                      |
| Batch phase 4            | `audio/transcript.phase4.json`          | Output lengkap mode `--transcribe`.                                                                                                                                                     |
| Diagnostik client        | `logs/transcriber.log`                  | Detail koneksi, chunk audio, VAD/preprocess, dan status WhisperLive.                                                                                                                    |
| Process log client       | `logs/process.log`                      | JSONL event per tahap: capture start, backend/device, chunk created, VAD pass/drop, queue, WebSocket, SERVER_READY, chunk sent, END_OF_AUDIO, transcript received.                      |
| Process log server       | `logs/whisperlive/process.log`          | JSONL event dari container: client connected, options, audio received, buffer size, boundary wait/process, ASR start/end, local agreement, TVE score/pending/drop/emit, send to client. |

Gunakan `--transcript-log audio\nama_file.json` untuk menyimpan live transcript
ke file berbeda.

Untuk observability production, gunakan process log sebagai sumber investigasi
utama karena setiap baris adalah JSON mandiri.

Reconnect otomatis menulis event tambahan ke `logs/process.log`:

- `client.chunk_buffered`: chunk ditahan di RAM karena source sedang disconnected.
- `client.reconnect_status`: percobaan reconnect, gagal, atau berhasil.
- `client.reconnect_buffer_flushed`: buffer lokal berhasil dikirim ulang ke server.

Buffer reconnect bersifat bounded per source dan default-nya 30 detik. Ketika
buffer penuh, chunk tertua dibuang agar audio terbaru tetap diprioritaskan.

```powershell
Get-Content -Wait logs\process.log
Get-Content -Wait logs\whisperlive\process.log
```

Jika transcript mic kosong, cek event `client.vad_drop` di `logs/process.log`.
Jika mayoritas reason adalah `vad_silence`, coba pilih device mic yang benar
atau jalankan sementara dengan `--no-mic-client-vad` untuk memastikan audio
benar-benar masuk ke server.

Jika transcript mic justru berisi audio meeting/speaker, biasanya device mic
yang dipilih adalah endpoint salah seperti Stereo Mix/loopback atau mic headset
menangkap suara output. Web UI menyaring alias tersebut dari dropdown mic; cek
`/api/audio-devices` untuk melihat `diagnostics.raw_mic_count` dan jumlah device
mic yang benar-benar ditampilkan.

## Kualitas Transkrip

Transkrip meeting sebaiknya diperlakukan sebagai catatan bantu, bukan bukti
tunggal keputusan. Untuk mengurangi halusinasi dan salah dengar:

Live mode WhisperLive memperlakukan Whisper sebagai penghasil hipotesis, bukan
sebagai keputusan akhir. Client mengirim audio PCM16 tiap 500 ms. Server menahan
ASR sampai speech boundary terdeteksi, atau memaksa proses setelah 5 detik jika
pembicaraan terus berlangsung. Hasil ASR kemudian melewati Sliding Window,
Local Agreement, dan Transcript Validation Engine sebelum dikirim ke client.

Transcript Validation Engine memberi `reliability_score` pada setiap segment
dengan indikator ASR confidence, language shape, context consistency, stability,
dan dictionary/format shape. Segment yang jelas buruk dibuang, segment di bawah
threshold ditahan di pending queue, dan segment yang muncul lagi dengan konteks
lebih stabil baru dikirim.

Flow produksi yang disarankan:

1. Server start lebih dulu dan preload `WHISPERLIVE_DEFAULT_MODEL=small`.
2. Client start hanya setelah handshake `SERVER_READY`, lalu mulai capture mic
   dan speaker sebagai dua WebSocket stream terpisah.
3. Client menyimpan live transcript final yang sudah stabil ke
   `audio/transcript_log.json`.
4. Server hanya mengirim segment yang lolos Local Agreement dan reliability
   threshold; segment meragukan masuk pending queue.
5. Chunk live tidak disimpan ke file secara default. Aktifkan
   `--debug-chunk-archive` hanya ketika perlu audit audio atau final reprocess.
6. Saat stop, client mengirim sisa chunk, mengirim `END_OF_AUDIO`, menunggu final
   drain dari server, lalu baru menutup koneksi.
7. Post-meeting refinement berjalan async dari chunk/audio + transcript final,
   bukan di jalur live, supaya latency meeting tetap rendah.

Partial mentah disembunyikan secara default karena belum stabil dan dapat berisi
tebakan. Aktifkan `--no-hide-partials` hanya untuk debugging kualitas audio/ASR.

Default prompt live menggunakan instruksi bahasa Inggris yang konservatif untuk
mengarahkan Faster-Whisper menghasilkan transcript Bahasa Indonesia, bukan
terjemahan Inggris. Untuk ASR, prompt ini bukan chat instruction seperti LLM;
ia lebih berperan sebagai konteks decoder, jadi tetap harus singkat dan tidak
meminta model mengarang kata yang tidak terdengar. Ubah
`WHISPERLIVE_INITIAL_PROMPT` dan `WHISPERLIVE_HOTWORDS` di `.env` jika glossary
domain perlu ditambah.

| Masalah                                               | Tindakan                                                                                                                    |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Suara terlalu kecil                                   | Dekatkan mic, naikkan input gain Windows, atau gunakan headset. Noise yang dinormalisasi bisa ikut terdengar kuat oleh ASR. |
| Banyak teks saat hening                               | Naikkan `--vad-threshold` ke `0.60`-`0.65` atau `--whisperlive-no-speech-thresh` ke `0.50`.                                 |
| Kata pendek terpotong                                 | Turunkan `--vad-threshold` ke `0.50`.                                                                                       |
| Tidak ada transcript tetapi chunk terkirim            | Pastikan `--no-server-vad` aktif. Log `VAD filter removed ...` berarti VAD server membuang audio yang sudah dikirim client. |
| Transcript terlambat keluar saat orang bicara panjang | Turunkan `--speech-boundary-max-wait-seconds` ke `3`-`4`.                                                                   |
| Potongan kalimat masih terlalu fragmental             | Naikkan `--speech-boundary-silence-seconds` ke `1.0`-`1.2`.                                                                 |
| Semua transcript masuk `[Me]`                         | Pakai headset agar speaker meeting tidak bocor ke mic, atau jalankan `--source speaker` untuk isolasi suara meeting.        |
| Akurasi `small` kurang                                | Coba `medium` sebagai kompromi, lalu `large-v3-turbo` jika GPU cukup.                                                       |
| Review keputusan penting                              | Simpan log JSON, cek `logs/transcriber.log`, dan verifikasi bagian ambigu dengan rekaman/sumber asli.                       |

LLM lokal cocok untuk post-processing ringan: merapikan tanda baca, kapitalisasi,
dan membuat ringkasan dari transcript yang sudah ada. Jangan gunakan LLM lokal
atau API LLM eksternal untuk "menebak" kata yang tidak terdengar. Jika perlu API
LLM luar, pakai hanya untuk tahap post-meeting dengan payload terbatas dan aturan
privasi yang jelas; jangan taruh di live path karena menambah latency, biaya,
dan risiko kebocoran data meeting.

---

## Performa & Ekspektasi

| Profile        | Model                       | Device  | Catatan                                                                     |
| -------------- | --------------------------- | ------- | --------------------------------------------------------------------------- |
| Balanced       | `small`                     | CPU/GPU | Default live/replay. Dipreload saat server start agar koneksi client cepat. |
| Quality        | `large-v3`                  | GPU     | Akurasi lebih tinggi, resource lebih besar.                                 |
| Higher quality | `medium` / `large-v3-turbo` | GPU     | Gunakan jika akurasi `small` kurang dan latency masih diterima.             |

Target awal:

| Metrik                          |           Target |
| ------------------------------- | ---------------: |
| End-to-end latency              |        2-5 detik |
| Empty/silence transcript        |      mendekati 0 |
| Low-reliability segment emitted |      mendekati 0 |
| Queue backlog                   | tidak terus naik |
| Dropped frames                  |             < 1% |

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
Mic stream     ──→ preprocess 16 kHz mono PCM16 ──→ WebSocket source=mic ─────┐
Speaker stream ──→ preprocess 16 kHz mono PCM16 ──→ WebSocket source=speaker ─┤
                                                                         ↓
WhisperLive server
  → speech boundary detection
  → Faster-Whisper hypothesis
  → local agreement
  → transcript validation engine
  → pending/emit decision
  → client transcript merger
```

### Batch Mode (`--record --preprocess --transcribe`)

```
--record     → mic.wav + speaker.wav
--preprocess → mic.preprocessed.wav + speaker.preprocessed.wav (VAD + normalisasi)
--transcribe → transcript.phase4.json + terminal output
```

---

## Platform Backend

| OS      | Backend capture speaker       |
| ------- | ----------------------------- |
| Windows | WASAPI Loopback (`soundcard`) |
| macOS   | ScreenCaptureKit              |
| Linux   | sounddevice (mic only)        |

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

Server sedang download/load model default saat startup atau GPU sedang warmup.
Default client menunggu `300` detik. Untuk first run model besar, command ini
lebih aman:

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
