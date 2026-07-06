# PLN Meeting Transcriber - Flow Rekomendasi

## Tujuan

Sistem dipakai untuk transkripsi online meeting dari dua sumber: `mic` untuk
suara user dan `speaker` untuk suara peserta meeting. Target utama adalah live
transcript yang cukup responsif, beban server rendah, dan final transcript yang
lebih bersih setelah meeting selesai.

## Flow Produksi

1. Server WhisperLive start lebih dulu melalui Docker.
2. Server preload Faster-Whisper `small` dari `WHISPERLIVE_DEFAULT_MODEL`.
3. Port WebSocket `9090` terbuka dan server siap menerima client.
4. Client start dan membuka dua WebSocket stream: `mic` dan `speaker`.
5. Client mengirim options, menunggu handshake `SERVER_READY`, lalu mulai capture.
6. Audio dipreprocess ke 16 kHz mono PCM16 dan dikirim sebagai chunk 500 ms.
7. Server menunggu speech boundary: tidak ada chunk speech baru selama 0.8 detik,
   atau memaksa proses setelah 5 detik jika speech terus berlangsung.
8. Server menjalankan sliding window 20 detik + local agreement hop 3 detik
   untuk menjaga konteks.
   VAD server dimatikan secara default karena client sudah melakukan preprocess
   dan VAD awal; ini mengurangi risiko semua kandidat transcript dibuang.
9. Server menjalankan Transcript Validation Engine sebelum segment dikirim.
10. Client hanya menyimpan transcript final/stabil ke `audio/transcript_log.json`.
11. Chunk audio tidak disimpan sebagai file secara default. Gunakan ring buffer
   di memori; aktifkan archive hanya saat debugging atau audit tertentu.
12. Saat stop, client mengirim sisa chunk, memberi sinyal end-of-audio, menunggu
    final drain, menyimpan transcript terakhir, lalu menutup koneksi.

## Model Default

Default operasional adalah `small`, bukan `large-v3-turbo`, karena:

- startup dan warmup lebih cepat;
- cukup ringan untuk preload saat server init;
- multilingual, sehingga aman untuk meeting Indonesia-Inggris;
- latency lebih stabil untuk live meeting.

Naikkan ke `medium`, `large-v3-turbo`, atau `large-v3` hanya jika GPU cukup dan
akurasi `small` tidak memadai.

## Strategi Anti-Halusinasi

Jalur live harus tetap konservatif:

- VAD client membuang silence/noise sebelum ASR;
- server memakai no-speech, avg-logprob, compression-ratio, repetition filter;
- speech boundary detection menghindari inferensi pada potongan kalimat yang
  masih berjalan;
- sliding window mempertahankan konteks 20 detik;
- local agreement hanya finalisasi teks yang stabil di beberapa hipotesis;
- Transcript Validation Engine memberi reliability score dari ASR confidence,
  language shape, context consistency, stability, dan dictionary/format shape;
- segment dengan reliability tinggi dikirim, segment meragukan masuk pending
  queue, segment buruk dibuang;
- partial transcript tidak disimpan sebagai final.

Live transcript tetap dapat salah. Jika perlu final pass berbasis audio, gunakan
rolling audio file atau debug chunk archive hanya untuk sesi yang memang diaudit.

## Final Transcript

Flow final yang disarankan:

1. Ambil rolling audio/debug archive `mic` dan `speaker` per sesi jika tersedia.
2. Gabungkan timeline berdasarkan timestamp.
3. Re-run ASR batch dengan window lebih besar dan model lebih akurat jika GPU ada.
4. Merge ulang hasil `mic` dan `speaker`.
5. Jalankan post-processing LLM hanya untuk merapikan teks yang sudah ada:
   punctuation, kapitalisasi, paragraphing, dedup, dan ringkasan.
6. Simpan output final terpisah dari live transcript.

LLM tidak boleh dipakai untuk mengarang kata yang tidak ada di audio. Untuk data
meeting sensitif, prioritaskan LLM lokal. API LLM luar hanya layak untuk
post-meeting jika ada persetujuan privasi, redaksi data sensitif, dan audit log.

## Reliability Policy

Whisper dianggap sebagai hypothesis generator. Segment baru dikirim setelah
melewati policy berikut:

| Reliability | Aksi |
| --- | --- |
| `>= 0.80` | Emit ke client. |
| `0.70 - 0.80` | Tahan di pending queue untuk dievaluasi pada window berikutnya. |
| `< 0.70` | Pending/drop sampai ada konteks yang cukup kuat. |

Faktor score saat ini deterministik dan murah:

- ASR confidence: `avg_logprob`, `no_speech_prob`, `compression_ratio`, word probability jika ada.
- Stability: completed segment dan pengulangan dari pending context.
- Context consistency: kelengkapan kalimat dan apakah hypothesis pernah muncul.
- Language shape: bentuk token wajar dan bukan phrase hallucination umum.
- Dictionary shape: rasio token unik dan istilah teknis/domain.

## Risiko Saat Ini

- Jika server VAD aktif, audio meeting yang sudah dipreprocess bisa tetap dibuang
  seluruhnya oleh Faster-Whisper.
- Jika client stop terlalu cepat, final segment bisa belum sempat diterima.
- Jika tidak ada rolling audio/debug archive, transcript live yang halu hanya
  bisa diperbaiki dari teks stabil yang sudah tersimpan.
- Jika model berbeda antara server preload dan client request, server tetap harus
  load model baru sehingga startup cepat tidak terasa.

## Keputusan Saat Ini

- Server preload `WHISPERLIVE_DEFAULT_MODEL=small`.
- Client default `WHISPERLIVE_MODEL=small`.
- Server-side Faster-Whisper VAD default off; client preprocess/VAD menjadi
  filter utama.
- Client default mengirim chunk 500 ms sebagai PCM16.
- Speech boundary detection default on: silence 0.8 detik, max wait 5 detik.
- Local agreement default memakai window 20 detik dan hop 3 detik.
- Transcript Validation Engine default on melalui server segment filter.
- Chunk archive default off; aktifkan hanya dengan `--debug-chunk-archive`.
- `large-v3-turbo` menjadi opsi kualitas, bukan default.
- LLM refinement ditempatkan setelah meeting, bukan di live path.

## Deployment Azure VM

Target deployment production yang disarankan jika VM memiliki GPU:

```text
Client laptop/PC
  - capture mic + speaker
  - preprocess 16 kHz mono PCM16
  - stream WebSocket ke server

Azure VM
  - Docker + NVIDIA runtime
  - container WhisperLive
  - expose WebSocket 9090
  - preload model small
```

Jika VM tidak memiliki GPU, server tetap bisa berjalan dengan CPU. Konsekuensinya:

- preload dan inferensi lebih lambat;
- kapasitas client lebih kecil;
- gunakan model `small` dulu, turun ke `base` jika CPU tidak kuat;
- jangan install `nvidia-smi` jika hardware NVIDIA memang tidak ada.

Output berikut berarti VM CPU-only:

```text
lspci | grep -i vga
# kosong

sudo lshw -C display
product: hyperv_drmdrmfb

nvidia-smi
Command 'nvidia-smi' not found
```

`hyperv_drmdrmfb` adalah framebuffer virtual Hyper-V, bukan GPU untuk CUDA.

IP publik VM saat ini:

```text
20.189.120.244
```

Port yang perlu terbuka:

| Port | Fungsi |
| --- | --- |
| `22` | SSH ke VM. |
| `80` | HTTP jika nanti memakai reverse proxy. |
| `443` | HTTPS/WSS jika nanti memakai TLS reverse proxy. |
| `9090` | WebSocket WhisperLive langsung dari client. |

### Deploy Server di VM

SSH ke VM:

```bash
ssh <user>@20.189.120.244
```

Install dependency dasar di VM Ubuntu:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
```

Install Docker jika belum ada:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Logout/login ulang setelah `usermod`, lalu cek:

```bash
docker version
docker compose version
```

Untuk VM GPU, pastikan NVIDIA driver dan NVIDIA Container Toolkit tersedia:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Jika command di atas gagal karena VM tidak punya GPU, gunakan deploy CPU di
bagian berikutnya.

Clone/copy project ke VM:

```bash
git clone <repo-url> PLN
cd PLN
cp .env.example .env
```

Pastikan `.env` server minimal berisi:

```env
WHISPERLIVE_IMAGE=pln-whisperlive:gpu
WHISPERLIVE_WS_PORT=9090
WHISPERLIVE_DEFAULT_MODEL=small
WHISPERLIVE_MAX_CLIENTS=4
WHISPERLIVE_MAX_CONNECTION_TIME=600
WHISPERLIVE_FORCE_SERVER_PROMPT=true
```

Jalankan server GPU:

```bash
docker compose --env-file .env -f WhisperLive/docker-compose.yml up -d --build
```

### Deploy Server CPU-only

Untuk VM tanpa GPU, gunakan compose CPU:

```bash
cd ~/meet-transcript
cp .env.example .env
```

Edit `.env` untuk CPU:

```env
WHISPERLIVE_CPU_IMAGE=pln-whisperlive:cpu
WHISPERLIVE_DEFAULT_MODEL=small
WHISPERLIVE_CPU_MAX_CLIENTS=2
WHISPERLIVE_CPU_OMP_NUM_THREADS=4
```

Jika VM kecil dan `small` terlalu lambat, ganti:

```env
WHISPERLIVE_DEFAULT_MODEL=base
```

Build dan start server CPU:

```bash
docker compose --env-file .env -f WhisperLive/docker-compose.cpu.yml up -d --build
```

Cek container CPU:

```bash
docker compose -f WhisperLive/docker-compose.cpu.yml ps
docker logs --tail 120 whisperlive-whisperlive-1  
```

Log normal CPU:

```text
Preloading shared faster_whisper model for key=('small', 'cpu', 'int8')
Loading model: small
```

Jangan gunakan `WhisperLive/docker-compose.yml` pada VM CPU-only karena file itu
memakai `gpus: all` dan environment NVIDIA. Gunakan
`WhisperLive/docker-compose.cpu.yml`.

Untuk GPU deploy, cek container:

```bash
docker compose -f WhisperLive/docker-compose.yml ps
docker logs --tail 120 whisperlive-whisperlive-1
```

Log normal saat startup:

```text
Preloading shared faster_whisper model for key=('small', 'cuda', 'float16')
Loading model: small
```

First preload bisa beberapa menit. Server benar-benar siap jika client sudah
mendapat `SERVER_READY`.

### Observability Process Log

Client menulis process log terstruktur ke:

```text
logs/process.log
```

Server Docker menulis process log terstruktur ke volume host:

```text
logs/whisperlive/process.log
```

Gunakan file ini untuk investigasi kasus transcript loncat atau hasil tidak
muncul. Event penting yang dicatat meliputi capture start, VAD pass/drop,
chunk queued/sent, WebSocket ready, audio received, buffer size, boundary
wait/process, ASR start/end, local agreement, TVE score/pending/drop/emit,
dan send ke client.

Live transcript JSON menyimpan dua tingkat hasil:

- `stability=candidate`, `completed=false`: hipotesis live/review dari server.
  Ini sengaja disimpan agar ucapan tidak hilang dan bisa dipakai final
  post-processing/offline refinement.
- `stability=stable`, `completed=true`: segment yang sudah completed melalui
  Local Agreement atau final flush.

Untuk speaker/system audio, default produksi adalah `--no-speaker-client-vad`
karena volume loopback aplikasi sering rendah. Mic tetap memakai client VAD.

Reconnect client aktif secara default untuk mode live. Jika WebSocket putus,
client tidak langsung menghentikan capture; chunk audio per source ditahan di
ring buffer RAM sampai `--reconnect-buffer-seconds` lalu client mencoba reconnect
dengan exponential backoff (`--reconnect-initial-backoff-seconds` sampai
`--reconnect-max-backoff-seconds`). Setelah tersambung lagi, buffer dikirim ulang
secara FIFO sebelum chunk baru dilanjutkan. Event yang perlu dipantau:
`client.chunk_buffered`, `client.reconnect_status`, dan
`client.reconnect_buffer_flushed`.

```bash
tail -f logs/whisperlive/process.log
```

Untuk device headset/laptop, client mendukung pemilihan input dan output:

```bash
uv run python -m src.main --live --mic-device "Headset" --speaker-device "Headset"
```

`--mic-device` memilih input microphone, sedangkan `--speaker-device` memilih
speaker/output loopback yang sedang dipakai meeting. Web UI dan desktop UI
menampilkan daftar mic yang sudah difilter dari alias Windows seperti Sound
Mapper, Primary Sound Capture Driver, Stereo Mix, PC Speaker, dan duplikasi host
API. Endpoint speaker tetap berada di dropdown loopback tersendiri.

Jika log menunjukkan `client.vad_drop` berulang pada mic, jalankan diagnosis
dengan:

```bash
uv run python -m src.main --live --no-mic-client-vad
```

### Firewall VM

Selain Network Security Group Azure, firewall OS di VM juga harus mengizinkan
port `9090`.

Jika memakai `ufw`:

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 9090/tcp
sudo ufw status
```

Uji dari laptop/client:

```powershell
Test-NetConnection 20.189.120.244 -Port 9090
```

Jika `TcpTestSucceeded` false, masalahnya biasanya salah satu dari:

- Azure NSG belum allow inbound `9090`.
- Firewall OS VM belum allow `9090`.
- Container belum running.
- Docker compose tidak publish `0.0.0.0:9090->9090/tcp`.

### Akses dari Client

Client tetap dijalankan di laptop/PC yang menangkap mic dan speaker. Server host
diarahkan ke IP publik VM:

```powershell
cd D:\PLN
uv run python -m src.main --live --server-host 20.189.120.244 --server-port 9090
```

Jika first run server masih preload model, gunakan timeout lebih panjang:

```powershell
uv run python -m src.main --live --server-host 20.189.120.244 --server-port 9090 --server-ready-timeout 600
```

Untuk UI lokal, isi:

| Field | Nilai |
| --- | --- |
| Server Host | `20.189.120.244` |
| WS Port | `9090` |
| Model | `small` |
| Source | `Mic + Speaker` |

Atau jalankan UI:

```powershell
cd D:\PLN
uv run python -m src.ui.server
```

Buka:

```text
http://127.0.0.1:8787
```

### Smoke Test Remote

Tes WebSocket tanpa meeting:

```powershell
cd D:\PLN
uv run python -m src.main --replay-file audio\sample.wav --replay-source mic --server-host 20.189.120.244 --server-port 9090 --server-ready-timeout 600
```

Jika belum punya sample WAV, cukup lakukan test koneksi port dulu dengan
`Test-NetConnection`. Smoke replay membutuhkan file WAV lokal yang valid.

### Mode Aman Production

Expose port `9090` langsung ke internet cukup untuk testing awal, tetapi untuk
production sebaiknya tambahkan salah satu pengaman berikut:

- Batasi inbound `9090` di Azure NSG hanya dari IP kantor/client.
- Pakai VPN/private network antara client dan VM.
- Pakai reverse proxy TLS dan WebSocket secure (`wss://`) di port `443`.
- Aktifkan API key server dan kirim auth header dari client.

Untuk koneksi publik tanpa TLS, audio meeting dikirim melalui WebSocket biasa
(`ws://`). Jangan gunakan pola ini untuk meeting sensitif di luar jaringan yang
dikontrol.
