# Analisis Transkrip Speaker

Tanggal analisis: 2026-07-04

Fokus analisis ini adalah jalur `speaker`, yaitu audio peserta meeting yang
ditangkap dari output speaker/headset melalui loopback, dipreprocess di client,
dikirim ke WhisperLive, lalu disimpan sebagai transcript `Meeting`.

## Ringkasan Eksekutif

Pipeline speaker sudah berjalan end-to-end. Dari log runtime terakhir, capture
speaker aktif, sinyal audio terdeteksi, 111 chunk berhasil dikirim ke server,
tidak ada chunk dropped, dan server mengembalikan transcript candidate serta
stable.

Masalah utama bukan koneksi atau format audio, melainkan kualitas finalisasi
transcript:

- Banyak hasil speaker masuk sebagai `candidate` dengan `reliability_action=review`.
- Stable transcript yang keluar masih fragmental pada beberapa bagian.
- Sebagian besar ucapan panjang baru stabil saat final flush.
- Local Agreement bekerja, tetapi cenderung hanya mengunci potongan kecil saat
  pembicaraan berjalan panjang.
- Normalisasi speaker ke target RMS -20 dB membantu audibilitas, tetapi juga
  bisa memperkuat noise/silence karena client VAD speaker default dimatikan.

Prioritas optimasi pertama sebaiknya bukan mengganti seluruh arsitektur, tetapi
menguatkan observability speaker, memperbaiki timebase, menyediakan arsip audio
ring-buffer untuk reprocess, lalu men-tuning Local Agreement dan speech boundary
khusus speaker.

## Jalur Speaker Saat Ini

1. CLI `--live --source speaker` membuat `WhisperLiveSessionConfig`.
2. Client membuka satu WebSocket untuk source `speaker`.
3. Capture speaker memakai `WindowsLoopbackStream`.
4. Audio loopback masuk sebagai float32 stereo 48 kHz.
5. Preprocessing mengubah audio menjadi mono 16 kHz, high-pass, normalisasi RMS,
   lalu memotong chunk 0,5 detik.
6. Chunk dikirim ke server sebagai PCM16 (`audio_format=int16`).
7. Server mengubah PCM16 ke float32 dan memasukkan ke buffer ASR.
8. WhisperLive memakai Faster-Whisper `small`, bahasa `id`, Local Agreement,
   speech boundary detection, dan Transcript Validation Engine.
9. Client menyimpan candidate untuk audit dan stable untuk output utama.

Referensi kode:

- Speaker loopback: `src/capture/win_loopback.py`
- Preprocessing: `src/engine/preprocessing.py`
- Live WhisperLive session: `src/engine/whisperlive_session.py`
- WebSocket client: `src/engine/whisperlive_client.py`
- Server audio decode/final flush: `WhisperLive/whisper_live/server.py`
- Local Agreement/final flush: `WhisperLive/whisper_live/backend/base.py`
- Transcript validation: `WhisperLive/whisper_live/postprocessing.py`

## Bukti Runtime Terakhir

File yang dianalisis:

- `audio/transcript_log.json`
- `logs/process.log`
- `logs/transcriber.log`
- `logs/whisperlive/process.log`

Session terakhir berjalan pada 2026-07-04 12:07:40 sampai 12:08:43 waktu lokal.

Konfigurasi terdeteksi:

- Source: `speaker`
- Model: `small`
- Language: `id`
- Audio format: `int16`
- Client chunk: `0.5` detik
- Server VAD: off
- Local Agreement: on
- Local Agreement window: `20` detik
- Local Agreement hop: `3` detik
- Speech boundary: on
- Speaker client VAD: off
- Capture sample rate: `48000`
- Channels: `2`

Ringkasan angka:

| Metrik | Nilai |
| --- | ---: |
| Chunk speaker dibuat | 111 |
| Chunk speaker dikirim | 111 |
| Chunk dropped | 0 |
| Bytes terkirim | 1.776.000 |
| Transcript received event | 12 |
| Segment diterima client | 16 |
| Candidate tersimpan | 11 |
| Stable tersimpan | 5 |
| ASR run di server | 12 |
| Rata-rata input RMS chunk | sekitar -33.07 dB |
| Rentang input RMS chunk | -39.51 dB sampai -29.67 dB |
| Latency ASR per run | sekitar 0.43 sampai 1.04 detik |

Kesimpulan dari angka ini:

- Capture speaker tidak kosong.
- Koneksi WebSocket dan format PCM16 sudah cocok.
- Throughput server cukup baik untuk model `small`.
- Bottleneck kualitas ada pada segment stabilization dan akurasi ASR, bukan pada
  pengiriman audio.

## Temuan Teknis

### 1. Speaker capture berhasil dan device benar terdeteksi

Log menunjukkan device loopback yang dipakai adalah `Speakers (Realtek(R)
Audio)`, channel 2, sample rate 48 kHz. First-batch RMS `0.042680`, jadi sinyal
speaker memang masuk.

Risiko:

- Jika meeting memakai headset/output lain, default speaker Windows bisa berbeda
  dari output meeting. UI/CLI sudah mendukung `--speaker-device`, tetapi perlu
  validasi lebih eksplisit saat speaker transcript kosong.

Rekomendasi:

- Tambahkan command diagnostik atau endpoint UI yang menampilkan level meter
  speaker real-time sebelum sesi dimulai.
- Simpan `selected_device.name`, `sample_rate`, `channels`, dan first 5 RMS
  readings ke process log.

### 2. Sample rate speaker dipaksa 48 kHz

`WindowsLoopbackStream` memakai `self.config.sample_rate or 48_000`. Ini aman
untuk banyak endpoint Windows, tetapi tidak selalu mengikuti default samplerate
perangkat.

Risiko:

- Jika endpoint berjalan di 44.1 kHz atau driver melakukan resampling internal,
  bisa ada artefak kecil atau drift timing.

Rekomendasi:

- Ambil sample rate dari metadata device jika tersedia, baru fallback ke 48 kHz.
- Tulis sample rate aktual dari recorder ke log, bukan hanya konfigurasi.

### 3. Timestamp chunk memakai `perf_counter`, bukan timebase session

Frame speaker diberi `timestamp_seconds=perf_counter()`. Di log, start chunk
client berada di angka besar seperti `12181.104`, sementara transcript server
dimulai dari `0.0`.

Risiko:

- Untuk single-source speaker, ini tidak merusak ASR.
- Untuk merge mic + speaker, timebase berbeda antar stream bisa membuat ordering
  tidak presisi jika salah satu source memakai timestamp dari clock berbeda.

Rekomendasi:

- Normalisasi timestamp capture menjadi offset relatif terhadap session start.
- Simpan dua field terpisah: `capture_clock_seconds` dan `session_seconds`.
- Merger sebaiknya memakai `session_seconds` atau timestamp server yang sudah
  relatif per source secara konsisten.

### 4. Speaker client VAD default off, tetapi log masih mencatat vad_pass/drop

Saat VAD dimatikan melalui `_without_client_vad`, threshold dibuat nol dan
`min_input_rms_db=-inf`. Karena itu setiap chunk 0,5 detik akan `vad_pass`,
sedangkan sisa 0,012 detik selalu `too_short`.

Risiko:

- Nama event `client.vad_pass` bisa menyesatkan karena sebenarnya bukan VAD
  speech yang aktif.
- Silence/noise speaker tetap dikirim dan kemudian dinormalisasi ke -20 dB.

Rekomendasi:

- Tambahkan field `client_vad_enabled=false` pada keputusan preprocessing.
- Ubah reason menjadi `vad_disabled_accepted` saat VAD dimatikan.
- Pertimbangkan VAD speaker ringan berbasis RMS adaptif, bukan threshold statis,
  agar silence murni tidak diperkuat dan dikirim.

### 5. Normalisasi RMS dapat memperkuat noise speaker

Preprocessor menargetkan RMS output -20 dB dengan max gain 24 dB. Pada session
terakhir, input RMS speaker berkisar -39.51 sampai -29.67 dB, lalu dinaikkan ke
sekitar -20 dB.

Manfaat:

- Audio meeting pelan menjadi lebih mudah ditranskrip.

Risiko:

- Noise ruangan, hiss, atau audio lemah non-speech ikut menguat.
- TVE menjadi sering memberi action `review` karena sinyal terdengar seperti
  speech tetapi confidence tidak cukup kuat.

Rekomendasi:

- Pisahkan profil normalisasi mic dan speaker.
- Untuk speaker, coba target RMS -23 dB atau -24 dB dan max gain 18 dB.
- Tambahkan noise floor estimator per 3-5 detik untuk membedakan silence dari
  speech pelan.

### 6. Local Agreement menghasilkan candidate panjang dan stable fragmental

Transcript log menunjukkan 11 candidate dan hanya 5 stable. Beberapa candidate
panjang 15-20 detik diberi `review`, lalu stable yang muncul bisa hanya potongan
pendek seperti `Kan kemarin` dan `datang sini pakai kacamata. Iya betul. Jadi`.

Risiko:

- Output utama user terasa tidak lengkap saat live.
- Stable transcript terlambat dan sebagian baru muncul pada final flush.

Rekomendasi:

- Untuk speaker, uji `local_agreement_hop_seconds=2.0` agar stabilisasi lebih
  sering.
- Uji `speech_boundary_max_wait_seconds=3.5` untuk mengurangi window 20 detik
  yang terlalu panjang saat pembicara terus bicara.
- Pertimbangkan policy "emit reviewed candidate as provisional" di UI, tetapi
  tetap bedakan jelas dari stable final.
- Jangan hanya mengandalkan stable untuk final transcript; gunakan candidate
  sebagai bahan reprocess/post-meeting.

### 7. TVE tidak banyak drop, tetapi banyak review

Server log menunjukkan `server.tve_score` dan `server.tve_emit` sebanyak 16,
tanpa `server.tve_pending/drop` pada session ini. Candidate review tetap dikirim
ke client karena skornya berada di sekitar 0.718-0.758.

Interpretasi:

- TVE tidak memblokir hasil speaker.
- Masalahnya adalah confidence/stability belum cukup tinggi, bukan filter terlalu
  agresif.

Rekomendasi:

- Jangan langsung menurunkan threshold TVE.
- Lebih dulu perbaiki chunking, speech boundary, prompt/glossary, dan kualitas
  input.

### 8. Final flush bekerja, tetapi close remote dilog sebagai warning

Setelah `END_OF_AUDIO`, server mengirim hasil final dan menutup WebSocket. Client
mencatat `CLIENT_REMOTE_CLOSED` sebagai warning.

Risiko:

- Ini terlihat seperti error padahal normal setelah finalisasi.

Rekomendasi:

- Jika `_audio_finished=True`, treat remote close sebagai normal/INFO.
- Tambahkan status `CLIENT_REMOTE_CLOSED_AFTER_EOS`.

### 9. Typo field process log menyulitkan observability

Di `client.capture_backend` untuk speaker, field bernama `backenn_audio`, bukan
`backend_audio`.

Risiko:

- Query log/monitoring sulit konsisten.

Rekomendasi:

- Perbaiki field menjadi `backend_audio`.
- Untuk kompatibilitas sementara, boleh tulis dua field selama satu versi.

### 10. Tidak ada arsip audio default untuk validasi kualitas

Live transcript tersedia, tetapi audio mentah/preprocessed tidak tersimpan secara
default kecuali `--debug-chunk-archive` aktif.

Risiko:

- Saat transcript speaker salah, sulit membedakan apakah masalah dari capture,
  preprocessing, ASR, atau postprocessing.

Rekomendasi:

- Tambahkan rolling audio buffer opsional per source, misalnya 5-10 menit,
  ditulis sebagai WAV segment besar, bukan ribuan file kecil.
- Simpan audio preprocessed 16 kHz mono untuk replay cepat.
- Gunakan buffer ini untuk final reprocess setelah meeting.

## Rekomendasi Prioritas

### Prioritas 1 - Observability dan diagnostik speaker

Tujuan: memastikan setiap masalah speaker bisa diklasifikasi cepat.

Perubahan yang disarankan:

- Perbaiki typo `backenn_audio` menjadi `backend_audio`.
- Tambahkan `client_vad_enabled` pada setiap VAD/preprocess decision.
- Tambahkan level meter speaker di UI/CLI: RMS saat ini, peak, device, sample
  rate, channels.
- Treat remote close setelah EOS sebagai normal.
- Tambahkan ringkasan session: candidate/stable count, avg RMS, ASR latency,
  completed ratio.

### Prioritas 2 - Kualitas input speaker

Tujuan: mengurangi noise yang ikut diperkuat.

Perubahan yang disarankan:

- Buat konfigurasi preprocessing khusus speaker:
  - target RMS -23 dB atau -24 dB,
  - max gain 18 dB,
  - adaptive silence gate ringan.
- Tambahkan opsi CLI/env:
  - `PLN_SPEAKER_TARGET_RMS_DB`
  - `PLN_SPEAKER_MAX_GAIN_DB`
  - `PLN_SPEAKER_MIN_RMS_DB`
- Uji A/B dari file replay speaker yang sama.

### Prioritas 3 - Stabilisasi transcript speaker

Tujuan: stable transcript lebih lengkap dan tidak terlalu terlambat.

Eksperimen yang disarankan:

- `--local-agreement-hop-seconds 2.0`
- `--speech-boundary-max-wait-seconds 3.5`
- `--speech-boundary-silence-seconds 1.0`
- `--whisper-model medium` atau `large-v3-turbo` jika GPU cukup

Kriteria sukses:

- Rasio stable/candidate naik.
- Stable tidak banyak berupa fragmen 1-3 kata.
- Latency tetap dalam target 2-5 detik.

### Prioritas 4 - Final transcript berbasis replay audio

Tujuan: live tetap responsif, final transcript lebih bersih.

Perubahan yang disarankan:

- Simpan rolling preprocessed WAV per source.
- Setelah meeting, replay/reprocess dengan window lebih besar dan model lebih
  akurat.
- Gabungkan candidate + stable + hasil reprocess.
- Post-process hanya untuk punctuation, kapitalisasi, paragraphing, dedup, dan
  ringkasan. Jangan gunakan LLM untuk menebak kata yang tidak terdengar.

## Checklist Investigasi Speaker Berikutnya

Saat hasil speaker buruk, cek urutan ini:

1. `logs/transcriber.log`: device loopback yang dipilih dan first-batch RMS.
2. `logs/process.log`: `client.capture_start`, `client.chunk_sent`,
   `client.transcript_received`.
3. `logs/whisperlive/process.log`: `server.audio_received`, `server.asr_end`,
   `server.local_agreement_result`, `server.tve_score`.
4. `audio/transcript_log.json`: jumlah `candidate` vs `stable`.
5. Jika perlu, jalankan ulang dengan `--debug-chunk-archive` atau rolling WAV.

## Kesimpulan

Speaker transcript sudah hidup secara teknis. Masalah yang terlihat sekarang
adalah kualitas final/stable transcript: kandidat cukup banyak, tetapi finalisasi
masih terlambat dan kadang fragmental. Fokus optimasi pertama sebaiknya:

1. memperjelas observability speaker,
2. mengurangi noise yang diperkuat oleh normalisasi,
3. men-tuning Local Agreement/speech boundary khusus speaker,
4. menambahkan audio replay buffer untuk final transcript yang lebih akurat.
