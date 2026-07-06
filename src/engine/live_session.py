from __future__ import annotations

import logging
import queue
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, perf_counter
from typing import Callable, Literal

import numpy as np

from src.capture.audio_frame import AudioFrame
from src.capture.errors import CaptureError, CaptureNotSupportedError
from src.capture.mic_stream import MicrophoneConfig, MicrophoneStream
from src.capture.win_loopback import WindowsLoopbackConfig, WindowsLoopbackStream
from src.engine.preprocessing import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk
from src.engine.whisper import OpenAIWhisperTranscriber, TranscriptionResult, WhisperConfig
from src.utils.formatter import format_transcript_line, result_to_line
from src.utils.logging import TranscriptLog, log_process_event
from src.utils.os_detector import AudioBackend, get_audio_backend

# def detect_os() -> object:
#     """Compatibility shim untuk test lama; runtime memakai get_audio_backend()."""

#     return get_audio_backend()

_log = logging.getLogger(__name__)

# konfigurasi sesi live transcription
# chunk_seconds: 
#   - berapa detik audio yang dikumpulkan sebelum dikirim ke Whisper.
#   - lebih rendah = latensi lebih rendah, risiko kata terpotong lebih tinggi.
#   - lebih tinggi = latensi lebih tinggi, konteks lebih baik untuk Whisper.
#   - rekomendasi  = 4.0 untuk CPU, 2.0 untuk GPU.

# source:
#  - sumber audio yang akan ditangkap: "mic", "speaker", atau "both".
#  - untuk rapat online, "both" menangkap suara Anda (mic) dan remote participants (speaker).

# max_chunk_queue_size:
#  - berapa banyak chunk yang belum diproses yang akan di-buffer.
#  - ketika penuh, chunk tertua akan dihapus untuk memberi preferensi pada audio terbaru.
@dataclass(frozen=True, slots=True)
class LiveSessionConfig:
    chunk_seconds: float = 4.0
    source: Literal["mic", "speaker", "both"] = "both"
    sample_rate: int | None = None
    block_size: int = 1_024
    queue_size: int = 128
    mic_device: int | str | None = None
    speaker_device: int | str | None = None
    max_chunk_queue_size: int = 16
    max_chunks_processed: int | None = None
    transcript_log_path: Path | None = None
    whisper_config: WhisperConfig | None = None


# menyimpan statistik runtime selama sesi live transcription
@dataclass
class LiveSessionStats:
    chunks_processed: int = 0
    chunks_dropped: int = 0
    results_accepted: int = 0
    results_rejected: int = 0
    start_time: float = field(default_factory=perf_counter)

    # waktu yang telah berlalu sejak sesi dimulai
    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.start_time


# config:
# - konfigurasi sesi live transcription. Jika None, akan menggunakan default.
# on_result:
# - callback opsional yang dipanggil untuk setiap hasil transkripsi yang diterima. Berguna untuk integrasi (misalnya, menulis ke file, mengirim ke UI).

# return:
# - LiveSessionStats: statistik runtime untuk sesi yang telah selesai.  
def run_live_session(
    config: LiveSessionConfig | None = None,
    *,
    on_result: Callable[[TranscriptionResult], None] | None = None,
) -> LiveSessionStats:

    if config is None: cfg = LiveSessionConfig() 
    if cfg.whisper_config is None: whisper_cfg = WhisperConfig()
    stats = LiveSessionStats()

    # Transcrip log ini akan menyimpan hasil transkripsi secara real-time ke file, sehingga jika terjadi crash atau penutupan mendadak, hasil transkripsi tetap tersimpan.
    transcript_log: TranscriptLog | None = None
    if cfg.transcript_log_path:
        transcript_log = TranscriptLog(cfg.transcript_log_path)
        transcript_log.save()

    # Shared state
    # untuk menghentikan sesi live transcription, kita menggunakan stop_event. Ketika stop_event diset, semua thread akan berhenti.
    stop_event = threading.Event()
    chunk_queue: queue.Queue[PreprocessedAudioChunk] = queue.Queue(
        maxsize=cfg.max_chunk_queue_size
    )

    # Capture streams & reader threads                                    #
    capture_streams: list = []
    reader_threads: list[threading.Thread] = []
    preprocess_config = PreprocessConfig(chunk_seconds=cfg.chunk_seconds)

    # (mic, both) berarti jika cfg.source adalah mic atau both, maka akan membuat stream mikrofon. 
    if cfg.source in ("mic", "both"):
        try:
            # Membuat stream mikrofon
            mic_stream = MicrophoneStream(
                MicrophoneConfig(
                    sample_rate=cfg.sample_rate,
                    block_size=cfg.block_size,
                    device=cfg.mic_device,
                    queue_size=cfg.queue_size,
                )
            )

            # Menambahkan stream mikrofon ke daftar capture_streams dan membuat thread pembaca untuk memproses frame audio dari mikrofon.
            capture_streams.append(mic_stream)
            reader_threads.append(
                threading.Thread(
                    target=_reader_thread,
                    args=(
                        mic_stream,
                        AudioPreprocessor(preprocess_config),
                        chunk_queue,
                        cfg.chunk_seconds,
                        stop_event,
                        stats,
                    ),
                    name="live-mic-reader",
                    daemon=True,
                )
            )

            _log.info("live session: mic stream created")
        except Exception as exc:  # pragma: no cover
            _log.warning("live session: mic stream unavailable — %s", exc)

    # (speaker, both) berarti jika cfg.source adalah speaker atau both, maka akan mencoba membuat stream loopback speaker.
    if cfg.source in ("speaker", "both"):
        audio_backend = get_audio_backend()
        if audio_backend is AudioBackend.WASAPI_LOOPBACK:
            try:
                spk_stream = WindowsLoopbackStream(
                    WindowsLoopbackConfig(
                        sample_rate=cfg.sample_rate,
                        block_size=cfg.block_size,
                        device=cfg.speaker_device,
                        queue_size=cfg.queue_size,
                    )
                )
                capture_streams.append(spk_stream)
                reader_threads.append(
                    threading.Thread(
                        target=_reader_thread,
                        args=(
                            spk_stream,
                            AudioPreprocessor(preprocess_config),
                            chunk_queue,
                            cfg.chunk_seconds,
                            stop_event,
                            stats,
                        ),
                        name="live-speaker-reader",
                        daemon=True,
                    )
                )
                _log.info("live session: speaker loopback stream created")
            except CaptureNotSupportedError as exc:
                _log.warning(
                    "live session: speaker loopback unavailable — %s\n"
                    "  Tip: pastikan 'Stereo Mix' atau loopback device aktif di Sound Settings.",
                    exc,
                )
        else:
            # Jika backend audio tidak mendukung loopback speaker, maka akan menampilkan peringatan bahwa hanya mikrofon yang didukung.
            _log.warning(
                "live session: speaker capture not supported on %s — mic only",
                audio_backend.value,
            )

    if not capture_streams:
        _log.error("live session: no capture streams available — aborting")
        print("[ERROR] Tidak ada audio stream yang tersedia. Periksa perangkat audio Anda.")
        return stats

    # handel ketika Ctrl+C maka akan menghentikan sesi live transcription
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigbreak = signal.getsignal(signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else None

    def _sigint_handler(signum: int, frame: object) -> None:
        _log.info("live session: stop signal received signal=%s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sigint_handler)

    # Start streams and reader threads                                    #
    active_streams: list = []
    for stream in capture_streams:
        try:
            stream.start()
            active_streams.append(stream)
            _log.info("live session: started stream %s", getattr(stream, "device", stream))
        except CaptureError as exc:
            _log.warning("live session: failed to start stream: %s", exc)

    if not active_streams:
        print("[ERROR] Gagal memulai audio stream. Periksa izin mikrofon / audio device.")
        signal.signal(signal.SIGINT, original_sigint)
        return stats

    # setiap thread pembaca mereka mulai membaca frame audio dari stream masing-masing dan memprosesnya menjadi chunk audio.
    for thread in reader_threads:
        thread.start()

    # Transcription loop (main thread — serialised, thread-safe) 
    transcriber = OpenAIWhisperTranscriber(whisper_cfg)

    source_labels = {"mic": "MIC", "speaker": "SPEAKER"}
    session_start_wall = monotonic() # waktu mulai sesi live transcription untuk menghitung durasi sesi.

    print("=" * 60)
    print("Live transcription aktif. Tekan Ctrl+C untuk berhenti.")
    source_list = ", ".join(
        source_labels.get(s, s.upper())
        for s in (["mic", "speaker"] if cfg.source == "both" else [cfg.source])
    )
    print(f"Sumber: {source_list}  |  Chunk: {cfg.chunk_seconds:.0f}s  |  Model: {whisper_cfg.model_name}")
    print("=" * 60)

    try:
        while not stop_event.is_set():
            try:
                chunk = chunk_queue.get(timeout=0.5) # mengambil chunk dalam antrian dengan timeout agar thread tidak terblokir selamanya jika tidak ada chunk yang tersedia.
            except queue.Empty:
                continue

            _log.debug(
                "live session: transcribing chunk source=%s duration=%.2fs",
                chunk.source,
                chunk.duration_seconds,
            )

            result = transcriber.transcribe_chunk(chunk)    # memproses chunk audio dengan Whisper.
            stats.chunks_processed += 1                     # increment jumlah chunk yang telah diproses

            if result.text:
                stats.results_accepted += 1                                 # increment jumlah hasil transkripsi yang diterima
                line = format_transcript_line(result_to_line(result))       # format hasil transkripsi
                print(line, flush=True)                                     # menampilkan hasil transkripsi ke terminal
                
                # transcript log aktif
                if transcript_log is not None:                             
                    transcript_log.append_result(result)

                # 
                if on_result is not None:
                    on_result(result)
                _log.info("live session: accepted source=%s text=%r", result.source, result.text[:80])
            else:
                stats.results_rejected += 1
                _log.debug(
                    "live session: rejected source=%s warning=%s",
                    result.source,
                    result.warning,
                )

            # jika sudah melebihi batas maksimal chunk yang diproses, maka akan menghentikan sesi live transcription.
            if cfg.max_chunks_processed is not None and stats.chunks_processed >= cfg.max_chunks_processed:
                stop_event.set()

    # mengakhiri sesi live transcription ketika Ctrl+C ditekan atau terjadi KeyboardInterrupt.
    finally:
        stop_event.set()                # diset untuk menghentikan semua thread pembaca dan stream audio.
        
        #  stream audio yang aktif
        for stream in active_streams:
            try:
                stream.stop() # hentikan stream audio
            except Exception as exc: 
                _log.debug("live session: error stopping stream: %s", exc)

        # thread pembaca yang aktif  
        for thread in reader_threads:
            thread.join(timeout=5.0) # menunggu thread pembaca selesai dengan timeout 5 detik. Jika thread tidak selesai dalam waktu tersebut, maka akan dilanjutkan tanpa menunggu lebih lama.
        
        # mengembalikan handler sinyal Ctrl+C ke handler asli agar tidak mengganggu program lain.
        signal.signal(signal.SIGINT, original_sigint)
        if hasattr(signal, "SIGBREAK") and original_sigbreak is not None:
            signal.signal(signal.SIGBREAK, original_sigbreak)

        elapsed = monotonic() - session_start_wall # menghitung durasi sesi live transcription

        print("\n" + "=" * 60)
        print(f"Sesi selesai. Durasi: {elapsed:.0f}s")
        print(
            f"Chunks diproses: {stats.chunks_processed}  |  "
            f"Diterima: {stats.results_accepted}  |  "
            f"Ditolak: {stats.results_rejected}  |  "
            f"Dropped: {stats.chunks_dropped}"
        )
        if transcript_log is not None:
            print(f"Transcript log: {cfg.transcript_log_path}")
        print("=" * 60)

    return stats


# akumulasi frame dari *stream*, preprocess, push ke *chunk_queue*.
# fungsi ini berjalan di thread daemon sendiri. Tidak sengaja merujuk model Whisper — hanya penangkapan audio dan preprocessing yang terjadi di sini.  Thread-safety dipertahankan dengan:
#   - Hanya menulis ke *chunk_queue* (Queue thread-safe).
#   - Hanya membaca *stop_event.is_set()* (bool-like atomik).
#   - *preprocessor* TIDAK dibagikan antar thread.

def _reader_thread(
    stream: object,
    preprocessor: AudioPreprocessor,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    chunk_seconds: float,
    stop_event: threading.Event,
    stats: LiveSessionStats,
) -> None:
    buffer: list[AudioFrame] = []
    source_name = "unknown"
    frames_seen = 0
    chunks_enqueued = 0
    first_frame_logged = False
    dropped_by_preprocess = 0

    # loop pembaca frame audio dari stream. 
    while not stop_event.is_set():
        frame = stream.read_frame(timeout=0.1)  # membaca frame audio dari stream dengan timeout 0.1 detik. jika terlalu cepat membebani cpu
        if frame is None:
            continue

        source_name = frame.source  # sumber audio dari frame
        frames_seen += 1 # increment jumlah frame yang telah dibaca dari stream    
        
        # frame pertama belum dicatat, maka akan mencatat event "client.capture_start" ke log proses dan menampilkan informasi frame pertama ke log. Hal ini untuk debugging dan monitoring performa sesi live transcription.
        if not first_frame_logged:
            log_process_event(
                "client.capture_start",
                source=frame.source,
                sample_rate=frame.sample_rate,
                channels=frame.channels,
                frame_count=frame.frame_count,
                frame_rms=round(_frame_rms(frame), 6),
                status=frame.status or "",
            )
            _log.info(
                "live session: first frame source=%s sample_rate=%s channels=%s frames=%s rms=%.6f status=%s",
                frame.source,
                frame.sample_rate,
                frame.channels,
                frame.frame_count,
                _frame_rms(frame),
                frame.status or "",
            )
            first_frame_logged = True
        buffer.append(frame)

        # Check durasi total buffer. Jika durasi total buffer lebih kecil dari chunk_seconds, maka akan melanjutkan membaca frame audio dari stream. Jika durasi total buffer lebih besar atau sama dengan chunk_seconds, maka akan memproses buffer menjadi chunk audio dan mengirimkannya ke chunk_queue.
        total_duration = sum(f.duration_seconds for f in buffer)
        if total_duration < chunk_seconds:
            continue

        input_rms = _buffer_rms(buffer)                             # menghitung rms unuk mengetahui kekuatan audio
        input_frames = sum(frame.frame_count for frame in buffer)   # menghitung jumlah frame audio yang ada di buffer
        
        log_process_event(
            "client.chunk_created",
            source=source_name,
            duration_seconds=round(total_duration, 3),
            input_frames=input_frames,
            input_rms=round(input_rms, 6),
            queue_size=chunk_queue.qsize(),
        )
        
        chunks = preprocessor.preprocess_frames(buffer) # meggabungkan frame audio di buffer menjadi chunk audio yang siap diproses lebih lanjut.
        buffer = [] # mengosongkan buffer setelah frame audio diubah menjadi chunk audio. 

        for decision in preprocessor.last_decisions:
            event = "client.vad_pass" if decision.get("passed") else "client.vad_drop"
            log_process_event(event, **decision, queue_size=chunk_queue.qsize())
        
        if not chunks:
            dropped_by_preprocess += 1
            _log.info(
                "live session: preprocess produced no chunks source=%s duration=%.3f input_frames=%s input_rms=%.6f dropped_windows=%s",
                source_name,
                total_duration,
                input_frames,
                input_rms,
                dropped_by_preprocess,
            )
            continue

        for chunk in chunks:
            try:
                chunk_queue.put_nowait(chunk) # menambahkan chunk audio ke chunk_queue tanpa menunggu jika antrian penuh.
                chunks_enqueued += 1 # increment jumlah chunk yang telah dimasukkan ke chunk_queue

                log_process_event(
                    "client.chunk_queued",
                    source=chunk.source,
                    start_seconds=round(chunk.start_seconds, 3),
                    duration_seconds=round(chunk.duration_seconds, 3),
                    frame_count=chunk.frame_count,
                    input_rms_db=round(chunk.input_rms_db, 2),
                    output_rms_db=round(chunk.rms_db, 2),
                    queue_size=chunk_queue.qsize(),
                    chunks_enqueued=chunks_enqueued,
                )
                _log.debug(
                    "live session: queued chunk source=%s duration=%.3f input_rms_db=%.2f output_rms_db=%.2f queued=%s",
                    chunk.source,
                    chunk.duration_seconds,
                    chunk.input_rms_db,
                    chunk.rms_db,
                    chunks_enqueued,
                )
            
            # jika antrian penuh, maka akan menghapus chunk tertua dari antrian dan menambahkan chunk baru ke antrian.
            except queue.Full:
                try:
                    chunk_queue.get_nowait() # mengambil data dari antrean secara langsung tanpa menunggu (queue FIFO) jadi yang didapatkan adalah chunk tertua. 
                except queue.Empty:
                    pass
                try:
                    chunk_queue.put_nowait(chunk) # menambahkan chunk baru ke antrian tanpa menunggu jika antrian penuh.
                    chunks_enqueued += 1
                    stats.chunks_dropped += 1

                    log_process_event(
                        "client.chunk_queue_drop",
                        source=source_name,
                        reason="queue_full_drop_oldest",
                        queue_size=chunk_queue.qsize(),
                        chunks_dropped=stats.chunks_dropped,
                    )
                    _log.warning(
                        "live session: chunk queue full — dropped oldest %s chunk", source_name
                    )
                
                except queue.Full:
                    pass
    
    # ketika buffer tidak kosong, maka akan memproses buffer menjadi chunk audio terakhir sebelum thread pembaca keluar. Hal ini untuk memastikan bahwa semua frame audio yang telah dibaca dari stream diproses menjadi chunk audio dan dikirim ke chunk_queue.
    if buffer:
        partial_duration = sum(f.duration_seconds for f in buffer) # durasi total buffer terakhir
        partial_frames = sum(frame.frame_count for frame in buffer) # jumlah frame audio terakhir
        partial_rms = _buffer_rms(buffer) # menghitung rms untuk mengetahui kekuatan audio terakhir

        _log.info(
            "live session: reader exiting with partial buffer source=%s duration=%.3f frames=%s rms=%.6f",
            source_name,
            partial_duration,
            partial_frames,
            partial_rms,
        )
        
        # ika durasi buffer terakhir cukup panjang, maka diproses menjadi chunk audio terakhir. Jika durasi buffer terakhir terlalu pendek, maka akan diabaikan.
        if partial_duration >= preprocessor.config.min_chunk_seconds:

            # untuk setiap chunk audio yang dihasilkan dari buffer terakhir
            for chunk in preprocessor.preprocess_frames(buffer):
                for decision in preprocessor.last_decisions:
                    event = "client.vad_pass" if decision.get("passed") else "client.vad_drop"
                    log_process_event(event, final_partial=True, **decision, queue_size=chunk_queue.qsize())
                try:
                    chunk_queue.put_nowait(chunk) # menambahkan chunk audio terakhir ke antrian.
                    chunks_enqueued += 1 # increment jumlah chunk yang telah dimasukkan ke chunk_queue

                    log_process_event(
                        "client.chunk_queued",
                        source=chunk.source,
                        start_seconds=round(chunk.start_seconds, 3),
                        duration_seconds=round(chunk.duration_seconds, 3),
                        frame_count=chunk.frame_count,
                        input_rms_db=round(chunk.input_rms_db, 2),
                        output_rms_db=round(chunk.rms_db, 2),
                        queue_size=chunk_queue.qsize(),
                        chunks_enqueued=chunks_enqueued,
                        final_partial=True,
                    )
                    _log.info(
                        "live session: queued final partial chunk source=%s duration=%.3f input_rms_db=%.2f output_rms_db=%.2f",
                        chunk.source,
                        chunk.duration_seconds,
                        chunk.input_rms_db,
                        chunk.rms_db,
                    )
                
                except queue.Full:
                    stats.chunks_dropped += 1
                    log_process_event(
                        "client.chunk_queue_drop",
                        source=source_name,
                        reason="final_partial_queue_full",
                        queue_size=chunk_queue.qsize(),
                        chunks_dropped=stats.chunks_dropped,
                    )
                    _log.warning("live session: final partial chunk dropped because queue is full source=%s", source_name)
    _log.debug(
        "live session: reader thread exiting (source=%s frames_seen=%s chunks_enqueued=%s dropped_windows=%s)",
        source_name,
        frames_seen,
        chunks_enqueued,
        dropped_by_preprocess,
    )


def _frame_rms(frame: AudioFrame) -> float:
    if frame.samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.samples.astype(np.float32)))))


def _buffer_rms(frames: list[AudioFrame]) -> float:
    samples = [frame.samples.astype(np.float32).reshape(-1) for frame in frames if frame.samples.size]
    if not samples:
        return 0.0
    joined = np.concatenate(samples)
    if joined.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(joined))))
