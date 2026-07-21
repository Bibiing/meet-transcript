from __future__ import annotations

import logging
import queue
import threading
from dataclasses import replace
from typing import TYPE_CHECKING

from src.capture.errors import CaptureError, CaptureNotSupportedError
from src.capture.mac_sys_audio import MacSystemAudioConfig, MacSystemAudioStream
from src.capture.mic_stream import MicrophoneConfig, MicrophoneStream
from src.capture.win_loopback import WindowsLoopbackConfig, WindowsLoopbackStream
from src.capture.models import AudioFrame
from src.preprocessing.core import AudioPreprocessor, PreprocessConfig, PreprocessedAudioChunk
import numpy as np
from src.utils.logging import log_process_event
from src.utils.status_ipc import emit_status
from src.utils.os_detector import AudioBackend, get_audio_backend
from src.whisper.models import WhisperLiveSessionConfig, WhisperLiveSessionStats

if TYPE_CHECKING:
    from src.preprocessing.vad import SileroVADFilter

_log = logging.getLogger(__name__)


def _build_client_vad_filter(cfg: WhisperLiveSessionConfig) -> "SileroVADFilter | None":
    """Bangun satu SileroVADFilter untuk satu source, atau None bila nonaktif.

    Dipanggil sekali per source di _build_capture_streams sehingga mic dan
    speaker memiliki instance terpisah (state RNN tidak boleh dibagi). Import
    dilakukan lazy agar saat fitur nonaktif, modul VAD/onnxruntime tak disentuh.
    """
    if not cfg.client_vad_enabled:
        return None
    from src.preprocessing.vad import SileroVADFilter, default_model_path

    model_path = cfg.client_vad_model_path or default_model_path()
    return SileroVADFilter(
        model_path=model_path,
        threshold=cfg.client_vad_threshold,
        hangover_chunks=cfg.client_vad_hangover_chunks,
    )


def _apply_client_vad(
    vad_filter: "SileroVADFilter | None",
    chunks: list[PreprocessedAudioChunk],
    stats: WhisperLiveSessionStats,
    source: str,
    queue_size: int,
) -> list[PreprocessedAudioChunk]:
    """Admission control: buang chunk non-speech sebelum masuk queue (ADR-001).

    Filter dimiliki eksklusif oleh satu reader thread (satu per source) sehingga
    state RNN-nya tidak perlu lock. Bila filter None (nonaktif) atau internal
    disabled (graceful degradation), seluruh chunk lolos. Chunk yang dibuang
    tidak dikirim; SentTimeline mengompensasi drift waktunya di sisi client.
    """
    if vad_filter is None:
        return chunks
    admitted: list[PreprocessedAudioChunk] = []
    for chunk in chunks:
        if vad_filter.is_speech(chunk):
            admitted.append(chunk)
            continue
        stats.add_vad_dropped()
        log_process_event(
            "client.vad_drop",
            source=source,
            start_seconds=round(chunk.start_seconds, 3),
            duration_seconds=round(chunk.duration_seconds, 3),
            output_rms_db=round(chunk.rms_db, 2),
            client_vad_dropped=stats.client_vad_dropped,
            queue_size=queue_size,
        )
    return admitted

# stream capture mic/speaker beserta reader thread-nya digunakan untuk menginisialisasi stream audio dari mikrofon dan speaker, serta membuat thread pembaca yang akan memproses audio dari hardware ke dalam antrian chunk.
def _build_capture_streams(
    cfg: WhisperLiveSessionConfig,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    stop_event: threading.Event,
    stats: WhisperLiveSessionStats,
) -> tuple[list[object], list[threading.Thread]]:
    capture_streams: list[object] = []
    reader_threads: list[threading.Thread] = []
    
    mic_preprocess_config = _build_mic_preprocess_config(cfg)
    speaker_preprocess_config = _build_speaker_preprocess_config(cfg)

    # Inisialisasi stream mikrofon
    if cfg.source in ("mic", "both"):
        try:
            mic_stream = MicrophoneStream(
                MicrophoneConfig(
                    sample_rate=cfg.sample_rate,
                    block_size=cfg.block_size,
                    device=cfg.mic_device,
                    queue_size=cfg.queue_size,
                )
            )
            capture_streams.append(mic_stream)

            log_process_event(
                "client.capture_backend",
                source="mic",
                backend="microphone",
                sample_rate=cfg.sample_rate,
                block_size=cfg.block_size,
                queue_size=cfg.queue_size,
                device=cfg.mic_device,
            )

            reader_threads.append(
                _reader("mic", mic_stream, mic_preprocess_config, chunk_queue, stop_event, stats, _build_client_vad_filter(cfg))
            )
        except Exception as exc:
            log_process_event("client.capture_backend_error", source="mic", backend="microphone", error=str(exc))
            _log.warning("whisperlive session: mic unavailable: %s", exc)
            # Mikrofon adalah input UTAMA. Menelan kegagalan ini membuat sesi berjalan
            # tanpa suara tanpa penjelasan apa pun ke pengguna. Naikkan ke parent agar
            # GUI dapat menampilkan panduan yang dapat ditindaklanjuti (W3).
            emit_status("mic", "CLIENT_MIC_ERROR", reason=str(exc))

    # Inisialisasi stream speaker
    if cfg.source in ("speaker", "both"):
        backend_audio = get_audio_backend()
        try:
            if backend_audio is AudioBackend.WASAPI_LOOPBACK:
                speaker_stream = WindowsLoopbackStream(
                    WindowsLoopbackConfig(
                        sample_rate=cfg.sample_rate,
                        block_size=cfg.block_size,
                        device=cfg.speaker_device,
                        queue_size=cfg.queue_size,
                    )
                )
                backend = "wasapi_loopback"
            elif backend_audio is AudioBackend.SCREENCAPTUREKIT:
                speaker_stream = MacSystemAudioStream(
                    MacSystemAudioConfig(
                        sample_rate=cfg.sample_rate or 48_000,
                        block_size=cfg.block_size,
                        queue_size=cfg.queue_size,
                    )
                )
                backend = "mac_system_audio"
            else:
                raise CaptureNotSupportedError(f"speaker capture is unsupported on {backend_audio.value}")
            capture_streams.append(speaker_stream)
            log_process_event(
                "client.capture_backend",
                source="speaker",
                backend=backend,
                backend_audio=backend_audio.value,

                sample_rate=cfg.sample_rate,
                block_size=cfg.block_size,
                queue_size=cfg.queue_size,
                device=cfg.speaker_device,
            )
            reader_threads.append(
                _reader("speaker", speaker_stream, speaker_preprocess_config, chunk_queue, stop_event, stats, _build_client_vad_filter(cfg))
            )
        except CaptureNotSupportedError as exc:
            log_process_event(
                "client.capture_backend_error",
                source="speaker",
                backend_audio=backend_audio.value,

                error=str(exc),
            )
            _log.warning("whisperlive session: speaker unavailable: %s", exc)

    return capture_streams, reader_threads

def _build_mic_preprocess_config(cfg: WhisperLiveSessionConfig) -> PreprocessConfig:
    return PreprocessConfig(
        chunk_seconds=cfg.chunk_seconds,
        min_chunk_seconds=cfg.chunk_seconds,
        target_rms_db=cfg.mic_target_rms_db,
        max_normalization_gain_db=cfg.mic_max_normalization_gain_db,
        noise_reduction_enabled=cfg.mic_noise_reduction,
    )


def _build_speaker_preprocess_config(cfg: WhisperLiveSessionConfig) -> PreprocessConfig:
    return PreprocessConfig(
        chunk_seconds=cfg.chunk_seconds,
        min_chunk_seconds=cfg.chunk_seconds,
        target_rms_db=cfg.speaker_target_rms_db,
        max_normalization_gain_db=cfg.speaker_max_normalization_gain_db,
        noise_reduction_enabled=False,
    )

# membuat thread pembaca stream yang memproses audio dari hardware ke dalam antrian chunk 
def _reader(
    source: str,
    stream: object,
    preprocess_config: PreprocessConfig,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    stop_event: threading.Event,
    stats: WhisperLiveSessionStats,
    vad_filter: "SileroVADFilter | None" = None,
) -> threading.Thread:
    return threading.Thread(
        target=_reader_thread,
        args=(
            stream,
            AudioPreprocessor(preprocess_config),
            chunk_queue,
            preprocess_config.chunk_seconds,
            stop_event,
            stats,
            vad_filter,
        ),
        name=f"whisperlive-{source}-reader",
        daemon=True,
    )


# akumulasi frame dari *stream*, preprocess, push ke *chunk_queue*.
# berjalan di thread daemon sendiri. Tidak merujuk model Whisper — hanya penangkapan audio dan preprocessing yang terjadi di sini.  Thread-safety dipertahankan dengan:
#   - Hanya menulis ke *chunk_queue* (Queue thread-safe).
#   - Hanya membaca *stop_event.is_set()* (bool-like atomik).
#   - *preprocessor* TIDAK dibagikan antar thread.

def _reader_thread(
    stream: object,
    preprocessor: AudioPreprocessor,
    chunk_queue: queue.Queue[PreprocessedAudioChunk],
    chunk_seconds: float,
    stop_event: threading.Event,
    stats: WhisperLiveSessionStats,
    vad_filter: "SileroVADFilter | None" = None,
) -> None:
    buffer: list[AudioFrame] = []
    source_name = "unknown"
    frames_seen = 0
    chunks_enqueued = 0
    first_frame_logged = False
    first_frame_timestamp: float | None = None
    dropped_by_preprocess = 0

    # loop pembaca frame audio dari stream. 
    while not stop_event.is_set():
        raw_frame = stream.read_frame(timeout=0.1)  # membaca frame audio dari stream dengan timeout 0.1 detik. jika terlalu cepat membebani cpu
        if raw_frame is None:
            continue

        if first_frame_timestamp is None:
            first_frame_timestamp = raw_frame.timestamp_seconds
        frame = _relative_frame(raw_frame, first_frame_timestamp)
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
                start_seconds=round(frame.timestamp_seconds, 3),
                capture_clock_seconds=round(raw_frame.timestamp_seconds, 3),
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
            log_process_event("client.preprocess_decision", **decision, queue_size=chunk_queue.qsize())
        
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

        # Admission control VAD lokal: buang chunk non-speech sebelum enqueue.
        chunks = _apply_client_vad(vad_filter, chunks, stats, source_name, chunk_queue.qsize())

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
                    stats.add_chunks_dropped()

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
            final_chunks = _apply_client_vad(
                vad_filter, preprocessor.preprocess_frames(buffer), stats, source_name, chunk_queue.qsize()
            )
            for chunk in final_chunks:
                for decision in preprocessor.last_decisions:
                    log_process_event(
                        "client.preprocess_decision",
                        final_partial=True,
                        **decision,
                        queue_size=chunk_queue.qsize(),
                    )
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
                    stats.add_chunks_dropped()
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


def _relative_frame(frame: AudioFrame, first_frame_timestamp: float) -> AudioFrame:
    relative_timestamp = max(0.0, frame.timestamp_seconds - first_frame_timestamp)
    if relative_timestamp == frame.timestamp_seconds:
        return frame
    return replace(frame, timestamp_seconds=relative_timestamp)


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
