from __future__ import annotations
import argparse
import platform
from pathlib import Path

from src.capture.recorder import CaptureResult, CaptureOptions, run_capture
from src.preprocessing.file_processing import PreprocessResult, preprocess_audio_dir

# live session
from src.whisper import (
    WhisperLiveProfile,
    DEFAULT_HOTWORDS,
    DEFAULT_INITIAL_PROMPT,
    # replay
    WhisperLiveReplayConfig,
    replay_wav_to_whisperlive,
    # session
    WhisperLiveSessionConfig,
    run_whisperlive_session, 
)

from src.settings import env_bool, env_float, env_int, env_optional, env_str, load_env_file
from src.utils.logging import configure_logging
from src.utils.os_detector import get_audio_backend


DEFAULT_SERVER_READY_TIMEOUT = 300.0

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Mode menjalankan aplikasi
    parser = argparse.ArgumentParser(description="Multiplatform real-time transcriber")
    parser.add_argument("--mode", choices=("record", "preprocess", "live", "replay-file"), default=None, help="Mode kerja aplikasi")

    # ASR / WhisperLive konfigurasi
    parser.add_argument("--asr-backend", choices=("whisperlive",), default="whisperlive", help="ASR backend for live mode: WhisperLive server")
    parser.add_argument("--server-host", default=env_str("WHISPERLIVE_HOST", "localhost"), help="WhisperLive server host for live mode")
    parser.add_argument("--server-port", type=int, default=env_int("WHISPERLIVE_PORT", 9090), help="WhisperLive websocket port for live mode")
    parser.add_argument("--server-wss", action="store_true", help="Use wss:// for WhisperLive websocket")
    parser.add_argument("--server-api-key", default=None, help="Optional WhisperLive API key")
    parser.add_argument("--server-ready-timeout", type=float, default=env_float("WHISPERLIVE_READY_TIMEOUT", DEFAULT_SERVER_READY_TIMEOUT), help="seconds to wait for WhisperLive SERVER_READY",)
    parser.add_argument("--live-chunk-seconds", type=float, default=env_float("WHISPERLIVE_CLIENT_CHUNK_SECONDS", 0.5), help="seconds of audio per client send in live mode",)
    parser.add_argument("--audio-format", choices=("float32", "int16", "uint8"), default=env_str("WHISPERLIVE_AUDIO_FORMAT", "int16"), help="audio payload format sent to WhisperLive",)
    parser.add_argument("--final-drain-seconds", type=float, default=env_float("WHISPERLIVE_FINAL_DRAIN_SECONDS", 10.0), help="seconds to wait for final WhisperLive results after Ctrl+C",)
    parser.add_argument("--auto-reconnect", action=argparse.BooleanOptionalAction, default=env_bool("WHISPERLIVE_AUTO_RECONNECT", True), help="reconnect WebSocket automatically when network/server disconnects",)
    parser.add_argument("--reconnect-initial-backoff-seconds", type=float, default=env_float("WHISPERLIVE_RECONNECT_INITIAL_BACKOFF_SECONDS", 1.0), help="initial reconnect delay in seconds",)
    parser.add_argument("--reconnect-max-backoff-seconds", type=float, default=env_float("WHISPERLIVE_RECONNECT_MAX_BACKOFF_SECONDS", 30.0), help="maximum reconnect delay in seconds",)
    parser.add_argument("--reconnect-buffer-seconds", type=float, default=env_float("WHISPERLIVE_RECONNECT_BUFFER_SECONDS", 30.0), help="local audio chunk buffer duration per source while reconnecting",)
    
    # Audio capture konfigurasi
    parser.add_argument("--seconds", type=float, default=3.0, help="recording duration for batch mode")
    parser.add_argument("--output-dir", type=Path, default=Path("audio"), help="directory to save captured audio WAV files")
    parser.add_argument("--preprocess-output-dir", type=Path, default=None, help="directory for preprocessed WAV files",)
    parser.add_argument("--source", choices=("mic", "speaker", "both"), default="both", help="source to record")
    parser.add_argument("--sample-rate", type=int, default=None, help="capture sample rate")
    parser.add_argument("--block-size", type=int, default=1_024, help="audio callback block size")
    parser.add_argument("--queue-size", type=int, default=64, help="max queued audio blocks per stream")
    parser.add_argument("--mic-channels", type=int, default=1, help="microphone channel count")
    parser.add_argument("--speaker-channels", type=int, default=None, help="speaker channel count")
    parser.add_argument("--mic-device", default=env_optional("PLN_MIC_DEVICE"), help="microphone device index or name, for example headset mic")
    parser.add_argument("--speaker-device", default=env_optional("PLN_SPEAKER_DEVICE"), help="speaker/output loopback device index or name, for example headset speaker")
    
    # Voice Activity Detection (VAD)
    parser.add_argument("--mic-server-vad", action=argparse.BooleanOptionalAction, default=env_bool("PLN_MIC_SERVER_VAD", True), help="enable server-side VAD for microphone chunks")
    parser.add_argument("--speaker-server-vad", action=argparse.BooleanOptionalAction, default=env_bool("PLN_SPEAKER_SERVER_VAD", False), help="enable server-side VAD for speaker/system-audio chunks")
    parser.add_argument("--mic-noise-reduction", action=argparse.BooleanOptionalAction, default=env_bool("PLN_MIC_NOISE_REDUCTION", False), help="enable noise gate for microphone preprocessing")
    parser.add_argument("--mic-target-rms-db", type=float, default=env_float("PLN_MIC_TARGET_RMS_DB", -20.0), help="microphone target RMS dB after normalization")
    parser.add_argument("--mic-max-normalization-gain-db", type=float, default=env_float("PLN_MIC_MAX_GAIN_DB", 18.0), help="maximum microphone normalization gain in dB")
    parser.add_argument("--speaker-target-rms-db", type=float, default=env_float("PLN_SPEAKER_TARGET_RMS_DB", -23.0), help="speaker target RMS dB after normalization")
    parser.add_argument("--speaker-max-normalization-gain-db", type=float, default=env_float("PLN_SPEAKER_MAX_GAIN_DB", 18.0), help="maximum speaker normalization gain in dB")
    parser.add_argument("--vad-threshold", type=float, default=env_float("WHISPERLIVE_VAD_THRESHOLD", 0.55), help="WhisperLive server VAD threshold")

    # Client-side VAD (admission control, ADR-001) — kill-switch default OFF (belum diverifikasi)
    parser.add_argument("--client-vad", action=argparse.BooleanOptionalAction, default=env_bool("PLN_CLIENT_VAD_ENABLED", False), help="enable client-side VAD to drop silent chunks before sending (admission control)")
    parser.add_argument("--client-vad-threshold", type=float, default=env_float("PLN_CLIENT_VAD_THRESHOLD", 0.5), help="client-side VAD speech probability threshold")
    parser.add_argument("--client-vad-hangover-chunks", type=int, default=env_int("PLN_CLIENT_VAD_HANGOVER_CHUNKS", 2), help="chunks kept active after speech ends (client VAD hangover)")
    
    # Konfigurasi Whisper / OpenAI ASR Model
    parser.add_argument("--whisper-model", default=None, help="Whisper model name")
    parser.add_argument("--whisper-language", default=env_str("WHISPERLIVE_LANGUAGE", "id"), help="language hint, for example id or en")
    parser.add_argument("--whisper-task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--whisper-device", default=None, help="Whisper device override, for example cpu or cuda")
    parser.add_argument("--whisper-fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--whisper-overlap-seconds", type=float, default=0.5)
    parser.add_argument("--whisper-file-chunk-seconds", type=float, default=10.0)
    parser.add_argument("--whisper-min-logprob", type=float, default=-1.0)
    parser.add_argument("--whisper-max-no-speech", type=float, default=0.60)
    parser.add_argument("--whisper-max-compression", type=float, default=2.2)
    parser.add_argument("--whisperlive-no-speech-thresh", type=float, default=env_float("WHISPERLIVE_NO_SPEECH_THRESH", 0.75), help="WhisperLive segment no_speech_prob filter threshold",)
    
    # Agreement & Sliding Window
    parser.add_argument("--local-agreement", action=argparse.BooleanOptionalAction, default=env_bool("WHISPERLIVE_LOCAL_AGREEMENT", True), help="finalize live captions only after consecutive sliding-window hypotheses agree",)
    parser.add_argument("--local-agreement-window-seconds", type=float, default=env_float("WHISPERLIVE_LOCAL_AGREEMENT_WINDOW_SECONDS", 20.0), help="maximum server-side sliding window duration",)
    parser.add_argument("--local-agreement-hop-seconds", type=float, default=env_float("WHISPERLIVE_LOCAL_AGREEMENT_HOP_SECONDS", 3.0), help="minimum interval between server-side sliding-window transcriptions",)
    
    # debugging & arsip chunk
    parser.add_argument("--debug-chunk-archive", action=argparse.BooleanOptionalAction, default=env_bool("WHISPERLIVE_DEBUG_CHUNK_ARCHIVE", False), help="write each live chunk to WAV files for debugging",)
    parser.add_argument("--chunk-archive-dir", type=Path, default=Path("audio") / "chunks", help="directory used when --debug-chunk-archive is enabled",)
    parser.add_argument("--rolling-audio-archive", action=argparse.BooleanOptionalAction, default=env_bool("WHISPERLIVE_ROLLING_AUDIO_ARCHIVE", False), help="write larger rolling preprocessed WAV segments for final reprocess")
    parser.add_argument("--rolling-audio-dir", type=Path, default=Path("audio") / "rolling", help="directory used when --rolling-audio-archive is enabled")
    parser.add_argument("--rolling-audio-segment-seconds", type=float, default=env_float("WHISPERLIVE_ROLLING_AUDIO_SEGMENT_SECONDS", 60.0), help="seconds per rolling audio WAV segment")
    
    # prompt & boundary speach
    parser.add_argument("--dynamic-prompt", action=argparse.BooleanOptionalAction, default=env_bool("WHISPERLIVE_DYNAMIC_PROMPT", True), help="send confirmed transcript text back to Whisper as decode context",)
    parser.add_argument("--speech-boundary-detection", action=argparse.BooleanOptionalAction, default=env_bool("WHISPERLIVE_SPEECH_BOUNDARY_DETECTION", True), help="delay ASR until recent speech appears to have ended, with a max-wait fallback",)
    parser.add_argument("--speech-boundary-silence-seconds", type=float, default=env_float("WHISPERLIVE_SPEECH_BOUNDARY_SILENCE_SECONDS", 0.8), help="seconds without new speech chunks before ASR is triggered",)
    parser.add_argument("--speech-boundary-max-wait-seconds", type=float, default=env_float("WHISPERLIVE_SPEECH_BOUNDARY_MAX_WAIT_SECONDS", 5.0), help="maximum seconds to wait before ASR runs during continuous speech",)
    parser.add_argument("--initial-prompt", default=env_str("WHISPERLIVE_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT), help="WhisperLive initial prompt",)
    parser.add_argument("--hotwords", default=env_str("WHISPERLIVE_HOTWORDS", DEFAULT_HOTWORDS), help="WhisperLive hotwords",)
    
    # Output & logging
    parser.add_argument("--transcript-output", type=Path, default=None, help="JSON output path for transcription results",)
    parser.add_argument("--transcript-log", type=Path, default=None, help="realtime transcript backup path",)
    parser.add_argument("--resume-transcript-log", action="store_true", help="append to an existing transcript log")
    parser.add_argument("--log-level", default=env_str("LOG_LEVEL", "INFO"), help="DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", type=Path, default=Path("logs") / "transcriber.log")
    parser.add_argument("--process-log-hot-path-detail", action=argparse.BooleanOptionalAction, default=env_bool("PROCESS_LOG_HOT_PATH_DETAIL", False), help="write every capture/chunk hot-path event instead of periodic summaries")
    parser.add_argument("--process-log-summary-interval-seconds", type=float, default=env_float("PROCESS_LOG_SUMMARY_INTERVAL_SECONDS", 5.0), help="seconds between process log hot-path summary records")
    parser.add_argument("--candidate-cache-max-entries", type=int, default=env_int("WHISPERLIVE_CANDIDATE_CACHE_MAX_ENTRIES", 2000), help="maximum candidate transcript keys kept for dedupe")
    parser.add_argument("--merger-emitted-cache-max-entries", type=int, default=env_int("WHISPERLIVE_MERGER_EMITTED_CACHE_MAX_ENTRIES", 5000), help="maximum emitted transcript keys kept for dedupe")
    parser.add_argument("--hide-partials", action=argparse.BooleanOptionalAction, default=env_bool("WHISPERLIVE_HIDE_PARTIALS", True), help="hide unstable live partial transcript previews",)

    # Replay-file mode
    parser.add_argument("--replay-file", type=Path, default=None, help="WAV file path to replay to WhisperLive server")
    parser.add_argument("--replay-source", choices=("mic", "speaker"), default="speaker", help="source label for replay-file mode")
    parser.add_argument("--replay-realtime", action=argparse.BooleanOptionalAction, default=True, help="replay at realtime speed (True) or as fast as possible (False)")

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    load_env_file(Path(".env"))
    env_model = env_optional("WHISPERLIVE_MODEL")

    args = parse_args(argv)
    mode = _resolve_mode(args)
    logger = configure_logging(args.log_level, args.log_file)   # konfigurasi log
    audio_backend = get_audio_backend()                         # deteksi backend audio

    print(f"Detected OS: {platform.system().strip().lower()}")
    print(f"Audio backend: {audio_backend.value}")
    logger.info("runtime audio_backend=%s", audio_backend.value)

    if mode is None:
        print("Status run completed.")
        return 0

    match mode:
        case "record":
            # simpan mic/speaker sebagai WAV untuk debugging.
            logger.info("capture start source=%s seconds=%s output_dir=%s", args.source, args.seconds, args.output_dir)
            print(f"source={args.source} seconds={args.seconds:g} output_dir={args.output_dir}")

            capture_results = run_capture(
                CaptureOptions(
                    seconds=args.seconds,
                    output_dir=args.output_dir,
                    source=args.source,
                    sample_rate=args.sample_rate,
                    block_size=args.block_size,
                    queue_size=args.queue_size,
                    mic_channels=args.mic_channels,
                    speaker_channels=args.speaker_channels,
                    mic_device=_device_arg(args.mic_device),
                    speaker_device=_device_arg(args.speaker_device),
                )
            )

            _print_capture_results(capture_results)
            for result in capture_results:
                if result.path is None:
                    logger.warning("capture source=%s warning=%s", result.source, result.warning)
                else:
                    logger.info(
                        "capture source=%s path=%s frames=%s duration=%.2f sample_rate=%s channels=%s",
                        result.source,
                        result.path,
                        result.frame_count,
                        result.duration_seconds,
                        result.sample_rate,
                        result.channels,
                    )
            return 0 
        
        case "preprocess":
            # Preprocessing batch mengubah WAV mentah menjadi audio 16 kHz mono
            input_dir = args.output_dir
            output_dir = args.preprocess_output_dir or args.output_dir

            logger.info("preprocessing start input_dir=%s output_dir=%s", input_dir, output_dir)
            print(f"preprocessing started: input_dir={input_dir} output_dir={output_dir}")

            preprocess_results = preprocess_audio_dir(input_dir, output_dir)
            _print_preprocess_results(preprocess_results)

            for result in preprocess_results:
                if result.output_path is None:
                    logger.info("preprocessing source=%s skipped_or_warning=%s", result.source, result.warning)
                else:
                    logger.info(
                        "preprocessing source=%s path=%s chunks=%s duration=%.2f",
                        result.source,
                        result.output_path,
                        result.chunk_count,
                        result.duration_seconds,
                    )
            return 0 if any(result.output_path is not None for result in preprocess_results) else 2

        case "live":
            log_transcript = args.transcript_log or (args.output_dir / "transcript_log.json")

            live_model = args.whisper_model or env_model or "medium"

            # ASR whisper server live
            profile = WhisperLiveProfile(
                language=args.whisper_language,
                task=args.whisper_task,
                model=live_model,
                use_vad=False,
                vad_threshold=args.vad_threshold,
                no_speech_thresh=args.whisperlive_no_speech_thresh,
                local_agreement=args.local_agreement,
                local_agreement_window_seconds=args.local_agreement_window_seconds,
                local_agreement_hop_seconds=args.local_agreement_hop_seconds,
                dynamic_prompt=args.dynamic_prompt,
                speech_boundary_detection=args.speech_boundary_detection,
                speech_boundary_silence_seconds=args.speech_boundary_silence_seconds,
                speech_boundary_max_wait_seconds=args.speech_boundary_max_wait_seconds,
                initial_prompt=args.initial_prompt,
                hotwords=args.hotwords,
            )
            
            # Konfigurasi sesi live WhisperLive
            live_cfg = WhisperLiveSessionConfig(
                server_host=args.server_host,
                server_port=args.server_port,
                use_wss=args.server_wss,
                api_key=args.server_api_key,
                ready_timeout=args.server_ready_timeout,
                chunk_seconds=args.live_chunk_seconds,
                audio_format=args.audio_format,
                source=args.source,
                sample_rate=args.sample_rate,
                block_size=args.block_size,
                queue_size=args.queue_size,
                mic_device=_device_arg(args.mic_device),
                speaker_device=_device_arg(args.speaker_device),
                mic_server_vad=args.mic_server_vad,
                speaker_server_vad=args.speaker_server_vad,
                client_vad_enabled=args.client_vad,
                client_vad_threshold=args.client_vad_threshold,
                client_vad_hangover_chunks=args.client_vad_hangover_chunks,
                mic_noise_reduction=args.mic_noise_reduction,
                mic_target_rms_db=args.mic_target_rms_db,
                mic_max_normalization_gain_db=args.mic_max_normalization_gain_db,
                speaker_target_rms_db=args.speaker_target_rms_db,
                speaker_max_normalization_gain_db=args.speaker_max_normalization_gain_db,
                auto_reconnect=args.auto_reconnect,
                reconnect_initial_backoff_seconds=args.reconnect_initial_backoff_seconds,
                reconnect_max_backoff_seconds=args.reconnect_max_backoff_seconds,
                reconnect_buffer_seconds=args.reconnect_buffer_seconds,
                final_drain_seconds=args.final_drain_seconds,
                show_partials=not args.hide_partials,
                log_transcript=log_transcript,
                resume_transcript_log=args.resume_transcript_log,
                chunk_archive_dir=args.chunk_archive_dir if args.debug_chunk_archive else None,
                rolling_audio_archive_dir=args.rolling_audio_dir if args.rolling_audio_archive else None,
                rolling_audio_segment_seconds=args.rolling_audio_segment_seconds,
                process_log_include_hot_path=args.process_log_hot_path_detail,
                process_log_summary_interval_seconds=args.process_log_summary_interval_seconds,
                candidate_cache_max_entries=args.candidate_cache_max_entries,
                merger_emitted_cache_max_entries=args.merger_emitted_cache_max_entries,
                profile=profile,
            )

            logger.info(
                "whisperlive session start source=%s chunk_seconds=%s model=%s server=%s:%s",
                args.source,
                args.live_chunk_seconds,
                live_model,
                args.server_host,
                args.server_port,
            )

            # jalankan sesi live WhisperLive
            run_whisperlive_session(live_cfg)
            return 0
        
        case "replay-file":
            # Replay untuk test whisper langsung dari file.
            replay_model = args.whisper_model or env_model or "medium"
            profile = WhisperLiveProfile(
                language=args.whisper_language,
                task=args.whisper_task,
                model=replay_model,
                use_vad=args.mic_server_vad if args.replay_source == "mic" else args.speaker_server_vad,
                vad_threshold=args.vad_threshold,
                no_speech_thresh=args.whisperlive_no_speech_thresh,
                local_agreement=args.local_agreement,
                local_agreement_window_seconds=args.local_agreement_window_seconds,
                local_agreement_hop_seconds=args.local_agreement_hop_seconds,
                dynamic_prompt=args.dynamic_prompt,
                speech_boundary_detection=args.speech_boundary_detection,
                speech_boundary_silence_seconds=args.speech_boundary_silence_seconds,
                speech_boundary_max_wait_seconds=args.speech_boundary_max_wait_seconds,
                initial_prompt=args.initial_prompt,
                hotwords=args.hotwords,
            )
            result = replay_wav_to_whisperlive(
                WhisperLiveReplayConfig(
                    wav_path=args.replay_file,
                    source=args.replay_source,
                    server_host=args.server_host,
                    server_port=args.server_port,
                    use_wss=args.server_wss,
                    api_key=args.server_api_key,
                    ready_timeout=args.server_ready_timeout,
                    chunk_seconds=args.live_chunk_seconds,
                    audio_format=args.audio_format,
                    realtime=args.replay_realtime,
                    profile=profile,
                )
            )
            print(f"Replay selesai. Chunks terkirim: {result.chunks_sent} | Results: {result.results_received}")
            return 0

        case _:
            print(f"Error: mode tidak dikenal: {mode}")
            return 1
    return 0

def _resolve_mode(args: argparse.Namespace) -> str | None:
    # Jika --mode diberikan secara eksplisit, gunakan itu.
    if args.mode:
        return args.mode
    # Jika --replay-file diberikan tanpa --mode, asumsikan mode replay-file.
    if args.replay_file is not None:
        return "replay-file"
    return None


def _print_capture_results(results: list[CaptureResult]) -> None:
    for result in results:
        label = result.source.upper()
        if result.path is None:
            print(f"[{label}] warning: {result.warning}")
            continue

        # Warn clearly if the captured file is silent (all-zeros)
        if result.warning and "silent" in result.warning.lower():
            print(
                f"[{label}] saved={result.path} (PERINGATAN: audio kosong/silent)\n"
                f"  -> Untuk speaker capture: pastikan ada audio yang diputar (Zoom/Teams/YouTube)\n"
                f"  -> Untuk mic capture: periksa apakah mikrofon aktif dan tidak di-mute"
            )
            continue

        print(
            f"[{label}] saved={result.path} "
            f"frames={result.frame_count} duration={result.duration_seconds:.2f}s "
            f"sample_rate={result.sample_rate} channels={result.channels}"
        )

def _device_arg(value: str | None) -> int | str | None:
    """Normalisasi argumen device dari CLI/env menjadi int, string, atau None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _print_preprocess_results(results: list[PreprocessResult]) -> None:
    for result in results:
        label = result.source.upper()
        if result.output_path is None:
            print(f"[{label}] preprocess warning: {result.warning}")
            continue
        print(
            f"[{label}] preprocessed={result.output_path} "
            f"chunks={result.chunk_count} duration={result.duration_seconds:.2f}s "
            "sample_rate=16000 channels=1"
        )


if __name__ == "__main__":
    raise SystemExit(main())
