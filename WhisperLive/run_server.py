import argparse
import os

from whisper_live.defaults import DEFAULT_HOTWORDS, DEFAULT_INITIAL_PROMPT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=9090, help="Websocket port to run the server on.")
    parser.add_argument('--backend', '-b', type=str, default='faster_whisper', help='Backends from ["tensorrt", "faster_whisper", "openvino"]')
    parser.add_argument('--faster_whisper_custom_model_path', '-fw', type=str, default=None, help="Custom Faster Whisper Model")
    parser.add_argument('--trt_model_path', '-trt', type=str, default=None, help='Whisper TensorRT model path')
    parser.add_argument('--trt_multilingual', '-m', action="store_true", help='Boolean only for TensorRT model. True if multilingual.')
    parser.add_argument('--trt_py_session', action="store_true", help='Boolean only for TensorRT model. Use python session or cpp session, By default uses Cpp.')
    parser.add_argument('--omp_num_threads', '-omp', type=int, default=1, help="Number of threads to use for OpenMP")
    parser.add_argument('--no_single_model', '-nsm', action='store_true', help='Set this if every connection should instantiate its own model. Only relevant for custom model, passed using -trt or -fw.')
    parser.add_argument('--max_clients', type=int, default=4, help='Maximum clients supported by the server.')
    parser.add_argument('--max_connection_time', type=int, default=300, help='The maximum duration (in seconds) a client can stay connected. Defaults to 300 seconds (5 minutes)')
    parser.add_argument('--cache_path', '-c', type=str, default="~/.cache/whisper-live/", help='Path to cache the converted ctranslate2 models.')
    parser.add_argument("--rest_port", type=int, default=8000, help="Port for the REST API server.")
    parser.add_argument("--enable_rest", action="store_true", help="Enable the OpenAI-compatible REST API endpoint.",)
    parser.add_argument('--cors-origins', type=str, default=None, help="Comma-separated list of allowed CORS origins (e.g., 'http://localhost:3000,http://example.com'). Defaults to localhost/127.0.0.1 on the WebSocket port.")
    parser.add_argument('--batch_inference', action='store_true', help='Enable batched GPU inference for concurrent sessions. Batches multiple sessions into a single GPU call for higher throughput.')
    parser.add_argument('--batch_max_size', type=int, default=8, help='Maximum batch size for batched inference (default: 8).')
    parser.add_argument('--batch_window_ms', type=int, default=50, help='Maximum time in ms to wait for batch to fill (default: 50).')
    parser.add_argument('--raw_pcm_input', action='store_true', help='Expect raw PCM int16 audio from clients instead of float32. Audio will be normalized to float32 range [-1.0, 1.0].')
    parser.add_argument('--metrics_port', type=int, default=0, help='Port for Prometheus /metrics endpoint. 0 = disabled (default). Requires prometheus_client.')
    parser.add_argument('--api_key', type=str, default=None, help='Optional API key for authenticating REST API and WebSocket connections. Clients must send "Authorization: Bearer <key>" header or "?token=<key>" query parameter.')
    parser.add_argument('--rate_limit_rpm', type=int, default=0, help='Maximum REST API requests per minute per client IP. 0 = unlimited (default).')
    parser.add_argument('--disable_segment_filter', action='store_true', help='Disable server-side anti-hallucination segment post-processing.')
    parser.add_argument(
        '--default_model',
        type=str,
        default=os.getenv("WHISPERLIVE_DEFAULT_MODEL", "small"),
        help='Default Faster-Whisper model preloaded and used when clients do not override one.',
    )
    parser.add_argument(
        '--default_initial_prompt',
        type=str,
        default=os.getenv("WHISPERLIVE_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT),
        help='Default Faster-Whisper initial prompt used when clients do not provide one.',
    )
    parser.add_argument(
        '--default_hotwords',
        type=str,
        default=os.getenv("WHISPERLIVE_HOTWORDS", DEFAULT_HOTWORDS),
        help='Default Faster-Whisper hotwords used when clients do not provide one.',
    )
    parser.add_argument(
        '--force_server_prompt',
        action='store_true',
        default=os.getenv("WHISPERLIVE_FORCE_SERVER_PROMPT", "true").strip().lower() in {"1", "true", "yes", "on"},
        help='Force server default prompt/hotwords and ignore client prompt overrides.',
    )
    args = parser.parse_args()

    if args.backend == "tensorrt":
        if args.trt_model_path is None:
            raise ValueError("Please Provide a valid tensorrt model path")

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    from whisper_live.postprocessing import SegmentPostProcessorFactory
    from whisper_live.server import TranscriptionServer

    segment_post_processor = None if args.disable_segment_filter else SegmentPostProcessorFactory()
    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port=args.port,
        backend=args.backend,
        faster_whisper_custom_model_path=args.faster_whisper_custom_model_path,
        whisper_tensorrt_path=args.trt_model_path,
        trt_multilingual=args.trt_multilingual,
        trt_py_session=args.trt_py_session,
        single_model=not args.no_single_model,
        max_clients=args.max_clients,
        max_connection_time=args.max_connection_time,
        cache_path=args.cache_path,
        rest_port=args.rest_port,
        enable_rest=args.enable_rest,
        cors_origins=args.cors_origins,
        batch_enabled=args.batch_inference,
        batch_max_size=args.batch_max_size,
        batch_window_ms=args.batch_window_ms,
        raw_pcm_input=args.raw_pcm_input,
        metrics_port=args.metrics_port,
        api_key=args.api_key,
        rate_limit_rpm=args.rate_limit_rpm,
        segment_post_processor=segment_post_processor,
        default_initial_prompt=args.default_initial_prompt,
        default_hotwords=args.default_hotwords,
        default_model=args.default_model,
        force_server_prompt=args.force_server_prompt,
    )
