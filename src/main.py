"""Command-line entrypoint for project smoke runs and phase 2 capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.capture.phase2_smoke import CaptureResult, Phase2CaptureOptions, run_phase2_capture
from src.utils.os_detector import detect_os, get_audio_backend


def main(argv: list[str] | None = None) -> int:
    """Run status output or phase 2 capture from the main entrypoint."""

    parser = argparse.ArgumentParser(description="Multiplatform real-time transcriber")
    parser.add_argument("--record", action="store_true", help="record phase 2 mic/speaker WAV files")
    parser.add_argument("--seconds", type=float, default=3.0, help="recording duration for phase 2")
    parser.add_argument("--output-dir", type=Path, default=Path("audio"), help="directory for phase 2 WAV files")
    parser.add_argument("--source", choices=("mic", "speaker", "both"), default="both", help="source to record")
    parser.add_argument("--sample-rate", type=int, default=None, help="capture sample rate")
    parser.add_argument("--block-size", type=int, default=1_024, help="audio callback block size")
    parser.add_argument("--queue-size", type=int, default=64, help="max queued audio blocks per stream")
    parser.add_argument("--mic-channels", type=int, default=1, help="microphone channel count")
    parser.add_argument("--speaker-channels", type=int, default=None, help="speaker channel count")
    args = parser.parse_args(argv)

    operating_system = detect_os()
    audio_backend = get_audio_backend(operating_system)

    print(f"Detected OS: {operating_system.value}")
    print(f"Audio backend: {audio_backend.value}")

    if not args.record:
        print("Smoke run completed.")
        return 0

    print(
        "Phase 2 capture started: "
        f"source={args.source} seconds={args.seconds:g} output_dir={args.output_dir}"
    )
    results = run_phase2_capture(
        Phase2CaptureOptions(
            seconds=args.seconds,
            output_dir=args.output_dir,
            source=args.source,
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            queue_size=args.queue_size,
            mic_channels=args.mic_channels,
            speaker_channels=args.speaker_channels,
        )
    )
    _print_capture_results(results)
    return 0 if any(result.path is not None for result in results) else 2


def _print_capture_results(results: list[CaptureResult]) -> None:
    for result in results:
        label = result.source.upper()
        if result.path is None:
            print(f"[{label}] warning: {result.warning}")
            continue

        print(
            f"[{label}] saved={result.path} "
            f"frames={result.frame_count} duration={result.duration_seconds:.2f}s "
            f"sample_rate={result.sample_rate} channels={result.channels}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
