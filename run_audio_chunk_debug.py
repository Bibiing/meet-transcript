from __future__ import annotations

from pathlib import Path
import sys

from rttranscriber.audio_chunk_debug_session import AudioChunkDebugConfig, AudioChunkDebugRunner


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runner = AudioChunkDebugRunner(AudioChunkDebugConfig(), Path("artifacts/python_chunks"))
    diagnostics, files = runner.run()
    print(diagnostics)
    print(f"generated_chunks={len(files)}")
    for output in files:
        print(f"chunk_file={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
