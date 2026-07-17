from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from src.capture.models import AudioFrame
from src.preprocessing.models import PreprocessConfig, PreprocessedAudioChunk
from src.capture.wav_sink import write_frames_to_wav
from src.preprocessing.core import AudioPreprocessor


from src.preprocessing.models import PreprocessResult

# mode: preprocess
def preprocess_audio_dir(
    input_dir: Path = Path("audio"),
    output_dir: Path = Path("audio"),
) -> list[PreprocessResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        preprocess_wav_file(
            input_dir / "mic.wav",
            output_dir / "mic.preprocessed.wav",
            source="mic",
            preprocessor=None,
        ),
        preprocess_wav_file(
            input_dir / "speaker.wav",
            output_dir / "speaker.preprocessed.wav",
            source="speaker",
            preprocessor=None,
        ),
    ]

def preprocess_wav_file(
    input_path: Path,
    output_path: Path,
    *,
    source: str,
    preprocessor: AudioPreprocessor | None = None,
) -> PreprocessResult:
    # cek apakah file input WAV ada
    if not input_path.exists():
        return PreprocessResult(
            source=source,
            input_path=input_path,
            output_path=None,
            chunk_count=0,
            duration_seconds=0.0,
            warning="input WAV does not exist",
        )

    try:
        frame = _read_wav_as_frame(input_path, source=source)                   # membaca file WAV menjadi AudioFrame
    except Exception as exc:
        return PreprocessResult(
            source=source,
            input_path=input_path,
            output_path=None,
            chunk_count=0,
            duration_seconds=0.0,
            warning=f"failed to read WAV file: {exc}",
        )

    processor = preprocessor or AudioPreprocessor(PreprocessConfig())  
    chunks = processor.preprocess_frames([frame])                           # berisi daftar PreprocessedAudioChunk hasil format conversion

    # cek apakah ada chunk yang cukup panjang untuk diproses
    if not chunks:
        return PreprocessResult(
            source=source,
            input_path=input_path,
            output_path=None,
            chunk_count=0,
            duration_seconds=0.0,
            warning="no chunk produced by preprocessing",
        )

    output = _write_chunks(output_path, chunks) # menulis chunk hasil preprocessing ke file WAV output
    duration = sum(chunk.duration_seconds for chunk in chunks) # mencatat durasi total chunk

    return PreprocessResult(
        source=source,
        input_path=input_path,
        output_path=output,
        chunk_count=len(chunks),
        duration_seconds=duration,
    )

# membaca file WAV menjadi AudioFrame, yang berisi sampel audio, sample rate, jumlah channel, dan timestamp
def _read_wav_as_frame(path: Path, *, source: str) -> AudioFrame:
    # buka dan mengambil informasi channel, sample rate, sample width, jumlah frame, dan membaca frame audio mentah dari file WAV
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()      # jumlah channel audio (1 untuk mono, 2 untuk stereo, dll.)
        sample_rate = wav_file.getframerate()   # sample rate audio dalam Hz
        sample_width = wav_file.getsampwidth()  # ukuran sampel audio dalam byte, misalnya 2 byte untuk 16-bit PCM, 1 untuk 8-bit PCM, dll.
        frames = wav_file.getnframes()          # jumlah frame audio dalam file WAV
        raw = wav_file.readframes(frames)       # membaca frame audio mentah dari file WAV sebagai bytes

    # karena whisper live hanya mendukung 16-bit PCM WAV, jika sample width bukan 2 byte, maka akan raise ValueError
    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got sample width={sample_width}")

    # normalisasi sampel audio dari 16-bit PCM menjadi float32 dalam rentang [-1.0, 1.0], reshape menjadi array 2D dengan shape (frame_count, channels)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    samples = pcm.reshape(-1, channels)
    return AudioFrame(
        source=source, 
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        timestamp_seconds=0.0,
    )

# menulis daftar PreprocessedAudioChunk ke file WAV output, mengembalikan path file output
# kenapa di perlukan? karena PreprocessedAudioChunk berisi sampel audio, sample rate, dan timestamp, sehingga perlu diubah menjadi AudioFrame agar bisa ditulis ke file WAV
def _write_chunks(path: Path, chunks: list[PreprocessedAudioChunk]) -> Path:
    frames = [
        AudioFrame(
            source=chunk.source,  # type: ignore[arg-type]
            samples=chunk.samples.reshape(-1, 1),
            sample_rate=chunk.sample_rate,
            channels=1,
            timestamp_seconds=chunk.start_seconds,
        )
        for chunk in chunks
    ]
    return write_frames_to_wav(path, frames)
