from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


class AudioQualityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def assess_audio_quality(audio: np.ndarray, sr: int) -> AudioQualityLevel:
    """Assess the audio quality using spectral features.
    
    Returns:
        AudioQualityLevel based on zero-crossing rate, spectral rolloff, and RMS energy.
    """
    try:
        import librosa
    except ImportError:
        _log.warning("librosa not installed, defaulting to MEDIUM quality")
        return AudioQualityLevel.MEDIUM

    if audio.size == 0:
        return AudioQualityLevel.LOW

    # Compute features
    # Ensure float32 or float64 for librosa
    y = audio.astype(np.float32)
    
    # Zero crossing rate (high ZCR often indicates noise or unvoiced speech)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # Spectral rolloff (frequency below which 85% of spectral energy lies)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    
    # RMS energy
    rms = np.mean(librosa.feature.rms(y=y))

    _log.debug("Audio quality assessment - ZCR: %.4f, Rolloff: %.2f, RMS: %.4f", zcr, rolloff, rms)

    # Heuristic classification
    # High quality: Good energy, distinct frequencies (high rolloff), low zero-crossing (less broadband noise)
    if rms > 0.01 and rolloff > 3000 and zcr < 0.1:
        return AudioQualityLevel.HIGH
    # Low quality: Low energy, low frequency emphasis, or very high ZCR (hiss)
    elif rms < 0.005 or rolloff < 1500 or zcr > 0.2:
        return AudioQualityLevel.LOW
    else:
        return AudioQualityLevel.MEDIUM


def adaptive_noise_reduction(audio: np.ndarray, sr: int, quality: AudioQualityLevel) -> np.ndarray:
    """Apply noisereduce based on assessed audio quality."""
    try:
        import noisereduce as nr
    except ImportError:
        _log.warning("noisereduce not installed, skipping adaptive noise reduction")
        return audio

    if quality == AudioQualityLevel.HIGH:
        # Light stationary reduction
        _log.info("Applying light stationary noise reduction (High Quality)")
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.3,
            n_fft=512,
            hop_length=128
        )
    elif quality == AudioQualityLevel.MEDIUM:
        # Standard stationary reduction
        _log.info("Applying standard stationary noise reduction (Medium Quality)")
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.7,
            n_fft=1024,
            hop_length=256
        )
    else:  # LOW
        # Multi-stage or aggressive non-stationary reduction
        _log.info("Applying aggressive non-stationary noise reduction (Low Quality)")
        # Stage 1: Stationary for general background
        stage1 = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.8,
            n_fft=1024,
            hop_length=256
        )
        # Stage 2: Non-stationary for varying noise
        stage2 = nr.reduce_noise(
            y=stage1,
            sr=sr,
            stationary=False,
            prop_decrease=0.95,
            time_mask_smooth_ms=50,
            n_fft=1024,
            hop_length=256
        )
        return stage2


def _process_chunk(args: dict[str, Any]) -> dict[str, Any]:
    """Worker function for parallel processing."""
    chunk = args["chunk"]
    sr = args["sr"]
    quality = args["quality"]
    idx = args["idx"]
    
    enhanced = adaptive_noise_reduction(chunk, sr, quality)
    return {"idx": idx, "enhanced": enhanced}


def enhance_audio_file(input_path: Path, output_path: Path, sr: int = 16000) -> Path:
    """Enhance a full audio file using parallel adaptive noise reduction."""
    try:
        import soundfile as sf
    except ImportError:
        _log.error("soundfile not installed, cannot read/write for offline enhancement")
        return input_path

    if not input_path.exists():
        _log.warning("Input file %s does not exist", input_path)
        return input_path

    # Read audio
    audio, file_sr = sf.read(str(input_path), dtype="float32")
    
    # Force mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if file_sr != sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        except ImportError:
            _log.error("librosa required for resampling")
            return input_path

    # Assess quality on a sample (first 30 seconds or full if shorter)
    sample_len = min(len(audio), sr * 30)
    quality = assess_audio_quality(audio[:sample_len], sr)
    _log.info("Assessed audio quality for %s: %s", input_path.name, quality.value)

    # Parameters for chunking (10 seconds with 1 second overlap)
    chunk_duration = 10
    overlap_duration = 1
    
    chunk_size = chunk_duration * sr
    overlap_size = overlap_duration * sr
    step_size = chunk_size - overlap_size

    # Split into chunks
    chunks = []
    for i in range(0, len(audio), step_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < overlap_size and i > 0:
            break
        chunks.append({"idx": i, "chunk": chunk, "sr": sr, "quality": quality})

    if not chunks:
        return input_path

    _log.info("Processing %d chunks for %s", len(chunks), input_path.name)
    
    # Process chunks in parallel
    results = []
    # Use max_workers=None to use all available CPUs, but cap at 4 to avoid OOM
    with ProcessPoolExecutor(max_workers=4) as executor:
        for res in executor.map(_process_chunk, chunks):
            results.append(res)
            
    # Sort results to maintain order
    results.sort(key=lambda x: x["idx"])

    # Overlap-add synthesis
    output_audio = np.zeros(len(audio), dtype=np.float32)
    window = np.hanning(overlap_size * 2).astype(np.float32)
    fade_in = window[:overlap_size]
    fade_out = window[overlap_size:]

    for i, res in enumerate(results):
        idx = res["idx"]
        enhanced = res["enhanced"]
        chunk_len = len(enhanced)
        
        if chunk_len == 0:
            continue
            
        # Apply crossfade windowing
        if i > 0 and chunk_len >= overlap_size:
            enhanced[:overlap_size] *= fade_in
        if i < len(results) - 1 and chunk_len >= overlap_size:
            enhanced[-overlap_size:] *= fade_out[:min(overlap_size, chunk_len)]
            
        # Add to output (handling length boundary)
        end_idx = min(idx + chunk_len, len(output_audio))
        add_len = end_idx - idx
        output_audio[idx:end_idx] += enhanced[:add_len]

    # Save enhanced audio
    sf.write(str(output_path), output_audio, sr)
    _log.info("Saved enhanced audio to %s", output_path)
    
    return output_path
