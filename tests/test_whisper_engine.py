from __future__ import annotations

from src.whisper import (
    DEFAULT_HOTWORDS,
    DEFAULT_INITIAL_PROMPT,
    WhisperLiveProfile,
    WhisperLiveReplayConfig,
    WhisperLiveSessionConfig,
)
from src.whisper.models import TranscriptionResult, TranscriptionSegment
from src.whisper.models import SendOutcome, WhisperLiveSessionStats
from src.whisper.session import _connection_config_for_source


def test_whisperlive_profile_defaults_match_live_contract() -> None:
    profile = WhisperLiveProfile()

    assert profile.model == "medium"
    assert profile.language == "id"
    assert profile.task == "transcribe"
    assert profile.local_agreement is True
    assert profile.speech_boundary_detection is True
    assert profile.initial_prompt == DEFAULT_INITIAL_PROMPT
    assert profile.hotwords == DEFAULT_HOTWORDS


def test_whisperlive_session_config_defaults_to_dual_source() -> None:
    config = WhisperLiveSessionConfig()

    assert config.source == "both"
    assert config.chunk_seconds == 0.5
    assert config.audio_format == "int16"
    assert config.session_id
    assert config.process_log_include_hot_path is False
    assert config.process_log_summary_interval_seconds == 5.0
    assert config.candidate_cache_max_entries == 2_000
    assert config.merger_emitted_cache_max_entries == 5_000

    assert config.speaker_target_rms_db == -23.0
    assert config.speaker_max_normalization_gain_db == 18.0
    assert config.rolling_audio_archive_dir is None
    assert config.rolling_audio_segment_seconds == 60.0
    assert config.resume_transcript_log is False
    assert config.profile.model == "medium"


def test_whisperlive_replay_config_defaults_to_mic_source(tmp_path) -> None:
    config = WhisperLiveReplayConfig(wav_path=tmp_path / "sample.wav")

    assert config.source == "mic"
    assert config.chunk_seconds == 0.5
    assert config.audio_format == "int16"


def test_connection_config_uses_source_specific_server_vad() -> None:
    config = WhisperLiveSessionConfig(mic_server_vad=True, speaker_server_vad=False)

    mic_config = _connection_config_for_source(config, "mic")
    speaker_config = _connection_config_for_source(config, "speaker")

    assert mic_config.profile.use_vad is True
    assert speaker_config.profile.use_vad is False


def test_transcription_result_end_seconds_uses_duration() -> None:
    result = TranscriptionResult(
        source="speaker",
        text="halo meeting",
        model_name="small",
        language="id",
        start_seconds=2.0,
        duration_seconds=1.5,
        segments=[TranscriptionSegment(start=2.0, end=3.5, text="halo meeting")],
    )

    assert result.end_seconds == 3.5


def test_whisperlive_session_stats_helpers_update_counters() -> None:
    stats = WhisperLiveSessionStats()

    stats.add_send_outcome(
        SendOutcome(
            sent=2,
            buffered=1,
            dropped=3,
            reconnect_attempts=4,
            reconnect_successes=1,
        )
    )
    stats.add_chunks_dropped()
    stats.add_result_received(2)
    stats.set_transcript_summary({"stable": 2})

    assert stats.chunks_sent == 2
    assert stats.chunks_buffered == 1
    assert stats.chunks_dropped == 4
    assert stats.reconnect_attempts == 4
    assert stats.reconnect_successes == 1
    assert stats.results_received == 2
    assert stats.transcript_summary == {"stable": 2}
