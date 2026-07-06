import json
from unittest.mock import MagicMock

from whisper_live.backend.base import ServeClientBase
from whisper_live.postprocessing import (
    SegmentPostProcessorFactory,
    SegmentStabilizer,
    collapse_repeated_words,
    evaluate_segment,
    is_degenerate_repetition,
)


class ConcreteServeClient(ServeClientBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language = "id"

    def transcribe_audio(self, input_sample):
        return None

    def handle_transcription_output(self, result, duration):
        pass


def test_segment_filter_drops_common_media_hallucination():
    processor = SegmentStabilizer()

    result = processor({
        "start": "0.000",
        "end": "1.000",
        "text": "Terima kasih sudah menonton",
        "completed": True,
    })

    assert result is None


def test_segment_filter_drops_prompt_echo():
    processor = SegmentStabilizer()

    result = processor({
        "start": "0.000",
        "end": "1.000",
        "text": "Pertahankan istilah teknis seperti API, database, deployment.",
        "completed": True,
    })

    assert result is None


def test_segment_filter_keeps_completed_short_thanks_at_different_times():
    processor = SegmentStabilizer()

    first = processor({
        "start": "0.000",
        "end": "2.500",
        "text": "terima kasih",
        "completed": True,
        "no_speech_prob": 0.1,
        "avg_logprob": -0.4,
        "compression_ratio": 1.2,
    })
    second = processor({
        "start": "3.000",
        "end": "5.500",
        "text": "terima kasih",
        "completed": True,
        "no_speech_prob": 0.1,
        "avg_logprob": -0.4,
        "compression_ratio": 1.2,
    })

    assert first is not None
    assert second is not None


def test_segment_filter_drops_exact_completed_duplicate():
    processor = SegmentStabilizer()
    segment = {
        "start": "0.000",
        "end": "2.500",
        "text": "terima kasih",
        "completed": True,
        "no_speech_prob": 0.1,
        "avg_logprob": -0.4,
        "compression_ratio": 1.2,
    }

    first = processor(segment)
    second = processor(segment)

    assert first is not None
    assert second is None


def test_segment_filter_drops_short_thanks_without_asr_evidence():
    processor = SegmentStabilizer()

    result = processor({
        "start": "0.000",
        "end": "1.000",
        "text": "Terima kasih.",
        "completed": True,
    })

    assert result is None


def test_segment_filter_collapses_repeated_words():
    processor = SegmentStabilizer()

    result = processor({
        "start": "0.000",
        "end": "1.000",
        "text": "billing billing billing endpoint endpoint endpoint",
        "completed": True,
    })

    assert result is not None
    assert result["text"] == "billing billing endpoint endpoint"


def test_degenerate_repetition_detects_repeated_ngram_loop():
    assert is_degenerate_repetition(
        "kita deploy endpoint kita deploy endpoint kita deploy endpoint kita deploy endpoint"
    )


def test_collapse_repeated_words_allows_two_occurrences():
    assert collapse_repeated_words("API API API database database database") == "API API database database"


def test_post_processor_factory_returns_isolated_session_state():
    factory = SegmentPostProcessorFactory()
    first_session = factory.new_session()
    second_session = factory.new_session()

    segment = {
        "start": "0",
        "end": "2.5",
        "text": "terima kasih",
        "completed": True,
        "no_speech_prob": 0.1,
        "avg_logprob": -0.4,
        "compression_ratio": 1.2,
    }
    first_session(segment)
    second_result = second_session(segment)

    assert second_result is not None


def test_send_transcription_drops_filtered_segments():
    ws = MagicMock()
    client = ConcreteServeClient(client_uid="uid", websocket=ws)
    client.segment_post_processor = SegmentStabilizer()

    client.send_transcription_to_client([
        {
            "start": "0.000",
            "end": "1.000",
            "text": "thanks for watching",
            "completed": True,
        }
    ])

    ws.send.assert_not_called()


def test_send_transcription_sends_processed_segments():
    ws = MagicMock()
    client = ConcreteServeClient(client_uid="uid", websocket=ws)
    client.segment_post_processor = SegmentStabilizer()

    client.send_transcription_to_client([
        {
            "start": "0.000",
            "end": "1.000",
            "text": "API API API",
            "completed": True,
        }
    ])

    ws.send.assert_called_once()
    payload = json.loads(ws.send.call_args[0][0])
    assert payload["segments"][0]["text"] == "API API"
    assert payload["segments"][0]["reliability_score"] >= 0.8


def test_reliability_score_penalizes_low_asr_confidence():
    evaluation = evaluate_segment(
        {
            "text": "hasil rapat masih dibahas",
            "completed": True,
            "avg_logprob": -1.8,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.2,
        },
        "hasil rapat masih dibahas",
    )

    assert evaluation.score < 0.8
    assert evaluation.action in {"review", "pending"}


def test_pending_segment_emits_after_repeated_stable_context():
    processor = SegmentStabilizer()
    segment = {
        "start": "0.000",
        "end": "2.000",
        "text": "hasil rapat masih dibahas",
        "completed": True,
        "avg_logprob": -1.8,
        "no_speech_prob": 0.1,
        "compression_ratio": 1.2,
    }

    first = processor(segment)
    second = processor({**segment, "start": "0.500", "end": "2.500"})

    assert first is not None
    assert first["reliability_action"] == "review"
    assert second is not None
    assert second["reliability_action"] == "emit"
    assert second["reliability_score"] >= 0.8


def test_very_low_reliability_remains_pending_even_when_repeated():
    processor = SegmentStabilizer()
    segment = {
        "start": "0.000",
        "end": "2.000",
        "text": "suara hening tidak jelas",
        "completed": True,
        "avg_logprob": -1.8,
        "no_speech_prob": 0.85,
        "compression_ratio": 2.9,
    }

    assert processor(segment) is None
    assert processor({**segment, "start": "0.500", "end": "2.500"}) is None
