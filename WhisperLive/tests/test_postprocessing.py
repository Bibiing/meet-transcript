import json
from unittest.mock import MagicMock

from whisper_live.backend.base import ServeClientBase
from whisper_live.postprocessing import (
    SegmentPostProcessorFactory,
    SegmentStabilizer,
    collapse_repeated_words,
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


def test_segment_filter_keeps_first_completed_short_thanks_but_drops_repeat():
    processor = SegmentStabilizer()

    first = processor({
        "start": "0.000",
        "end": "1.000",
        "text": "terima kasih",
        "completed": True,
    })
    second = processor({
        "start": "1.000",
        "end": "2.000",
        "text": "terima kasih",
        "completed": True,
    })

    assert first is not None
    assert second is None


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

    first_session({"start": "0", "end": "1", "text": "terima kasih", "completed": True})
    second_result = second_session({"start": "0", "end": "1", "text": "terima kasih", "completed": True})

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
