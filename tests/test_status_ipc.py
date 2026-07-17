from __future__ import annotations

from src.utils.status_ipc import (
    LEVEL_SENTINEL,
    STATUS_SENTINEL,
    format_level_line,
    format_status_line,
    parse_level_line,
    parse_status_line,
)


def test_level_round_trip() -> None:
    line = format_level_line("mic", -23.4)
    assert line.startswith(LEVEL_SENTINEL)
    assert line.endswith("\n")
    assert parse_level_line(line) == ("mic", -23.4)


def test_parse_level_ignores_other_lines() -> None:
    assert parse_level_line(format_status_line("mic", "SERVER_READY")) is None
    assert parse_level_line("[Me] level -20 dB error\n") is None
    assert parse_level_line(f"{LEVEL_SENTINEL}{{bad-json}}\n") is None
    assert parse_level_line(f'{LEVEL_SENTINEL}{{"source": "mic"}}\n') is None
    # bool bukan rms_db yang valid
    assert parse_level_line(f'{LEVEL_SENTINEL}{{"source": "mic", "rms_db": true}}\n') is None


def test_command_round_trip_and_defensive_parse() -> None:
    from src.utils.status_ipc import COMMAND_SENTINEL, format_command_line, parse_command_line

    line = format_command_line("set_mute", source="mic", muted=True)
    assert line.startswith(COMMAND_SENTINEL)
    cmd, payload = parse_command_line(line)
    assert cmd == "set_mute"
    assert payload["source"] == "mic"
    assert payload["muted"] is True

    # Input stdin tak terduga tidak boleh pernah menjadi perintah.
    assert parse_command_line("halo dunia\n") is None
    assert parse_command_line(format_status_line("mic", "SERVER_READY")) is None
    assert parse_command_line(f"{COMMAND_SENTINEL}{{rusak}}\n") is None
    assert parse_command_line(f"{COMMAND_SENTINEL}[1,2]\n") is None
    assert parse_command_line(f'{COMMAND_SENTINEL}{{"source":"mic"}}\n') is None  # tanpa cmd


def test_status_and_level_channels_do_not_cross() -> None:
    # Baris level bukan status, dan sebaliknya.
    assert parse_status_line(format_level_line("speaker", -30.0)) is None
    assert parse_level_line(format_status_line("speaker", "SERVER_READY")) is None


def test_format_and_parse_round_trip() -> None:
    line = format_status_line("mic", "SERVER_READY")
    assert line.startswith(STATUS_SENTINEL)
    assert line.endswith("\n")
    assert parse_status_line(line) == ("mic", "SERVER_READY")


def test_parse_ignores_prose_and_transcript() -> None:
    # Baris prosa status dan teks transcript tidak boleh diinterpretasi sebagai sinyal.
    assert parse_status_line("[MIC] websocket connected\n") is None
    assert parse_status_line("[00:00:01 - 00:00:02] [Me] tidak ada error sama sekali\n") is None
    assert parse_status_line("[MIC] disconnected: DISCONNECT\n") is None


def test_parse_rejects_corrupted_payload() -> None:
    # Sentinel dengan JSON rusak (mis. akibat interleave antar-thread) -> None, bukan crash.
    assert parse_status_line(f"{STATUS_SENTINEL}{{not-json}}\n") is None
    assert parse_status_line(f"{STATUS_SENTINEL}[1, 2, 3]\n") is None
    assert parse_status_line(f'{STATUS_SENTINEL}{{"source": 1, "status": "X"}}\n') is None
    assert parse_status_line(f'{STATUS_SENTINEL}{{"source": "mic"}}\n') is None


def test_parse_handles_non_ascii_status_source() -> None:
    line = format_status_line("speaker", "SERVER_READY")
    assert parse_status_line(line) == ("speaker", "SERVER_READY")
