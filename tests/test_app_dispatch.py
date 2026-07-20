from __future__ import annotations

import sys

import pytest

from src.app import is_cli_invocation, main
from src.core import engine
from src.core.engine import UiOptions, build_live_command


def test_is_cli_invocation_detects_modes() -> None:
    assert is_cli_invocation(["--mode", "live"]) is True
    assert is_cli_invocation(["--mode=preprocess"]) is True
    assert is_cli_invocation(["--replay-file", "x.wav"]) is True
    assert is_cli_invocation([]) is False
    assert is_cli_invocation(["--source", "mic"]) is False  # tanpa --mode -> GUI


def test_dispatch_routes_cli_invocation_to_cli_main(monkeypatch) -> None:
    called = {}

    def fake_cli_main(argv):
        called["argv"] = argv
        return 0

    def fail_gui():
        raise AssertionError("GUI must NOT start for a CLI invocation")

    monkeypatch.setitem(sys.modules, "src.main", type(sys)("src.main"))
    sys.modules["src.main"].main = fake_cli_main
    monkeypatch.setitem(sys.modules, "src.qt_client", type(sys)("src.qt_client"))
    sys.modules["src.qt_client"].run_gui = fail_gui

    rc = main(["--mode", "live", "--source", "mic"])
    assert rc == 0
    assert called["argv"] == ["--mode", "live", "--source", "mic"]


def test_dispatch_routes_no_mode_to_gui(monkeypatch) -> None:
    called = {}

    def fake_run_gui():
        called["gui"] = True
        return 0

    def fail_cli(argv):
        raise AssertionError("CLI must NOT run without a mode argument")

    monkeypatch.setitem(sys.modules, "src.qt_client", type(sys)("src.qt_client"))
    sys.modules["src.qt_client"].run_gui = fake_run_gui
    monkeypatch.setitem(sys.modules, "src.main", type(sys)("src.main"))
    sys.modules["src.main"].main = fail_cli

    rc = main([])
    assert rc == 0
    assert called["gui"] is True


def test_build_live_command_dev_uses_dispatcher_module(monkeypatch) -> None:
    monkeypatch.setattr(engine, "is_frozen", lambda: False)
    command = build_live_command(UiOptions())
    assert command[:3] == [sys.executable, "-m", "src.app"]
    assert "-m" in command  # dev: interpreter + modul dispatcher


def test_build_live_command_frozen_uses_exe_without_dash_m(monkeypatch) -> None:
    monkeypatch.setattr(engine, "is_frozen", lambda: True)
    command = build_live_command(UiOptions())
    # Packaged: exe men-dispatch pada args; TIDAK boleh ada `-m` (Nuitka tak dukung).
    assert command[0] == sys.executable
    assert "-m" not in command
    assert command[1:3] == ["--mode", "live"]
