"""Entry point for the PyQt6 desktop application.

Usage:
    uv run python -m src.desktop
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.runtime.settings import load_env_file


def main() -> int:
    load_env_file(Path(".env"))

    # Import PyQt after env is loaded so any Qt env vars take effect.
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont

    from src.desktop.app import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("PLN Meeting Transcriber")
    app.setOrganizationName("PLN")

    # Set a sensible default font.
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
