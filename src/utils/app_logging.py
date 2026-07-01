"""Application logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path
import sys


def configure_logging(level: str = "INFO", log_file: Path = Path("logs") / "transcriber.log") -> logging.Logger:
    """Configure root logging for CLI and file diagnostics."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(_coerce_level(level))

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    root.addHandler(console_handler)

    logger = logging.getLogger("transcriber")
    logger.info("logging configured level=%s file=%s", level.upper(), log_file)
    return logger


def _coerce_level(level: str) -> int:
    normalized = level.upper()
    if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    return int(getattr(logging, normalized))
