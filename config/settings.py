"""Runtime settings for the transcriber foundation."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Small settings container for phase 1."""

    log_level: str = "INFO"


def load_settings() -> Settings:
    """Load settings from environment variables."""

    return Settings(log_level=os.getenv("LOG_LEVEL", "INFO").upper())
