from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = "Webtoolbox"
    host: str = os.getenv("WEBTOOLBOX_HOST", "0.0.0.0")
    port: int = _get_env_int("WEBTOOLBOX_PORT", 8000)
    data_dir: Path = Path(os.getenv("WEBTOOLBOX_DATA_DIR", "/data/webtoolbox"))
    audio_subdir: str = os.getenv("WEBTOOLBOX_AUDIO_SUBDIR", "audio")
    transcript_subdir: str = os.getenv("WEBTOOLBOX_TRANSCRIPT_SUBDIR", "transcripts")
    logs_dir: Path = Path(os.getenv("WEBTOOLBOX_LOGS_DIR", "logs"))
    max_upload_bytes: int = _get_env_int("WEBTOOLBOX_MAX_UPLOAD_BYTES", 200 * 1024 * 1024)
    model_size: str = os.getenv("WEBTOOLBOX_WHISPER_MODEL", "small")

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / self.audio_subdir

    @property
    def transcript_dir(self) -> Path:
        return self.data_dir / self.transcript_subdir


settings = Settings()
