from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    items = [part.strip() for part in value.split(",")]
    normalized = [item for item in items if item]
    return normalized or default


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
    estonian_model: str = os.getenv("WEBTOOLBOX_WHISPER_ESTONIAN_MODEL", "")
    default_model_profile: str = os.getenv("WEBTOOLBOX_DEFAULT_MODEL_PROFILE", "general")
    default_transcribe_language: str = os.getenv("WEBTOOLBOX_DEFAULT_TRANSCRIBE_LANGUAGE", "auto")
    whisper_cpu_threads: int = _get_env_int("WEBTOOLBOX_WHISPER_CPU_THREADS", 12)
    whisper_num_workers: int = _get_env_int("WEBTOOLBOX_WHISPER_NUM_WORKERS", 2)
    whisper_beam_size: int = _get_env_int("WEBTOOLBOX_WHISPER_BEAM_SIZE", 1)
    llm_default_provider: str = os.getenv("WEBTOOLBOX_LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("WEBTOOLBOX_OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_default_model: str = os.getenv("WEBTOOLBOX_OPENAI_MODEL", "gpt-5-mini")
    openai_models: list[str] = field(
        default_factory=lambda: _get_env_list("WEBTOOLBOX_OPENAI_MODELS", ["gpt-5-mini", "gpt-5"])
    )
    openai_timeout_seconds: int = _get_env_int("WEBTOOLBOX_OPENAI_TIMEOUT_SECONDS", 120)
    ollama_base_url: str = os.getenv("WEBTOOLBOX_OLLAMA_BASE_URL", "http://192.168.1.104:11434")
    ollama_default_model: str = os.getenv("WEBTOOLBOX_OLLAMA_MODEL", "qwen2.5:7b")
    ollama_models: list[str] = field(
        default_factory=lambda: _get_env_list("WEBTOOLBOX_OLLAMA_MODELS", ["qwen2.5:7b", "qwen2.5:14b"])
    )
    ollama_timeout_seconds: int = _get_env_int("WEBTOOLBOX_OLLAMA_TIMEOUT_SECONDS", 1200)

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / self.audio_subdir

    @property
    def transcript_dir(self) -> Path:
        return self.data_dir / self.transcript_subdir


settings = Settings()
