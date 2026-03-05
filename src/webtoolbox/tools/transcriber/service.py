from __future__ import annotations

import asyncio
import logging
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

from webtoolbox.tools.transcriber.progress import JobState, ProgressManager

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionService:
    """Handles transcription jobs using faster-whisper with a whisper fallback."""

    general_model: str
    estonian_model: str
    cpu_threads: int
    num_workers: int
    beam_size: int
    progress_manager: ProgressManager

    def __post_init__(self) -> None:
        self._engine_name: str | None = None
        self._models: dict[str, tuple[object, str]] = {}

    async def transcribe_to_text(
        self,
        job_id: str,
        audio_path: Path,
        transcript_path: Path,
        model_profile: str,
        language: str,
    ) -> None:
        try:
            event_loop = asyncio.get_running_loop()
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.running,
                message=f"Loading model ({model_profile})",
                percent=5,
            )
            transcript_text = await asyncio.to_thread(
                self._run_transcription_sync,
                job_id,
                audio_path,
                event_loop,
                model_profile,
                language,
            )
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.running,
                message="Finalizing transcript file",
                percent=95,
            )
            transcript_path.write_text(transcript_text, encoding="utf-8")
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.completed,
                message="Transcription completed",
                percent=100,
                transcript_file=transcript_path.name,
            )
            logger.info(
                "transcription_completed",
                extra={
                    "job_id": job_id,
                    "audio_file": audio_path.name,
                    "transcript_file": transcript_path.name,
                    "engine": self._engine_name,
                    "model_profile": model_profile,
                    "language": language,
                },
            )
        except Exception as exc:
            logger.exception(
                "transcription_failed",
                extra={"job_id": job_id, "audio_file": audio_path.name, "error": str(exc)},
            )
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.failed,
                message="Transcription failed",
                error=str(exc),
            )

    def _run_transcription_sync(
        self,
        job_id: str,
        audio_path: Path,
        event_loop: asyncio.AbstractEventLoop,
        model_profile: str,
        language: str,
    ) -> str:
        model_name = self._resolve_model_name(model_profile)
        model, engine = self._get_or_create_model_sync(model_name, model_profile)
        self._engine_name = engine
        normalized_language = self._normalize_language(language)

        if engine == "faster-whisper":
            return self._run_faster_whisper(model, job_id, audio_path, event_loop, normalized_language)
        return self._run_openai_whisper(model, job_id, audio_path, event_loop, normalized_language)

    def _resolve_model_name(self, model_profile: str) -> str:
        if model_profile == "estonian":
            model_name = self.estonian_model.strip()
            if not model_name:
                raise ValueError(
                    "Special Estonian model is not configured. Set WEBTOOLBOX_WHISPER_ESTONIAN_MODEL first."
                )
            return model_name
        return self.general_model

    @staticmethod
    def _normalize_language(language: str) -> str | None:
        value = (language or "auto").strip().lower()
        if value in {"", "auto"}:
            return None
        return value

    def _publish_progress_sync(
        self,
        event_loop: asyncio.AbstractEventLoop,
        job_id: str,
        message: str,
        percent: int,
    ) -> None:
        future: concurrent.futures.Future = asyncio.run_coroutine_threadsafe(
            self.progress_manager.update(
                job_id=job_id,
                state=JobState.running,
                message=message,
                percent=percent,
            ),
            event_loop,
        )
        future.result()

    def _get_or_create_model_sync(self, model_name: str, model_profile: str):
        if model_name in self._models:
            return self._models[model_name]

        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(0, self.cpu_threads),
                num_workers=max(1, self.num_workers),
            )
            self._models[model_name] = (model, "faster-whisper")
            logger.info(
                "transcription_engine_selected",
                extra={
                    "engine": "faster-whisper",
                    "model_name": model_name,
                    "model_profile": model_profile,
                    "cpu_threads": max(0, self.cpu_threads),
                    "num_workers": max(1, self.num_workers),
                    "beam_size": max(1, self.beam_size),
                },
            )
            return self._models[model_name]
        except Exception as faster_exc:
            logger.warning(
                "faster_whisper_unavailable",
                extra={
                    "reason": str(faster_exc),
                    "fallback": "openai-whisper",
                    "model_name": model_name,
                    "model_profile": model_profile,
                },
            )

        if model_profile != "general":
            raise RuntimeError(
                "Special model must be a faster-whisper compatible model (CTranslate2)."
            )

        if self.cpu_threads > 0:
            try:
                import torch

                torch.set_num_threads(self.cpu_threads)
                logger.info("openai_whisper_threads_configured", extra={"cpu_threads": self.cpu_threads})
            except Exception:
                logger.warning("openai_whisper_threads_config_failed", extra={"cpu_threads": self.cpu_threads})

        import whisper

        model = whisper.load_model(model_name)
        self._models[model_name] = (model, "openai-whisper")
        logger.info(
            "transcription_engine_selected",
            extra={"engine": "openai-whisper", "model_name": model_name, "model_profile": model_profile},
        )
        return self._models[model_name]

    def _run_faster_whisper(
        self,
        model,
        job_id: str,
        audio_path: Path,
        event_loop: asyncio.AbstractEventLoop,
        language: str | None,
    ) -> str:
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=max(1, self.beam_size),
            vad_filter=True,
            language=language,
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        lines: list[str] = []

        for index, segment in enumerate(segments, start=1):
            lines.append(segment.text.strip())
            if duration > 0:
                percent = min(94, max(10, int((segment.end / duration) * 90)))
            else:
                percent = min(94, 10 + index)
            self._publish_progress_sync(
                event_loop=event_loop,
                job_id=job_id,
                message=f"Processing segment {index}",
                percent=percent,
            )

        return "\n".join(line for line in lines if line)

    def _run_openai_whisper(
        self,
        model,
        job_id: str,
        audio_path: Path,
        event_loop: asyncio.AbstractEventLoop,
        language: str | None,
    ) -> str:
        self._publish_progress_sync(
            event_loop=event_loop,
            job_id=job_id,
            message="Running openai-whisper transcription",
            percent=50,
        )
        result = model.transcribe(str(audio_path), language=language)
        return str(result.get("text", "")).strip()
