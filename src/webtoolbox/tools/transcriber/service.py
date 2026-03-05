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

    model_size: str
    progress_manager: ProgressManager

    def __post_init__(self) -> None:
        self._engine_name: str | None = None
        self._model = None

    async def transcribe_to_text(self, job_id: str, audio_path: Path, transcript_path: Path) -> None:
        try:
            event_loop = asyncio.get_running_loop()
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.running,
                message="Loading transcription model",
                percent=5,
            )
            transcript_text = await asyncio.to_thread(self._run_transcription_sync, job_id, audio_path, event_loop)
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

    def _run_transcription_sync(self, job_id: str, audio_path: Path, event_loop: asyncio.AbstractEventLoop) -> str:
        model, engine = self._get_or_create_model_sync()
        self._engine_name = engine

        if engine == "faster-whisper":
            return self._run_faster_whisper(model, job_id, audio_path, event_loop)
        return self._run_openai_whisper(model, job_id, audio_path, event_loop)

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

    def _get_or_create_model_sync(self):
        if self._model is not None:
            return self._model, self._engine_name

        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self._model = model
            self._engine_name = "faster-whisper"
            logger.info("transcription_engine_selected", extra={"engine": self._engine_name})
            return self._model, self._engine_name
        except Exception as faster_exc:
            logger.warning(
                "faster_whisper_unavailable",
                extra={"reason": str(faster_exc), "fallback": "openai-whisper"},
            )

        import whisper

        model = whisper.load_model(self.model_size)
        self._model = model
        self._engine_name = "openai-whisper"
        logger.info("transcription_engine_selected", extra={"engine": self._engine_name})
        return self._model, self._engine_name

    def _run_faster_whisper(
        self,
        model,
        job_id: str,
        audio_path: Path,
        event_loop: asyncio.AbstractEventLoop,
    ) -> str:
        segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
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
    ) -> str:
        self._publish_progress_sync(
            event_loop=event_loop,
            job_id=job_id,
            message="Running openai-whisper transcription",
            percent=50,
        )
        result = model.transcribe(str(audio_path))
        return str(result.get("text", "")).strip()
