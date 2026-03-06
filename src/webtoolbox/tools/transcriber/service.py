from __future__ import annotations

import asyncio
import json
import logging
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import socket
from urllib import error, request

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
            if await self.progress_manager.is_cancelled(job_id):
                return
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
            if await self.progress_manager.is_cancelled(job_id):
                return
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
        except asyncio.CancelledError:
            await self.progress_manager.cancel_job(job_id)
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
            if self.progress_manager.is_cancelled_sync(job_id):
                raise asyncio.CancelledError()
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
        if self.progress_manager.is_cancelled_sync(job_id):
            raise asyncio.CancelledError()
        result = model.transcribe(str(audio_path), language=language)
        if self.progress_manager.is_cancelled_sync(job_id):
            raise asyncio.CancelledError()
        return str(result.get("text", "")).strip()


@dataclass
class TranscriptLLMService:
    """Formats and summarizes transcript text files using a local Ollama model."""

    openai_api_key: str
    openai_base_url: str
    openai_timeout_seconds: int
    ollama_base_url: str
    ollama_timeout_seconds: int
    prompts_dir: Path
    progress_manager: ProgressManager

    def __post_init__(self) -> None:
        self._prompts = {
            "format": self._load_prompt("format_prompt.txt"),
            "summary": self._load_prompt("summary_prompt.txt"),
        }

    def _load_prompt(self, filename: str) -> str:
        path = self.prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    async def process_transcript(
        self,
        job_id: str,
        source_path: Path,
        output_path: Path,
        mode: str,
        provider: str,
        model: str,
    ) -> None:
        if mode not in self._prompts:
            raise ValueError(f"Unsupported mode: {mode}")

        try:
            if await self.progress_manager.is_cancelled(job_id):
                return
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.running,
                message="Reading transcript",
                percent=5,
            )
            source_text = source_path.read_text(encoding="utf-8")
            if not source_text.strip():
                raise ValueError("Transcript is empty")

            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.running,
                message="Sending text to Ollama",
                percent=30,
            )

            prompt = self._prompts[mode]
            result_text = await asyncio.to_thread(
                self._call_provider_sync,
                provider,
                model,
                prompt,
                source_text,
                job_id,
            )
            if await self.progress_manager.is_cancelled(job_id):
                return

            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.running,
                message="Saving result file",
                percent=90,
            )
            output_path.write_text(result_text, encoding="utf-8")

            completed_message = "Formatted transcript created" if mode == "format" else "Transcript summary created"
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.completed,
                message=completed_message,
                percent=100,
                transcript_file=output_path.name,
            )
            logger.info(
                "transcript_llm_completed",
                extra={
                    "job_id": job_id,
                    "source_file": source_path.name,
                    "output_file": output_path.name,
                    "mode": mode,
                    "provider": provider,
                    "model": model,
                    "openai_base_url": self.openai_base_url,
                    "ollama_base_url": self.ollama_base_url,
                },
            )
        except asyncio.CancelledError:
            await self.progress_manager.cancel_job(job_id)
        except Exception as exc:
            logger.exception(
                "transcript_llm_failed",
                extra={
                    "job_id": job_id,
                    "source_file": source_path.name,
                    "output_file": output_path.name,
                    "mode": mode,
                    "provider": provider,
                    "model": model,
                    "error": str(exc),
                },
            )
            await self.progress_manager.update(
                job_id=job_id,
                state=JobState.failed,
                message="Transcript processing failed",
                error=str(exc),
            )

    def _call_provider_sync(
        self,
        provider: str,
        model: str,
        prompt: str,
        source_text: str,
        job_id: str,
    ) -> str:
        provider_name = (provider or "").strip().lower()
        if provider_name == "openai":
            return self._call_openai_sync(model=model, prompt=prompt, source_text=source_text)
        if provider_name == "ollama":
            return self._call_ollama_sync(model=model, prompt=prompt, source_text=source_text, job_id=job_id)
        raise RuntimeError(f"Unsupported LLM provider: {provider}")

    def _call_openai_sync(self, model: str, prompt: str, source_text: str) -> str:
        api_key = self.openai_api_key.strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        url = f"{self.openai_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": source_text},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        timeout_value = self.openai_timeout_seconds if self.openai_timeout_seconds > 0 else None

        try:
            if timeout_value is None:
                response_ctx = request.urlopen(req)
            else:
                response_ctx = request.urlopen(req, timeout=timeout_value)

            with response_ctx as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP error: {exc.code} {error_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                "OpenAI request timed out. Increase WEBTOOLBOX_OPENAI_TIMEOUT_SECONDS for large files."
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid JSON response from OpenAI") from exc

        choices = parsed.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI returned no choices")

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        output = str(message.get("content", "")).strip()
        if not output:
            raise RuntimeError("OpenAI returned empty response")
        return output

    def _call_ollama_sync(self, model: str, prompt: str, source_text: str, job_id: str) -> str:
        url = f"{self.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": f"{prompt}\n\n---\n\n{source_text}",
            "stream": True,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        timeout_value = self.ollama_timeout_seconds if self.ollama_timeout_seconds > 0 else None

        try:
            if timeout_value is None:
                response_ctx = request.urlopen(req)
            else:
                response_ctx = request.urlopen(req, timeout=timeout_value)

            with response_ctx as response:
                parts: list[str] = []
                for raw_line in response:
                    if self.progress_manager.is_cancelled_sync(job_id):
                        raise asyncio.CancelledError()

                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        payload_line = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("Invalid streaming JSON response from Ollama") from exc

                    if payload_line.get("error"):
                        raise RuntimeError(f"Ollama error: {payload_line['error']}")

                    chunk = str(payload_line.get("response", ""))
                    if chunk:
                        parts.append(chunk)

                    if payload_line.get("done") is True:
                        break
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                "Ollama request timed out. Increase WEBTOOLBOX_OLLAMA_TIMEOUT_SECONDS for large files."
            ) from exc

        output = "".join(parts).strip()
        if not output:
            raise RuntimeError("Ollama returned empty response")
        return output
