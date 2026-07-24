from __future__ import annotations

import asyncio
import hmac
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from webtoolbox.config import settings
from webtoolbox.tools.transcriber.routes import transcription_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["api"])

ALLOWED_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
CHUNK_SIZE = 1024 * 1024


def _require_api_token(x_api_key: str | None) -> None:
    expected_token = settings.api_token.strip()
    if not expected_token:
        logger.error("api_token_not_configured")
        raise HTTPException(status_code=503, detail="Transcription API is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected_token):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _save_temp_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.ogg").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    total_bytes = 0
    temp_file = tempfile.NamedTemporaryFile(prefix="webtoolbox-api-", suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    try:
        with temp_file:
            while chunk := await upload.read(CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Audio file exceeds the upload limit")
                temp_file.write(chunk)
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="et"),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """Transcribe an audio upload with the dedicated Estonian CTranslate2 model."""
    _require_api_token(x_api_key)
    normalized_language = (language or "et").strip().lower()
    if normalized_language != "et":
        raise HTTPException(status_code=400, detail="Only Estonian (et) is supported")

    audio_path = await _save_temp_upload(file)
    try:
        transcript = await asyncio.to_thread(
            transcription_service.transcribe_sync,
            audio_path,
            "estonian",
            normalized_language,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("api_transcription_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Transcription failed") from exc
    finally:
        audio_path.unlink(missing_ok=True)

    logger.info("api_transcription_completed", extra={"language": normalized_language})
    return JSONResponse({"text": transcript, "language": normalized_language, "model_profile": "estonian"})
