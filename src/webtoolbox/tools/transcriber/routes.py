from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from webtoolbox.common.file_utils import InvalidFilenameError, validate_filename
from webtoolbox.config import settings
from webtoolbox.tools.transcriber.file_store import AudioStore, TextStore
from webtoolbox.tools.transcriber.progress import JobState, ProgressManager
from webtoolbox.tools.transcriber.service import TranscriptionService

logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="templates")

router = APIRouter(prefix="/tools/transcriber", tags=["transcriber"])

AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "flac", "ogg"}
TEXT_EXTENSIONS = {"txt"}


audio_store = AudioStore(base_dir=settings.audio_dir, allowed_extensions=AUDIO_EXTENSIONS)
text_store = TextStore(base_dir=settings.transcript_dir, allowed_extensions=TEXT_EXTENSIONS)
progress_manager = ProgressManager()
transcription_service = TranscriptionService(
    general_model=settings.model_size,
    estonian_model=settings.estonian_model,
    cpu_threads=settings.whisper_cpu_threads,
    num_workers=settings.whisper_num_workers,
    beam_size=settings.whisper_beam_size,
    progress_manager=progress_manager,
)

MODEL_PROFILES = {"general", "estonian"}
TRANSCRIBE_LANGUAGES = {"auto", "et", "ru", "en"}



def _safe_transcript_name(audio_file: str) -> str:
    return f"{Path(audio_file).stem}.txt"


def _list_context() -> dict[str, object]:
    return {
        "audio_files": audio_store.list_files(),
        "text_files": text_store.list_files(),
        "selected_model_profile": settings.default_model_profile if settings.default_model_profile in MODEL_PROFILES else "general",
        "selected_language": settings.default_transcribe_language if settings.default_transcribe_language in TRANSCRIBE_LANGUAGES else "auto",
        "estonian_model_configured": bool(settings.estonian_model.strip()),
    }


def _validate_transcription_options(model_profile: str, language: str) -> tuple[str, str]:
    profile = (model_profile or "general").strip().lower()
    lang = (language or "auto").strip().lower()

    if profile not in MODEL_PROFILES:
        raise HTTPException(status_code=400, detail="Unknown model profile")
    if lang not in TRANSCRIBE_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unknown transcription language")
    if profile == "estonian" and not settings.estonian_model.strip():
        raise HTTPException(
            status_code=400,
            detail="Special Estonian model is not configured on server",
        )
    return profile, lang


async def _save_upload_streaming(upload: UploadFile) -> str:
    """Store upload in chunks to support large files without high memory usage."""

    filename = validate_filename(upload.filename or "")
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File extension is not allowed")

    target = audio_store.base_dir / filename
    size = 0
    try:
        with target.open("wb") as out_file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
                out_file.write(chunk)
    except HTTPException:
        if target.exists():
            target.unlink()
        raise

    logger.info(
        "audio_uploaded",
        extra={"file_name": filename, "bytes": size, "directory": str(audio_store.base_dir)},
    )
    return filename


@router.get("", response_class=HTMLResponse)
async def transcriber_page(request: Request) -> HTMLResponse:
    context = _list_context()
    return templates.TemplateResponse(
        request=request,
        name="transcriber/index.html",
        context=context,
    )


@router.get("/partials/lists", response_class=HTMLResponse)
async def transcriber_lists_partial(request: Request) -> HTMLResponse:
    context = _list_context()
    return templates.TemplateResponse(
        request=request,
        name="transcriber/partials/file_lists.html",
        context=context,
    )


@router.post("/upload/audio", response_class=HTMLResponse)
async def upload_audio(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    try:
        await _save_upload_streaming(file)
    except InvalidFilenameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("audio_upload_failed", extra={"file_name": file.filename, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Audio upload failed") from exc
    return await transcriber_lists_partial(request)


@router.post("/delete/{kind}", response_class=HTMLResponse)
async def delete_file(request: Request, kind: str, filename: str = Form(...)) -> HTMLResponse:
    try:
        if kind == "audio":
            audio_store.delete(filename)
        elif kind == "text":
            text_store.delete(filename)
        else:
            raise HTTPException(status_code=400, detail="Unknown file kind")
    except InvalidFilenameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("file_deleted_request", extra={"kind": kind, "file_name": filename})
    return await transcriber_lists_partial(request)


@router.get("/download/{kind}/{filename}")
async def download_file(kind: str, filename: str) -> FileResponse:
    try:
        if kind == "audio":
            path = audio_store.get_path(filename)
        elif kind == "text":
            path = text_store.get_path(filename)
        else:
            raise HTTPException(status_code=400, detail="Unknown file kind")
    except (InvalidFilenameError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=path, filename=path.name, media_type="application/octet-stream")


@router.post("/transcode")
async def start_transcription(
    audio_filename: str = Form(...),
    model_profile: str = Form(default="general"),
    language: str = Form(default="auto"),
) -> JSONResponse:
    try:
        audio_filename = validate_filename(audio_filename)
        audio_path = audio_store.get_path(audio_filename)
    except (InvalidFilenameError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile, selected_language = _validate_transcription_options(model_profile, language)

    transcript_name = _safe_transcript_name(audio_filename)
    transcript_path = text_store.base_dir / transcript_name

    job_id = str(uuid4())
    await progress_manager.init_job(job_id=job_id, message="Job queued")

    logger.info(
        "transcription_started",
        extra={
            "job_id": job_id,
            "audio_file": audio_filename,
            "transcript_file": transcript_name,
            "model_profile": profile,
            "language": selected_language,
        },
    )

    asyncio.create_task(
        transcription_service.transcribe_to_text(
            job_id=job_id,
            audio_path=audio_path,
            transcript_path=transcript_path,
            model_profile=profile,
            language=selected_language,
        )
    )

    return JSONResponse({"job_id": job_id, "transcript_file": transcript_name})


@router.post("/audio/delete", response_class=HTMLResponse)
async def delete_audio_bulk(request: Request, selected_files: list[str] = Form(default=[])) -> HTMLResponse:
    deleted: list[str] = []
    for name in selected_files:
        try:
            audio_store.delete(name)
            deleted.append(name)
        except InvalidFilenameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("audio_bulk_delete", extra={"files": deleted, "count": len(deleted)})
    return await transcriber_lists_partial(request)


@router.post("/audio/transcode")
async def transcode_audio_bulk(
    selected_files: list[str] = Form(default=[]),
    model_profile: str = Form(default="general"),
    language: str = Form(default="auto"),
) -> JSONResponse:
    if not selected_files:
        raise HTTPException(status_code=400, detail="Select at least one audio file")

    profile, selected_language = _validate_transcription_options(model_profile, language)

    jobs: list[dict[str, str]] = []
    for audio_filename in selected_files:
        try:
            safe_name = validate_filename(audio_filename)
            audio_path = audio_store.get_path(safe_name)
        except (InvalidFilenameError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        transcript_name = _safe_transcript_name(safe_name)
        transcript_path = text_store.base_dir / transcript_name

        job_id = str(uuid4())
        await progress_manager.init_job(job_id=job_id, message=f"Job queued for {safe_name}")

        logger.info(
            "transcription_started",
            extra={
                "job_id": job_id,
                "audio_file": safe_name,
                "transcript_file": transcript_name,
                "model_profile": profile,
                "language": selected_language,
            },
        )

        asyncio.create_task(
            transcription_service.transcribe_to_text(
                job_id=job_id,
                audio_path=audio_path,
                transcript_path=transcript_path,
                model_profile=profile,
                language=selected_language,
            )
        )

        jobs.append({"job_id": job_id, "audio_file": safe_name, "transcript_file": transcript_name})

    return JSONResponse({"jobs": jobs})


@router.post("/text/delete", response_class=HTMLResponse)
async def delete_text_bulk(request: Request, selected_files: list[str] = Form(default=[])) -> HTMLResponse:
    deleted: list[str] = []
    for name in selected_files:
        try:
            text_store.delete(name)
            deleted.append(name)
        except InvalidFilenameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("text_bulk_delete", extra={"files": deleted, "count": len(deleted)})
    return await transcriber_lists_partial(request)


@router.websocket("/ws/{job_id}")
async def transcription_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    queue = await progress_manager.subscribe(job_id)
    try:
        latest = await progress_manager.latest(job_id)
        if latest:
            await websocket.send_text(json.dumps(latest.__dict__))

        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event.__dict__))
            if event.state in {JobState.completed, JobState.failed}:
                break
    except WebSocketDisconnect:
        logger.info("websocket_disconnected", extra={"job_id": job_id})
    finally:
        await progress_manager.unsubscribe(job_id, queue)
        try:
            await websocket.close()
        except RuntimeError:
            pass
