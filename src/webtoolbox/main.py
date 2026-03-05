from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webtoolbox.config import settings
from webtoolbox.logging_setup import setup_logging
from webtoolbox.tools.transcriber.routes import router as transcriber_router

setup_logging(settings.logs_dir)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.include_router(transcriber_router)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "client": request.client.host if request.client else "unknown",
                    "error": str(exc),
                },
            )
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client": request.client.host if request.client else "unknown",
            },
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={
                "tools": [
                    {
                        "name": "Audio -> Text Transcoder",
                        "description": "Upload audio, transcribe in the background, and manage transcript files.",
                        "url": "/tools/transcriber",
                    }
                ]
            },
        )

    return app


app = create_app()
