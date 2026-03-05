from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DailyFileHandler(logging.FileHandler):
    """A FileHandler that writes logs to a date-stamped file and rotates daily."""

    def __init__(self, logs_dir: Path, prefix: str = "webtoolbox") -> None:
        self.logs_dir = logs_dir
        self.prefix = prefix
        self.current_date = datetime.now().date()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        filename = self._build_filename(self.current_date)
        super().__init__(filename=filename, encoding="utf-8")

    def _build_filename(self, date_value: datetime.date) -> str:
        return str(self.logs_dir / f"{self.prefix}-{date_value.isoformat()}.log")

    def emit(self, record: logging.LogRecord) -> None:
        today = datetime.now().date()
        if today != self.current_date:
            self.current_date = today
            self.acquire()
            try:
                if self.stream:
                    self.stream.close()
                    self.stream = None
                self.baseFilename = self._build_filename(today)
                self.stream = self._open()
            finally:
                self.release()
        super().emit(record)


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for consistent, structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=True)


def setup_logging(logs_dir: Path) -> None:
    """Configure root logging with console and daily rotating file outputs."""

    formatter = JsonFormatter()

    file_handler = DailyFileHandler(logs_dir=logs_dir, prefix="webtoolbox")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.error").handlers.clear()
