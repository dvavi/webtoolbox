from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class ProgressEvent:
    job_id: str
    state: JobState
    message: str
    percent: int | None = None
    transcript_file: str | None = None
    error: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProgressManager:
    """Tracks jobs and broadcasts progress events to websocket listeners."""

    def __init__(self) -> None:
        self._events: dict[str, ProgressEvent] = {}
        self._queues: dict[str, list[asyncio.Queue[ProgressEvent]]] = {}
        self._lock = asyncio.Lock()

    async def init_job(self, job_id: str, message: str) -> None:
        await self.update(job_id=job_id, state=JobState.pending, message=message, percent=0)

    async def update(
        self,
        job_id: str,
        state: JobState,
        message: str,
        percent: int | None = None,
        transcript_file: str | None = None,
        error: str | None = None,
    ) -> None:
        event = ProgressEvent(
            job_id=job_id,
            state=state,
            message=message,
            percent=percent,
            transcript_file=transcript_file,
            error=error,
        )

        logger.info(
            "progress_event",
            extra={
                "job_id": job_id,
                "progress_state": state,
                "progress_message": message,
                "percent": percent,
                "transcript_file": transcript_file,
                "error": error,
            },
        )

        async with self._lock:
            self._events[job_id] = event
            queues = list(self._queues.get(job_id, []))

        for queue in queues:
            await queue.put(event)

    async def latest(self, job_id: str) -> ProgressEvent | None:
        async with self._lock:
            return self._events.get(job_id)

    async def subscribe(self, job_id: str) -> asyncio.Queue[ProgressEvent]:
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        async with self._lock:
            self._queues.setdefault(job_id, []).append(queue)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue[ProgressEvent]) -> None:
        async with self._lock:
            queues = self._queues.get(job_id, [])
            if queue in queues:
                queues.remove(queue)
            if not queues and job_id in self._queues:
                del self._queues[job_id]
