from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


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
        self._cancelled_jobs: set[str] = set()
        self._cancel_lock = threading.Lock()

    async def init_job(self, job_id: str, message: str) -> None:
        with self._cancel_lock:
            self._cancelled_jobs.discard(job_id)
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

    async def cancel_job(self, job_id: str) -> bool:
        async with self._lock:
            latest = self._events.get(job_id)
        if latest is None:
            return False
        if latest.state in {JobState.completed, JobState.failed, JobState.cancelled}:
            return True

        with self._cancel_lock:
            self._cancelled_jobs.add(job_id)

        await self.update(
            job_id=job_id,
            state=JobState.cancelled,
            message="Job cancelled by user",
            percent=latest.percent,
        )
        return True

    def is_cancelled_sync(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled_jobs

    async def is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled_jobs

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
