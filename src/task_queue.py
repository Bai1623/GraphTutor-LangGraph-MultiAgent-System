"""Lightweight in-process task queue for slow request workflows.

This keeps OCR/document parsing off the request path today, while preserving
an API boundary that can later be backed by Celery, RQ, or a managed queue.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

TaskStatus = Literal["queued", "running", "succeeded", "failed"]
TaskFunc = Callable[[], Awaitable[dict[str, Any]]]


@dataclass
class TaskRecord:
    task_id: str
    kind: str
    status: TaskStatus = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None

    def to_dict(self, *, include_traceback: bool = False) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }
        if include_traceback:
            payload["traceback"] = self.traceback
        return payload


class TaskQueue:
    def __init__(self, *, maxsize: int = 100, workers: int = 2) -> None:
        self._queue: asyncio.Queue[tuple[str, TaskFunc]] = asyncio.Queue(maxsize=maxsize)
        self._records: dict[str, TaskRecord] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._worker_count = max(1, workers)
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return bool(self._workers)

    async def start(self) -> None:
        if self._workers:
            return
        self._closed = False
        for index in range(self._worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(index)))
        logger.info("Task queue started with %d workers", self._worker_count)

    async def stop(self) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Task queue stopped")

    async def submit(self, kind: str, func: TaskFunc) -> TaskRecord:
        if self._closed:
            raise RuntimeError("Task queue is closed.")
        task_id = str(uuid.uuid4())
        record = TaskRecord(task_id=task_id, kind=kind)
        async with self._lock:
            self._records[task_id] = record
        try:
            self._queue.put_nowait((task_id, func))
        except asyncio.QueueFull as exc:
            async with self._lock:
                self._records.pop(task_id, None)
            raise RuntimeError("Task queue is full. Please retry later.") from exc
        return record

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._lock:
            return self._records.get(task_id)

    async def join(self) -> None:
        await self._queue.join()

    async def _worker_loop(self, worker_index: int) -> None:
        while True:
            task_id, func = await self._queue.get()
            record = await self.get(task_id)
            if record is None:
                self._queue.task_done()
                continue

            record.status = "running"
            record.started_at = time.time()
            record.updated_at = record.started_at
            try:
                record.result = await func()
                record.status = "succeeded"
            except Exception as exc:
                record.status = "failed"
                record.error = str(exc)
                record.traceback = traceback.format_exc()
                logger.exception(
                    "Background task failed",
                    extra={"task_id": task_id, "task_kind": record.kind, "worker": worker_index},
                )
            finally:
                record.finished_at = time.time()
                record.updated_at = record.finished_at
                self._queue.task_done()
