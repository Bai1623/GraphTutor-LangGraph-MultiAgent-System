from __future__ import annotations

import pytest

from src.task_queue import TaskQueue


@pytest.mark.asyncio
async def test_task_queue_runs_submitted_coroutine():
    queue = TaskQueue(workers=1)
    await queue.start()
    try:
        record = await queue.submit("example", lambda: _result())
        await queue.join()

        finished = await queue.get(record.task_id)
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.result == {"ok": True}
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_task_queue_records_failures():
    queue = TaskQueue(workers=1)
    await queue.start()
    try:
        record = await queue.submit("example", lambda: _failure())
        await queue.join()

        finished = await queue.get(record.task_id)
        assert finished is not None
        assert finished.status == "failed"
        assert "boom" in str(finished.error)
    finally:
        await queue.stop()


async def _result() -> dict:
    return {"ok": True}


async def _failure() -> dict:
    raise RuntimeError("boom")
