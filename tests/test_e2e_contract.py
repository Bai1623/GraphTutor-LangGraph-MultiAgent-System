"""HTTP contract smoke tests for the public stream, resume, and upload paths."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def _event(event_type: str, **payload: object) -> str:
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"


def _wait_for_task(client: TestClient, status_url: str) -> dict:
    for _ in range(30):
        response = client.get(status_url)
        if response.json().get("status") in {"succeeded", "failed"}:
            return response.json()
        time.sleep(0.05)
    return response.json()


def _login(client: TestClient) -> None:
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "123456"},
    )
    assert response.status_code == 200


def test_stream_and_resume_http_sse_contract(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "admin")
    monkeypatch.setenv("AUTH_PASSWORD", "123456")
    monkeypatch.setenv("AUTH_SECRET", "test-secret")
    import app as app_module

    async def fake_stream(*args, **kwargs):
        yield _event("thread_id", thread_id="thread-e2e")
        yield _event("text", content="答案")
        yield _event("done")

    async def fake_resume(*args, **kwargs):
        yield _event("thread_id", thread_id=kwargs.get("thread_id", "thread-e2e"))
        yield _event("done")

    with patch.object(app_module, "generate_sse", new=fake_stream), patch.object(
        app_module, "generate_resume_sse", new=fake_resume
    ):
        with TestClient(app_module.app) as client:
            _login(client)
            stream = client.post("/stream", json={"query": "二次函数怎么复习？"})
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            assert '"type": "thread_id"' in stream.text
            assert '"type": "done"' in stream.text

            resumed = client.post(
                "/resume",
                json={"thread_id": "thread-e2e", "feedback": "减少每天题量"},
            )
            assert resumed.status_code == 200
            assert resumed.headers["content-type"].startswith("text/event-stream")
            assert '"type": "done"' in resumed.text


def test_document_parse_upload_contract(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "admin")
    monkeypatch.setenv("AUTH_PASSWORD", "123456")
    monkeypatch.setenv("AUTH_SECRET", "test-secret")
    import app as app_module

    result = SimpleNamespace(
        questions=[],
        recognized_text="第1题 内容",
        query="请讲解第1题",
        filenames=["exam.pdf"],
        parser="contract-test",
        segmenter_used=False,
        artifact_id="artifact-e2e",
        preview="第1题 内容",
        artifacts=[],
    )
    with patch.object(
        app_module, "parse_exam_uploads_prepared", new=AsyncMock(return_value=result)
    ):
        with TestClient(app_module.app) as client:
            _login(client)
            response = client.post(
                "/documents/parse",
                data={"question": "讲解第1题"},
                files={"files": ("exam.pdf", b"%PDF-1.4\n", "application/pdf")},
            )
            assert response.status_code == 202
            accepted = response.json()
            assert accepted["kind"] == "document_parse"
            status = _wait_for_task(client, accepted["status_url"])
            assert status["status"] == "succeeded"
            assert status["result"]["artifact_id"] == "artifact-e2e"
