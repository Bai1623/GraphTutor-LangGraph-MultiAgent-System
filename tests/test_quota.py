from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.quota import DailyQuotaStore, QuotaExceeded, QuotaLimits, set_quota_store


def test_quota_store_counts_and_persists(tmp_path):
    path = tmp_path / "quota.json"
    store = DailyQuotaStore(
        path,
        limits=QuotaLimits(requests=2, tokens=100, uploads=1, retries=1),
    )

    snapshot = store.consume("admin", requests=1, tokens=40)

    assert snapshot["used"]["requests"] == 1
    assert snapshot["used"]["tokens"] == 40
    assert snapshot["remaining"]["requests"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))


def test_quota_store_rejects_over_limit(tmp_path):
    store = DailyQuotaStore(
        tmp_path / "quota.json",
        limits=QuotaLimits(requests=1, tokens=100, uploads=1, retries=1),
    )

    store.consume("admin", requests=1)

    with pytest.raises(QuotaExceeded) as exc:
        store.consume("admin", requests=1)

    assert exc.value.metric == "requests"


def test_stream_endpoint_returns_429_when_request_quota_exceeded(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_USERNAME", "admin")
    monkeypatch.setenv("AUTH_PASSWORD", "123456")
    monkeypatch.setenv("AUTH_SECRET", "test-secret")

    import app as app_module

    store = DailyQuotaStore(
        tmp_path / "quota.json",
        limits=QuotaLimits(requests=1, tokens=100, uploads=1, retries=1),
    )
    store.consume("admin", requests=1)
    set_quota_store(store)
    try:
        client = TestClient(app_module.app)
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "123456"},
        )
        assert login_response.status_code == 200

        response = client.post("/stream", json={"query": "导数是什么"})

        assert response.status_code == 429
        assert response.json()["metric"] == "requests"
    finally:
        set_quota_store(None)


@pytest.mark.anyio
async def test_generate_sse_stops_when_token_quota_exceeded(tmp_path):
    import app as app_module

    store = DailyQuotaStore(
        tmp_path / "quota.json",
        limits=QuotaLimits(requests=10, tokens=100, uploads=10, retries=10),
    )
    set_quota_store(store)
    try:
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=_AsyncIteratorMock([
            _chat_model_end("generate_answer", 80, 50, 130),
        ]))
        mock_graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=(), tasks=[]))

        collected = []
        async for sse in app_module.generate_sse("q", mock_graph, quota_user_id="admin"):
            collected.append(sse)

        payloads = [json.loads(item.removeprefix("data: ").strip()) for item in collected]
        assert any(payload.get("error") == "quota_exceeded" for payload in payloads)
        assert store.snapshot("admin")["used"]["tokens"] == 0
    finally:
        set_quota_store(None)


@pytest.mark.anyio
async def test_generate_sse_counts_retry_quota(tmp_path):
    import app as app_module

    store = DailyQuotaStore(
        tmp_path / "quota.json",
        limits=QuotaLimits(requests=10, tokens=1_000, uploads=10, retries=1),
    )
    set_quota_store(store)
    try:
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=_AsyncIteratorMock([
            _node_end("evaluate_hallucination", {"retry_count": 1}),
            _node_end("evaluate_hallucination", {"retry_count": 2}),
        ]))
        mock_graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=(), tasks=[]))

        collected = []
        async for sse in app_module.generate_sse("q", mock_graph, quota_user_id="admin"):
            collected.append(sse)

        payloads = [json.loads(item.removeprefix("data: ").strip()) for item in collected]
        assert any(payload.get("metric") == "retries" for payload in payloads)
        assert store.snapshot("admin")["used"]["retries"] == 1
    finally:
        set_quota_store(None)


class _AsyncIteratorMock:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _chat_model_end(node_name: str, input_tokens: int, output_tokens: int, total_tokens: int):
    output = SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    )
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output},
    }


def _node_end(node_name: str, output: dict):
    return {
        "event": "on_chain_end",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output},
    }
