"""Unit tests for app.py — CORS, lifespan graph, and endpoint wiring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCORSConfiguration:
    """Verify CORS origins come from environment, not hardcoded wildcard."""

    def test_no_hardcoded_wildcard_origins(self):
        """app.py must not contain allow_origins=['*']."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert 'allow_origins=["*"]' not in content
        assert "allow_origins=['*']" not in content

    def test_cors_reads_from_env(self):
        """ALLOWED_ORIGINS env var should control CORS origins."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert "ALLOWED_ORIGINS" in content

    def test_cors_default_is_localhost(self):
        """Default CORS origin should be http://localhost:3000."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert "http://localhost:3000" in content


class TestNoGlobalGraph:
    """Verify graph is stored on app.state, not as a module global."""

    def test_no_global_graph_variable(self):
        """app.py must not have a module-level 'graph = None' or 'global graph'."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        # Should not have module-level graph = None
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped == "graph = None":
                pytest.fail("Found module-level 'graph = None' in app.py")
            if stripped == "global graph":
                pytest.fail("Found 'global graph' in app.py")

    def test_graph_stored_on_app_state(self):
        """Lifespan should store graph on app.state."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert "app.state.graph" in content

    def test_generate_sse_accepts_graph_param(self):
        """generate_sse should accept graph as a parameter."""
        from app import generate_sse
        import inspect

        sig = inspect.signature(generate_sse)
        assert "graph" in sig.parameters


class TestPyprojectToml:
    """Verify pyproject.toml has required sections."""

    def test_pyproject_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    def test_has_project_section(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[project]" in content

    def test_has_dependencies(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "dependencies" in content
        assert "langchain" in content
        assert "fastapi" in content

    def test_has_dev_dependencies(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[project.optional-dependencies]" in content
        assert "pytest" in content

    def test_has_pytest_config(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.pytest.ini_options]" in content
        assert 'asyncio_mode = "auto"' in content


class TestEnvExample:
    """Verify .env.example has ALLOWED_ORIGINS."""

    def test_allowed_origins_in_env_example(self):
        content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "ALLOWED_ORIGINS" in content


class TestInputValidation:
    """Verify Pydantic max_length constraints on request schemas (SEC-01)."""

    def test_chat_request_rejects_oversized_query(self):
        from pydantic import ValidationError
        from src.schemas import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(query="x" * 5000)

    def test_chat_request_accepts_normal_query(self):
        from src.schemas import ChatRequest

        req = ChatRequest(query="正常长度的问题")
        assert req.query == "正常长度的问题"

    def test_resume_request_rejects_oversized_plan(self):
        from pydantic import ValidationError
        from src.schemas import ResumeRequest

        with pytest.raises(ValidationError):
            ResumeRequest(thread_id="t-1", edited_plan="x" * 20000)

    def test_resume_request_accepts_normal_plan(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="t-1", edited_plan="## 正常计划")
        assert req.edited_plan == "## 正常计划"


class TestFeedbackEndpoint:
    """Verify feedback writes are append-only and test-isolated."""

    def test_feedback_endpoint_appends_jsonl(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTH_USERNAME", "admin")
        monkeypatch.setenv("AUTH_PASSWORD", "123456")
        monkeypatch.setenv("AUTH_SECRET", "test-secret")

        import app as app_module

        feedback_file = tmp_path / "ratings.jsonl"
        monkeypatch.setattr(app_module, "FEEDBACK_FILE", feedback_file)

        client = TestClient(app_module.app)
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "123456"},
        )
        assert login_response.status_code == 200

        for message_id, rating in (("msg-1", "up"), ("msg-2", "down")):
            response = client.post(
                "/feedback",
                json={
                    "message_id": message_id,
                    "rating": rating,
                    "query_preview": "二次函数怎么学？",
                },
            )
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "rating": rating}

        lines = feedback_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        entries = [json.loads(line) for line in lines]
        assert [entry["message_id"] for entry in entries] == ["msg-1", "msg-2"]
        assert [entry["rating"] for entry in entries] == ["up", "down"]
        assert all(entry["query_preview"] == "二次函数怎么学？" for entry in entries)


class TestObservabilityEndpoints:
    """Verify liveness, readiness, request ids, and metrics."""

    def test_healthz_is_public_and_returns_ok(self):
        import app as app_module

        client = TestClient(app_module.app)
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_legacy_health_endpoints_remain_public(self):
        import app as app_module

        client = TestClient(app_module.app)

        assert client.get("/health").status_code == 200
        assert client.get("/ping").status_code == 200

    def test_readyz_reports_not_ready_without_graph(self):
        import app as app_module

        if hasattr(app_module.app.state, "graph"):
            delattr(app_module.app.state, "graph")

        client = TestClient(app_module.app)
        response = client.get("/readyz")

        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["checks"]["graph"] is False

    def test_readyz_reports_ready_when_runtime_initialized(self):
        import app as app_module

        app_module.app.state.graph = object()
        app_module.app.state.db_uri_configured = False
        app_module.app.state.checkpointer_ready = False

        client = TestClient(app_module.app)
        response = client.get("/readyz")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

        delattr(app_module.app.state, "graph")

    def test_request_id_is_echoed(self):
        import app as app_module
        from src.tracing.logging import REQUEST_ID_HEADER

        client = TestClient(app_module.app)
        response = client.get("/healthz", headers={REQUEST_ID_HEADER: "req-test-1"})

        assert response.headers[REQUEST_ID_HEADER] == "req-test-1"

    def test_metrics_endpoint_exposes_dashboard_snapshot(self):
        import app as app_module
        from src.tracing.metrics import reset_metrics

        reset_metrics()
        client = TestClient(app_module.app)
        client.get("/healthz")
        response = client.get("/metrics")

        assert response.status_code == 200
        payload = response.json()
        assert {"process", "http", "llm", "rag", "semantic_cache"} <= set(payload)
        assert payload["http"]["requests_total"] >= 1
