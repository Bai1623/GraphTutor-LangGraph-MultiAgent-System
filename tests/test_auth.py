"""Tests for signed-cookie login without a database."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.auth import create_session_token, credentials_match, verify_session_token


def test_default_credentials(monkeypatch):
    monkeypatch.delenv("AUTH_USERNAME", raising=False)
    monkeypatch.delenv("AUTH_PASSWORD", raising=False)

    assert credentials_match("admin", "123456")
    assert not credentials_match("admin", "wrong")


def test_session_token_round_trip(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", "test-secret")
    monkeypatch.setenv("AUTH_USERNAME", "admin")

    token = create_session_token("admin", now=1_000)

    assert verify_session_token(token, now=1_001) == "admin"


def test_expired_or_tampered_session_is_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", "test-secret")
    token = create_session_token("admin", now=1_000)

    assert verify_session_token(token, now=1_000 + 13 * 60 * 60) is None
    assert verify_session_token(token + "x", now=1_001) is None


def test_login_cookie_protects_business_endpoints(monkeypatch):
    monkeypatch.setenv("AUTH_USERNAME", "admin")
    monkeypatch.setenv("AUTH_PASSWORD", "123456")
    monkeypatch.setenv("AUTH_SECRET", "test-secret")
    from app import app

    client = TestClient(app)

    assert client.get("/auth/me").status_code == 401
    assert client.post(
        "/auth/login",
        json={"username": "admin", "password": "wrong"},
    ).status_code == 401
    assert client.post(
        "/auth/login",
        json={"username": "admin", "password": "123456"},
    ).status_code == 200
    assert client.get("/auth/me").json()["username"] == "admin"
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").status_code == 401
