"""Small signed-cookie authentication for private deployments."""

from __future__ import annotations

import hashlib
import hmac
import os
import time

from fastapi import Request


COOKIE_NAME = "gaokao_tutor_session"
DEFAULT_SESSION_HOURS = 12


def configured_username() -> str:
    return os.getenv("AUTH_USERNAME", "admin")


def credentials_match(username: str, password: str) -> bool:
    expected_user = configured_username()
    expected_password = os.getenv("AUTH_PASSWORD", "123456")
    return hmac.compare_digest(username, expected_user) and hmac.compare_digest(
        password,
        expected_password,
    )


def create_session_token(username: str, *, now: int | None = None) -> str:
    issued_at = int(time.time() if now is None else now)
    hours = _session_hours()
    expires_at = issued_at + hours * 60 * 60
    payload = f"{username}.{expires_at}"
    signature = hmac.new(
        _signing_key(),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}.{signature}"


def verify_session_token(token: str | None, *, now: int | None = None) -> str | None:
    if not token:
        return None
    try:
        username, expires_raw, signature = token.rsplit(".", 2)
        expires_at = int(expires_raw)
    except (TypeError, ValueError):
        return None

    payload = f"{username}.{expires_at}"
    expected = hmac.new(
        _signing_key(),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    current_time = int(time.time() if now is None else now)
    if expires_at <= current_time or not hmac.compare_digest(signature, expected):
        return None
    if not hmac.compare_digest(username, configured_username()):
        return None
    return username


def authenticated_username(request: Request) -> str | None:
    return verify_session_token(request.cookies.get(COOKIE_NAME))


def session_max_age_seconds() -> int:
    return _session_hours() * 60 * 60


def _signing_key() -> bytes:
    configured = os.getenv("AUTH_SECRET")
    if configured:
        return configured.encode("utf-8")
    password = os.getenv("AUTH_PASSWORD", "123456")
    return hashlib.sha256(f"gaokao-tutor:{password}".encode("utf-8")).digest()


def _session_hours() -> int:
    try:
        value = int(os.getenv("AUTH_SESSION_HOURS", str(DEFAULT_SESSION_HOURS)))
    except ValueError:
        value = DEFAULT_SESSION_HOURS
    return max(1, min(value, 24 * 30))
