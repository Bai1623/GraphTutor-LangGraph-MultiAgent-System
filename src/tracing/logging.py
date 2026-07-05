"""Structured logging helpers with request-id propagation."""

from __future__ import annotations

import contextvars
import json
import logging
import os
from datetime import UTC, datetime, timezone
from typing import Any

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="-",
)


def get_request_id() -> str:
    """Return the request id bound to the current context."""
    return _request_id.get()


def set_request_id(request_id: str):
    """Bind a request id to the current context and return the reset token."""
    return _request_id.set(request_id)


def reset_request_id(token) -> None:
    """Reset request id context after a request finishes."""
    _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Attach request_id to every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Small JSON log formatter for app and access logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        for key in (
            "method",
            "path",
            "status_code",
            "duration_ms",
            "client",
            "user_agent",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging() -> None:
    """Configure root logging once.

    Set LOG_FORMAT=text to keep the classic readable formatter locally.
    JSON is the default because production log collectors prefer structured
    fields over parsed free text.
    """
    root = logging.getLogger()
    if getattr(root, "_gaokao_logging_configured", False):
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(level)

    formatter: logging.Formatter
    if os.getenv("LOG_FORMAT", "json").lower() == "text":
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
        )
    else:
        formatter = JsonFormatter()

    if not root.handlers:
        root.addHandler(logging.StreamHandler())

    request_filter = RequestIdFilter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
        handler.addFilter(request_filter)

    root.__dict__["_gaokao_logging_configured"] = True
