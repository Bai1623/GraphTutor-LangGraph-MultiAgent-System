"""Simple per-user daily quota store.

This is intentionally small and local-file based. It protects a single-node
deployment from accidental spend spikes and can be replaced by Redis/Postgres
without changing the API boundary.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUOTA_FILE = PROJECT_ROOT / "data" / "quota" / "daily.json"


class QuotaExceeded(RuntimeError):
    def __init__(self, metric: str, limit: int, used: int) -> None:
        self.metric = metric
        self.limit = limit
        self.used = used
        super().__init__(f"Daily {metric} quota exceeded: {used}/{limit}")


@dataclass(frozen=True)
class QuotaLimits:
    requests: int
    tokens: int
    uploads: int
    retries: int

    @classmethod
    def from_env(cls) -> QuotaLimits:
        return cls(
            requests=_env_int("QUOTA_DAILY_REQUESTS", 200, 1, 100_000),
            tokens=_env_int("QUOTA_DAILY_TOKENS", 300_000, 1_000, 50_000_000),
            uploads=_env_int("QUOTA_DAILY_UPLOADS", 50, 1, 10_000),
            retries=_env_int("QUOTA_DAILY_RETRIES", 30, 0, 100_000),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "tokens": self.tokens,
            "uploads": self.uploads,
            "retries": self.retries,
        }


class DailyQuotaStore:
    def __init__(self, path: Path = DEFAULT_QUOTA_FILE, limits: QuotaLimits | None = None) -> None:
        self._path = path
        self._limits = limits or QuotaLimits.from_env()
        self._lock = Lock()

    @property
    def limits(self) -> QuotaLimits:
        return self._limits

    def consume(
        self,
        user_id: str,
        *,
        requests: int = 0,
        tokens: int = 0,
        uploads: int = 0,
        retries: int = 0,
    ) -> dict[str, Any]:
        if not quota_enabled():
            return self.snapshot(user_id)

        user_key = _normalize_user_id(user_id)
        today = _today()
        with self._lock:
            data = self._load()
            day_bucket = data.setdefault(today, {})
            counters = day_bucket.setdefault(user_key, _empty_counters())
            proposed = {
                "requests": int(counters.get("requests", 0)) + requests,
                "tokens": int(counters.get("tokens", 0)) + tokens,
                "uploads": int(counters.get("uploads", 0)) + uploads,
                "retries": int(counters.get("retries", 0)) + retries,
            }
            self._raise_if_exceeded(proposed)
            counters.update(proposed)
            counters["updated_at"] = time.time()
            self._save(data)
            return self._snapshot_from_counters(user_key, today, counters)

    def snapshot(self, user_id: str) -> dict[str, Any]:
        user_key = _normalize_user_id(user_id)
        today = _today()
        with self._lock:
            data = self._load()
            counters = data.get(today, {}).get(user_key, _empty_counters())
            return self._snapshot_from_counters(user_key, today, counters)

    def _raise_if_exceeded(self, counters: dict[str, int]) -> None:
        limits = self._limits.to_dict()
        for metric, limit in limits.items():
            used = int(counters.get(metric, 0))
            if used > limit:
                raise QuotaExceeded(metric, limit, used)

    def _snapshot_from_counters(
        self,
        user_id: str,
        day: str,
        counters: dict[str, Any],
    ) -> dict[str, Any]:
        limits = self._limits.to_dict()
        used = {
            "requests": int(counters.get("requests", 0)),
            "tokens": int(counters.get("tokens", 0)),
            "uploads": int(counters.get("uploads", 0)),
            "retries": int(counters.get("retries", 0)),
        }
        return {
            "user_id": user_id,
            "date": day,
            "enabled": quota_enabled(),
            "limits": limits,
            "used": used,
            "remaining": {
                metric: max(0, limits[metric] - value)
                for metric, value in used.items()
            },
        }

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(self._path)


_store: DailyQuotaStore | None = None


def get_quota_store() -> DailyQuotaStore:
    global _store
    if _store is None:
        _store = DailyQuotaStore()
    return _store


def set_quota_store(store: DailyQuotaStore | None) -> None:
    global _store
    _store = store


def quota_enabled() -> bool:
    return os.getenv("QUOTA_ENABLED", "true").lower() != "false"


def quota_headers(snapshot: dict[str, Any]) -> dict[str, str]:
    remaining = snapshot.get("remaining", {})
    limits = snapshot.get("limits", {})
    return {
        "X-Quota-Requests-Remaining": str(remaining.get("requests", 0)),
        "X-Quota-Tokens-Remaining": str(remaining.get("tokens", 0)),
        "X-Quota-Uploads-Remaining": str(remaining.get("uploads", 0)),
        "X-Quota-Retries-Remaining": str(remaining.get("retries", 0)),
        "X-Quota-Requests-Limit": str(limits.get("requests", 0)),
        "X-Quota-Tokens-Limit": str(limits.get("tokens", 0)),
    }


def quota_error_response(exc: QuotaExceeded) -> dict[str, Any]:
    return {
        "error": "quota_exceeded",
        "metric": exc.metric,
        "limit": exc.limit,
        "used": exc.used,
        "detail": "今日使用额度已用完，请明天再试或调整配额。",
    }


def _empty_counters() -> dict[str, Any]:
    return {
        "requests": 0,
        "tokens": 0,
        "uploads": 0,
        "retries": 0,
        "updated_at": time.time(),
    }


def _normalize_user_id(user_id: str) -> str:
    cleaned = "".join(ch for ch in (user_id or "anonymous").strip() if ch.isalnum() or ch in "._-@")
    return cleaned[:128] or "anonymous"


def _today() -> str:
    return date.today().isoformat()


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))
