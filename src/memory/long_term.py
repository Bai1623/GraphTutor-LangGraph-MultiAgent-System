"""Structured cross-session memory with conflict replacement and retrieval."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.config import get_setting

logger = logging.getLogger(__name__)

_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory"
_STORE_FILE = _STORE_DIR / "user_memories.json"
_MAX_MEMORIES_PER_USER = 60


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex}")
    type: Literal["profile", "progress", "episode"] = "profile"
    subject: str = ""
    topic: str = ""
    content: str
    confidence: float = 0.8
    importance: float = 0.6
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    last_confirmed_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    valid_until: str | None = None
    source_thread_id: str = ""
    source_message_ids: list[str] = Field(default_factory=list)
    status: Literal["active", "superseded", "expired"] = "active"


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    chinese = set(re.findall(r"[\u3400-\u9fff]{2,}", lowered))
    latin = set(re.findall(r"[a-z0-9_]+", lowered))
    # Character bigrams make lexical retrieval useful without another embedding call.
    bigrams = {
        lowered[index : index + 2]
        for index in range(max(len(lowered) - 1, 0))
        if "\u3400" <= lowered[index] <= "\u9fff"
    }
    return chinese | latin | bigrams


def _is_expired(record: MemoryRecord) -> bool:
    if not record.valid_until:
        return False
    try:
        return datetime.fromisoformat(record.valid_until) <= datetime.now(UTC)
    except ValueError:
        return False


def _recency(record: MemoryRecord) -> float:
    try:
        updated = datetime.fromisoformat(record.last_confirmed_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        age_days = max((datetime.now(UTC) - updated).days, 0)
        return 1.0 / (1.0 + age_days / 30)
    except ValueError:
        return 0.5


class MemoryStore:
    """Small JSON-backed store suitable for the current single-process demo."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        if _STORE_FILE.exists():
            try:
                with open(_STORE_FILE, encoding="utf-8") as file:
                    self._data = json.load(file)
                self._migrate_legacy_entries()
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load memory file, starting fresh")
                self._data = {}
        self._loaded = True

    def _migrate_legacy_entries(self) -> None:
        """Convert the previous string-list format in memory, then save lazily."""
        changed = False
        for _user_id, entry in list(self._data.items()):
            if "memories" in entry:
                continue
            legacy_facts = entry.get("facts", [])
            entry["memories"] = [
                MemoryRecord(content=fact).model_dump() for fact in legacy_facts
            ]
            entry.pop("facts", None)
            changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = _STORE_FILE.with_suffix(".json.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as file:
                json.dump(self._data, file, ensure_ascii=False, indent=2)
            temporary.replace(_STORE_FILE)
        except Exception:
            logger.warning("Failed to save memories", exc_info=True)

    def _records(self, user_id: str) -> list[MemoryRecord]:
        entry = self._data.get(user_id, {})
        records: list[MemoryRecord] = []
        for raw in entry.get("memories", []):
            try:
                record = MemoryRecord.model_validate(raw)
                if _is_expired(record) and record.status == "active":
                    record.status = "expired"
                    raw["status"] = "expired"
                records.append(record)
            except Exception:
                logger.warning("Ignoring invalid memory record for user %s", user_id)
        return records

    def upsert_memory(self, user_id: str, memory: MemoryRecord) -> bool:
        """Insert a memory and supersede an active record with the same key."""
        normalized_content = memory.content.strip()
        if len(normalized_content) < 4:
            return False
        memory.content = normalized_content

        with self._lock:
            self._ensure_loaded()
            entry = self._data.setdefault(
                user_id, {"memories": [], "last_updated": ""}
            )
            records = self._records(user_id)
            for record in records:
                if (
                    record.status == "active"
                    and record.content == memory.content
                    and record.type == memory.type
                ):
                    return False

            key = (memory.type, memory.subject, memory.topic)
            if memory.topic:
                for raw, record in zip(entry["memories"], records, strict=False):
                    if record.status == "active" and (
                        record.type,
                        record.subject,
                        record.topic,
                    ) == key:
                        raw["status"] = "superseded"

            entry["memories"].append(memory.model_dump())
            active_records = [
                raw
                for raw in entry["memories"]
                if raw.get("status", "active") == "active"
            ]
            if len(active_records) > _MAX_MEMORIES_PER_USER:
                active_ids = {raw["memory_id"] for raw in active_records[-_MAX_MEMORIES_PER_USER:]}
                for raw in entry["memories"]:
                    if (
                        raw.get("status", "active") == "active"
                        and raw.get("memory_id") not in active_ids
                    ):
                        raw["status"] = "superseded"

            entry["last_updated"] = datetime.now(UTC).isoformat()
            self._save()
            return True

    def add_fact(self, user_id: str, fact: str) -> bool:
        """Backward-compatible profile-memory insertion."""
        return self.upsert_memory(user_id, MemoryRecord(content=fact))

    def get_memories(
        self,
        user_id: str,
        *,
        include_inactive: bool = False,
    ) -> list[MemoryRecord]:
        with self._lock:
            self._ensure_loaded()
            records = self._records(user_id)
            if include_inactive:
                return records
            return [record for record in records if record.status == "active"]

    def get_facts(self, user_id: str) -> list[str]:
        return [record.content for record in self.get_memories(user_id)]

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        intent: str = "",
        subject: str = "",
        top_k: int | None = None,
    ) -> list[MemoryRecord]:
        top_k = top_k or int(get_setting("memory.long_term_top_k", 6))
        query_terms = _terms(f"{query} {subject}")
        type_bonus = {
            "academic": {"progress": 0.18},
            "planning": {"profile": 0.12, "progress": 0.08},
            "emotional": {"episode": 0.15, "profile": 0.05},
        }.get(intent, {})

        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.get_memories(user_id):
            memory_terms = _terms(
                f"{record.content} {record.subject} {record.topic}"
            )
            overlap = (
                len(query_terms & memory_terms) / max(len(query_terms), 1)
                if query_terms
                else 0.0
            )
            subject_bonus = 0.15 if subject and record.subject == subject else 0.0
            score = (
                0.40 * overlap
                + 0.25 * record.importance
                + 0.20 * _recency(record)
                + 0.15 * record.confidence
                + subject_bonus
                + type_bonus.get(record.type, 0.0)
            )
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def clear_facts(self, user_id: str) -> None:
        with self._lock:
            self._ensure_loaded()
            if user_id in self._data:
                del self._data[user_id]
                self._save()

    def summarize_for_prompt(
        self,
        user_id: str,
        query: str = "",
        *,
        intent: str = "",
        subject: str = "",
    ) -> str:
        records = self.retrieve(
            user_id,
            query,
            intent=intent,
            subject=subject,
        )
        if not records:
            return ""
        lines = ["[与当前请求相关的用户记忆]"]
        for record in records:
            freshness = record.last_confirmed_at[:10]
            lines.append(
                f"- ({record.type}, 确认于{freshness}, 置信度{record.confidence:.2f}) "
                f"{record.content}"
            )
        return "\n".join(lines)


_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
