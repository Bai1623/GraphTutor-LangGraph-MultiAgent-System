"""Tests for structured long-term memory behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.memory import long_term
from src.memory.long_term import MemoryRecord, MemoryStore


def _isolated_store(tmp_path, monkeypatch) -> MemoryStore:
    monkeypatch.setattr(long_term, "_STORE_DIR", tmp_path)
    monkeypatch.setattr(long_term, "_STORE_FILE", tmp_path / "memories.json")
    return MemoryStore()


def test_same_topic_supersedes_old_memory(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    assert store.upsert_memory(
        "u1",
        MemoryRecord(
            type="profile",
            topic="daily_minutes",
            content="每天可学习180分钟",
        ),
    )
    assert store.upsert_memory(
        "u1",
        MemoryRecord(
            type="profile",
            topic="daily_minutes",
            content="每天可学习60分钟",
        ),
    )

    assert store.get_facts("u1") == ["每天可学习60分钟"]
    all_records = store.get_memories("u1", include_inactive=True)
    assert {record.status for record in all_records} == {"active", "superseded"}


def test_expired_memory_is_not_retrieved(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    store.upsert_memory(
        "u1",
        MemoryRecord(
            type="episode",
            topic="emotion",
            content="今天考前焦虑",
            valid_until=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        ),
    )
    assert store.get_memories("u1") == []


def test_retrieval_prefers_subject_and_intent(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    store.upsert_memory(
        "u1",
        MemoryRecord(
            type="progress",
            subject="math",
            topic="derivative",
            content="数学导数含参讨论薄弱",
            importance=0.9,
        ),
    )
    store.upsert_memory(
        "u1",
        MemoryRecord(
            type="profile",
            subject="chinese",
            topic="preference",
            content="语文偏好阅读讲解",
        ),
    )

    result = store.retrieve(
        "u1",
        "导数题怎么做",
        intent="academic",
        subject="math",
        top_k=1,
    )
    assert result[0].topic == "derivative"


def test_legacy_string_facts_are_migrated(tmp_path, monkeypatch):
    memory_file = tmp_path / "memories.json"
    memory_file.write_text(
        '{"u1":{"facts":["数学导数较弱"],"last_updated":"2026-06-01"}}',
        encoding="utf-8",
    )
    store = _isolated_store(tmp_path, monkeypatch)
    assert store.get_facts("u1") == ["数学导数较弱"]
    assert "memories" in memory_file.read_text(encoding="utf-8")
