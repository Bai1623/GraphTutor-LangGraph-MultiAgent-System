"""Disk-backed storage for large context artifacts.

Large tool outputs should not be carried verbatim through LangGraph state or
LLM prompts. This module stores the full payload on disk and returns compact
references that can safely remain in state, summaries, and eval traces.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "context_artifacts"
_DEFAULT_PREVIEW_CHARS = 800
_DEFAULT_RECOVERY_CHARS = 6000
_ARTIFACT_ID_PATTERN = re.compile(r"^ctx_[A-Za-z0-9_-]{8,128}$")


class ContextArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    path: str
    preview: str
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def make_preview(value: Any, *, max_chars: int = _DEFAULT_PREVIEW_CHARS) -> str:
    """Return a whitespace-normalized preview for prompt/state use."""
    text = " ".join(_coerce_text(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _stats_for_payload(payload: Any) -> dict[str, Any]:
    raw = _coerce_text(payload)
    stats: dict[str, Any] = {
        "chars": len(raw),
    }
    if isinstance(payload, list):
        stats["items"] = len(payload)
    elif isinstance(payload, dict):
        stats["keys"] = sorted(str(key) for key in payload)[:20]
    return stats


class ContextArtifactStore:
    """Small JSON store for recoverable context payloads."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _STORE_DIR

    def put(
        self,
        *,
        kind: str,
        payload: Any,
        preview_source: Any | None = None,
        metadata: dict[str, Any] | None = None,
        max_preview_chars: int = _DEFAULT_PREVIEW_CHARS,
    ) -> ContextArtifactRef:
        created_at = _now()
        artifact_id = f"ctx_{uuid.uuid4().hex}"
        day_dir = self.root / created_at[:10]
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{artifact_id}.json"
        preview = make_preview(
            payload if preview_source is None else preview_source,
            max_chars=max_preview_chars,
        )
        body = {
            "artifact_id": artifact_id,
            "kind": kind,
            "created_at": created_at,
            "metadata": metadata or {},
            "payload": payload,
        }
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        return ContextArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            path=str(path),
            preview=preview,
            stats=_stats_for_payload(payload),
            created_at=created_at,
        )

    def load(self, artifact_id: str) -> dict[str, Any] | None:
        if not _ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            return None
        for path in self.root.glob(f"**/{artifact_id}.json"):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def recover(
        self,
        artifact_id: str,
        *,
        question_number: str | None = None,
        page: int | None = None,
        max_chars: int = _DEFAULT_RECOVERY_CHARS,
    ) -> dict[str, Any] | None:
        """Recover a bounded payload, optionally narrowed by question or source page."""
        body = self.load(artifact_id)
        if body is None:
            return None
        payload = body.get("payload")
        selected = payload
        selector: dict[str, Any] = {}

        questions = payload.get("questions", []) if isinstance(payload, dict) else []
        if question_number:
            normalized = question_number.strip()
            selector["question_number"] = normalized
            selected = [
                item
                for item in questions
                if str(item.get("number", "")).strip() == normalized
            ]
        elif page is not None:
            selector["page"] = page
            selected = [
                item
                for item in questions
                if page in (item.get("source_pages") or [])
            ]

        text = _coerce_text(selected)
        truncated = len(text) > max_chars
        content = text[:max_chars].rstrip() + ("…" if truncated else "")
        return {
            "artifact_id": artifact_id,
            "kind": body.get("kind", "unknown"),
            "selector": selector,
            "content": content,
            "matched_items": len(selected) if isinstance(selected, list) else None,
            "truncated": truncated,
            "full_chars": len(text),
        }


def get_context_artifact_store() -> ContextArtifactStore:
    return ContextArtifactStore()


def artifact_ref_dict(ref: ContextArtifactRef) -> dict[str, Any]:
    return ref.model_dump()


def compact_with_artifact(
    item: dict[str, Any],
    *,
    kind: str,
    text_key: str = "content",
    preview_chars: int = _DEFAULT_PREVIEW_CHARS,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a dict payload and replace the large text field with a preview."""
    text = _coerce_text(item.get(text_key))
    ref = get_context_artifact_store().put(
        kind=kind,
        payload=item,
        preview_source=text or item,
        metadata=metadata,
        max_preview_chars=preview_chars,
    )
    compact = dict(item)
    compact[text_key] = ref.preview
    compact["preview"] = ref.preview
    compact["artifact_id"] = ref.artifact_id
    compact["artifact_ref"] = artifact_ref_dict(ref)
    compact["recoverable"] = True
    compact["full_content_chars"] = len(text)
    return compact
