"""Shared prompt assembly for compressed session and long-term memory."""

from __future__ import annotations

import json
from typing import Mapping


def build_memory_context(state: Mapping, *, include_session: bool = True) -> str:
    """Return compact memory sections without replaying the full conversation."""
    sections: list[str] = []
    long_term = str(state.get("long_term_memory", "")).strip()
    if long_term:
        sections.append(long_term)

    session_summary = str(state.get("session_summary", "")).strip()
    if include_session and session_summary:
        try:
            episode = json.loads(session_summary)
            task = episode.get("task", {})
            lines = [
                (
                    f"任务: intent={task.get('intent', 'unknown')}, "
                    f"subject={task.get('subject') or '未知'}, "
                    f"topic={task.get('topic') or '未知'}, "
                    f"status={task.get('status', 'active')}"
                )
            ]
            labels = (
                ("学生状态", "student_state"),
                ("硬约束", "constraints"),
                ("已确认结论", "decisions"),
                ("知识进展", "knowledge_progress"),
                ("待处理事项", "open_loops"),
            )
            for label, key in labels:
                items = episode.get(key, [])
                if items:
                    lines.append(
                        f"{label}:\n"
                        + "\n".join(f"- {item.get('text', '')}" for item in items)
                    )
            session_text = "\n".join(lines)
        except Exception:
            session_text = session_summary
        sections.append(f"[本次会话较早内容]\n{session_text}")

    if not sections:
        return ""
    return "\n\n".join(sections)
