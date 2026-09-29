"""Shared prompt assembly for compressed session and long-term memory."""

from __future__ import annotations

import json
from collections.abc import Mapping

_ALL_SESSION_SECTIONS = frozenset({
    "task",
    "gaokao_profile",
    "subject_goals",
    "weak_points",
    "recent_scores",
    "study_preferences",
    "student_state",
    "constraints",
    "decisions",
    "knowledge_progress",
    "open_loops",
    "current_documents",
    "current_questions",
    "artifact_refs",
})

_NODE_CONTEXT_SECTIONS: dict[str, frozenset[str]] = {
    "supervisor": frozenset({
        "task",
        "gaokao_profile",
        "subject_goals",
        "weak_points",
        "student_state",
        "open_loops",
    }),
    "emotional_response": frozenset({
        "task",
        "study_preferences",
        "student_state",
        "constraints",
        "open_loops",
    }),
    "drafter_node": frozenset({
        "task",
        "gaokao_profile",
        "subject_goals",
        "weak_points",
        "recent_scores",
        "study_preferences",
        "student_state",
        "constraints",
        "decisions",
        "knowledge_progress",
        "open_loops",
    }),
    "generate_answer": frozenset({
        "task",
        "gaokao_profile",
        "weak_points",
        "study_preferences",
        "constraints",
        "decisions",
        "knowledge_progress",
        "open_loops",
        "current_documents",
        "current_questions",
        "artifact_refs",
    }),
}


def _render_session_episode(episode: Mapping, sections: frozenset[str]) -> str:
    """Render only the structured summary sections needed by one consumer."""
    lines: list[str] = []
    task = episode.get("task") or {}
    if "task" in sections:
        lines.append(
            f"任务: intent={task.get('intent', 'unknown')}, "
            f"subject={task.get('subject') or '未知'}, "
            f"topic={task.get('topic') or '未知'}, "
            f"status={task.get('status', 'active')}"
        )

    gaokao_state = episode.get("gaokao_state") or {}
    if "gaokao_profile" in sections:
        gaokao_profile = []
        for label, key in (
            ("年级", "grade"),
            ("省份", "province"),
            ("选科/方向", "exam_track"),
            ("目标分", "target_score"),
            ("目标院校", "target_university"),
        ):
            value = str(gaokao_state.get(key) or "").strip()
            if value:
                gaokao_profile.append(f"{label}: {value}")
        if gaokao_profile:
            lines.append("高考学习状态:\n" + "\n".join(f"- {item}" for item in gaokao_profile))

    labels = (
        ("科目目标", "subject_goals", gaokao_state),
        ("薄弱点", "weak_points", gaokao_state),
        ("近期成绩", "recent_scores", gaokao_state),
        ("学习偏好", "study_preferences", gaokao_state),
        ("学生状态", "student_state", episode),
        ("硬约束", "constraints", episode),
        ("已确认结论", "decisions", episode),
        ("知识进展", "knowledge_progress", episode),
        ("待处理事项", "open_loops", episode),
    )
    for label, key, source in labels:
        if key not in sections:
            continue
        items = source.get(key) or []
        if items:
            lines.append(
                f"{label}:\n" + "\n".join(f"- {item.get('text', '')}" for item in items)
            )

    documents = episode.get("current_documents") or []
    if "current_documents" in sections and documents:
        lines.append(
            "当前文档:\n"
            + "\n".join(
                (
                    f"- artifact={doc.get('artifact_id') or '无'}, "
                    f"filename={doc.get('filename') or '未知'}, "
                    f"questions={','.join(doc.get('question_numbers') or []) or '未知'}, "
                    f"summary={doc.get('summary') or ''}"
                )
                for doc in documents
            )
        )

    questions = episode.get("current_questions") or []
    if "current_questions" in sections and questions:
        lines.append(
            "当前题目:\n"
            + "\n".join(
                (
                    f"- {question.get('number') or '未编号'} "
                    f"{question.get('subject') or '未知'} "
                    f"artifact={question.get('artifact_id') or '无'}: "
                    f"{question.get('stem_preview') or ''}"
                )
                for question in questions
            )
        )

    artifact_refs = episode.get("artifact_refs") or []
    if "artifact_refs" in sections and artifact_refs:
        lines.append(
            "可恢复 artifact 引用:\n"
            + "\n".join(
                f"- {ref.get('artifact_id')} ({ref.get('kind') or 'unknown'}): {ref.get('preview') or ''}"
                for ref in artifact_refs
                if ref.get("artifact_id")
            )
        )
    return "\n".join(lines)


def _build_memory_context(
    state: Mapping,
    *,
    include_session: bool,
    session_sections: frozenset[str],
) -> str:
    sections: list[str] = []
    long_term = str(state.get("long_term_memory", "")).strip()
    if long_term:
        sections.append(long_term)

    session_summary = str(state.get("session_summary", "")).strip()
    if include_session and session_summary:
        try:
            episode = json.loads(session_summary)
            if not isinstance(episode, Mapping):
                raise TypeError("session summary must be a JSON object")
            session_text = _render_session_episode(episode, session_sections)
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Older deployments may still contain free-text summaries. Keep them
            # intact until the next successful structured compaction.
            session_text = session_summary
        if session_text:
            sections.append(f"[本次会话较早内容]\n{session_text}")

    return "\n\n".join(sections)


def build_memory_context(state: Mapping, *, include_session: bool = True) -> str:
    """Return compact memory sections without replaying the full conversation."""
    return _build_memory_context(
        state,
        include_session=include_session,
        session_sections=_ALL_SESSION_SECTIONS,
    )


def build_node_context(state: Mapping, node_name: str) -> str:
    """Return a least-context projection for a prompt-consuming graph node.

    Unknown nodes deliberately receive the full compact memory. This makes new
    nodes backward compatible until an explicit allowlist is added.
    """
    session_sections = _NODE_CONTEXT_SECTIONS.get(node_name, _ALL_SESSION_SECTIONS)
    return _build_memory_context(
        state,
        include_session=True,
        session_sections=session_sections,
    )
