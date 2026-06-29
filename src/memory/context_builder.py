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
                ("科目目标", "gaokao_state.subject_goals"),
                ("薄弱点", "gaokao_state.weak_points"),
                ("近期成绩", "gaokao_state.recent_scores"),
                ("学习偏好", "gaokao_state.study_preferences"),
                ("学生状态", "student_state"),
                ("硬约束", "constraints"),
                ("已确认结论", "decisions"),
                ("知识进展", "knowledge_progress"),
                ("待处理事项", "open_loops"),
            )
            gaokao_state = episode.get("gaokao_state") or {}
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

            for label, key in labels:
                if "." in key:
                    parent, child = key.split(".", 1)
                    items = (episode.get(parent) or {}).get(child, [])
                else:
                    items = episode.get(key, [])
                if items:
                    lines.append(
                        f"{label}:\n"
                        + "\n".join(f"- {item.get('text', '')}" for item in items)
                    )
            documents = episode.get("current_documents") or []
            if documents:
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
            if questions:
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
            if artifact_refs:
                lines.append(
                    "可恢复 artifact 引用:\n"
                    + "\n".join(
                        f"- {ref.get('artifact_id')} ({ref.get('kind') or 'unknown'}): {ref.get('preview') or ''}"
                        for ref in artifact_refs
                        if ref.get("artifact_id")
                    )
                )
            session_text = "\n".join(lines)
        except Exception:
            session_text = session_summary
        sections.append(f"[本次会话较早内容]\n{session_text}")

    if not sections:
        return ""
    return "\n\n".join(sections)
