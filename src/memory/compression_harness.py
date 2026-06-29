"""Evaluation helpers for conversation compression quality."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.memory.artifacts import ContextArtifactStore
from src.memory.compressor import CompressionResult, SessionEpisode, estimate_message_tokens


def message_from_dict(raw: dict[str, Any]) -> BaseMessage:
    role = raw.get("role", "human")
    content = str(raw.get("content", ""))
    message_id = raw.get("id")
    if role in {"human", "user", "student"}:
        return HumanMessage(id=message_id, content=content)
    if role in {"system"}:
        return SystemMessage(id=message_id, content=content)
    return AIMessage(id=message_id, content=content)


def messages_from_case(raw_messages: list[dict[str, Any]]) -> list[BaseMessage]:
    return [message_from_dict(item) for item in raw_messages]


def text_from_messages(messages: list[BaseMessage]) -> str:
    return "\n".join(str(getattr(message, "content", "")) for message in messages)


def episode_from_case(raw: dict[str, Any] | None) -> SessionEpisode:
    return SessionEpisode.model_validate(raw or {})


def result_from_static_episode(
    *,
    before_messages: list[BaseMessage],
    episode: SessionEpisode,
    recent_messages: list[BaseMessage],
) -> CompressionResult:
    summary = SystemMessage(
        content=(
            "[会话摘要]\n"
            f"{episode.prompt_text()}\n\n"
            "[结构化数据]\n"
            f"{episode.model_dump_json()}"
        )
    )
    after_messages = [summary, *recent_messages]
    return CompressionResult(
        messages=after_messages,
        summary_json=episode.model_dump_json(),
        before_tokens=estimate_message_tokens(before_messages),
        after_tokens=estimate_message_tokens(after_messages),
        compressed=True,
    )


def token_reduction_ratio(result: CompressionResult) -> float:
    if result.before_tokens <= 0:
        return 0.0
    return round((result.before_tokens - result.after_tokens) / result.before_tokens, 4)


def _contains_term(text: str, term: str) -> bool:
    return term.strip().lower() in text.lower()


def term_retention_rate(text: str, expected_terms: list[str]) -> float:
    terms = [term for term in expected_terms if term.strip()]
    if not terms:
        return 1.0
    hits = sum(1 for term in terms if _contains_term(text, term))
    return round(hits / len(terms), 4)


def answer_consistency_score(before_text: str, after_text: str, required_terms: list[str]) -> float:
    """Estimate whether compressed context preserves answer-critical terms.

    This harness intentionally avoids live answer generation by default. The
    score measures whether terms that should drive the answer are present both
    before and after compression.
    """
    terms = [term for term in required_terms if term.strip()]
    if not terms:
        return 1.0
    before_hits = {term for term in terms if _contains_term(before_text, term)}
    if not before_hits:
        return 0.0
    after_hits = sum(1 for term in before_hits if _contains_term(after_text, term))
    return round(after_hits / len(before_hits), 4)


def _artifact_ids_from_episode(summary_json: str, after_text: str) -> set[str]:
    artifact_ids: set[str] = set(re.findall(r"ctx_[a-f0-9]{8,}", after_text))
    if not summary_json:
        return artifact_ids
    try:
        data = json.loads(summary_json)
    except json.JSONDecodeError:
        return artifact_ids
    for ref in data.get("artifact_refs") or []:
        artifact_id = str(ref.get("artifact_id") or "")
        if artifact_id:
            artifact_ids.add(artifact_id)
    for document in data.get("current_documents") or []:
        artifact_id = str(document.get("artifact_id") or "")
        if artifact_id:
            artifact_ids.add(artifact_id)
    for question in data.get("current_questions") or []:
        artifact_id = str(question.get("artifact_id") or "")
        if artifact_id:
            artifact_ids.add(artifact_id)
    return artifact_ids


def artifact_recoverability_rate(
    *,
    summary_json: str,
    after_text: str,
    expected_artifact_ids: list[str],
    store: ContextArtifactStore,
) -> float:
    expected = [artifact_id for artifact_id in expected_artifact_ids if artifact_id.strip()]
    if not expected:
        return 1.0
    present = _artifact_ids_from_episode(summary_json, after_text)
    hits = 0
    for artifact_id in expected:
        if artifact_id in present and store.load(artifact_id) is not None:
            hits += 1
    return round(hits / len(expected), 4)


@dataclass(slots=True)
class CompressionHarnessMetrics:
    case_id: str
    before_tokens: int
    after_tokens: int
    token_reduction: float
    constraint_retention: float
    answer_consistency: float
    artifact_recoverability: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "token_reduction": self.token_reduction,
            "constraint_retention": self.constraint_retention,
            "answer_consistency": self.answer_consistency,
            "artifact_recoverability": self.artifact_recoverability,
            "passed": self.passed,
        }


def evaluate_compression_result(
    *,
    case_id: str,
    before_messages: list[BaseMessage],
    result: CompressionResult,
    expected_constraints: list[str],
    answer_terms: list[str],
    expected_artifact_ids: list[str],
    thresholds: dict[str, float],
    store: ContextArtifactStore,
) -> CompressionHarnessMetrics:
    before_text = text_from_messages(before_messages)
    after_text = text_from_messages(result.messages)
    metrics = CompressionHarnessMetrics(
        case_id=case_id,
        before_tokens=result.before_tokens,
        after_tokens=result.after_tokens,
        token_reduction=token_reduction_ratio(result),
        constraint_retention=term_retention_rate(after_text, expected_constraints),
        answer_consistency=answer_consistency_score(before_text, after_text, answer_terms),
        artifact_recoverability=artifact_recoverability_rate(
            summary_json=result.summary_json,
            after_text=after_text,
            expected_artifact_ids=expected_artifact_ids,
            store=store,
        ),
        passed=False,
    )
    metrics.passed = (
        metrics.token_reduction >= thresholds.get("token_reduction", 0.0)
        and metrics.constraint_retention >= thresholds.get("constraint_retention", 1.0)
        and metrics.answer_consistency >= thresholds.get("answer_consistency", 1.0)
        and metrics.artifact_recoverability >= thresholds.get("artifact_recoverability", 1.0)
    )
    return metrics
