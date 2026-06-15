"""Token-budgeted working memory and structured session compression."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from pydantic import BaseModel, Field

from src.config import get_setting
from src.graph.llm import get_node_llm

logger = logging.getLogger(__name__)


class AnchoredItem(BaseModel):
    """A compact fact that can be traced back to source messages."""

    text: str
    source_message_ids: list[str] = Field(default_factory=list)


class SessionTask(BaseModel):
    intent: Literal["academic", "planning", "emotional", "unknown"] = "unknown"
    subject: str = ""
    topic: str = ""
    status: Literal["active", "blocked", "completed"] = "active"


class SessionEpisode(BaseModel):
    """Loss-resistant summary for content that left the recent-message window."""

    task: SessionTask = Field(default_factory=SessionTask)
    student_state: list[AnchoredItem] = Field(default_factory=list)
    constraints: list[AnchoredItem] = Field(default_factory=list)
    decisions: list[AnchoredItem] = Field(default_factory=list)
    knowledge_progress: list[AnchoredItem] = Field(default_factory=list)
    open_loops: list[AnchoredItem] = Field(default_factory=list)

    def prompt_text(self) -> str:
        sections: list[str] = [
            (
                f"任务: intent={self.task.intent}, subject={self.task.subject or '未知'}, "
                f"topic={self.task.topic or '未知'}, status={self.task.status}"
            )
        ]
        labels = (
            ("学生状态", self.student_state),
            ("硬约束", self.constraints),
            ("已确认结论", self.decisions),
            ("知识进展", self.knowledge_progress),
            ("待处理事项", self.open_loops),
        )
        for label, items in labels:
            if items:
                sections.append(f"{label}:\n" + "\n".join(f"- {item.text}" for item in items))
        return "\n".join(sections)


@dataclass(slots=True)
class CompressionResult:
    messages: list[BaseMessage]
    summary_json: str
    before_tokens: int
    after_tokens: int
    compressed: bool


def estimate_message_tokens(messages: list[BaseMessage]) -> int:
    """Use LangChain's model-agnostic approximate token counter."""
    if not messages:
        return 0
    # Asking trim_messages for a very large budget exercises the same approximate
    # counter used by LangChain without depending on a provider tokenizer.
    total_chars = sum(len(_message_text(message)) for message in messages)
    # Chinese characters are commonly close to one token; Latin text is closer
    # to four characters per token. This conservative blend avoids late compaction.
    cjk_chars = sum(
        1
        for message in messages
        for char in _message_text(message)
        if "\u3400" <= char <= "\u9fff"
    )
    non_cjk_chars = max(total_chars - cjk_chars, 0)
    return cjk_chars + (non_cjk_chars + 3) // 4 + len(messages) * 4


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _is_session_summary(message: BaseMessage) -> bool:
    return isinstance(message, SystemMessage) and _message_text(message).startswith("[会话摘要]")


def _split_recent_window(
    messages: list[BaseMessage],
    *,
    recent_turns: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split on human turns so a user/assistant exchange is not cut in half."""
    clean_messages = [message for message in messages if not _is_session_summary(message)]
    index = len(clean_messages) - 1
    human_turns = 0
    while index >= 0:
        if isinstance(clean_messages[index], HumanMessage):
            human_turns += 1
            if human_turns > recent_turns:
                break
        index -= 1
    return clean_messages[: index + 1], clean_messages[index + 1 :]


def _serialize_messages(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "学生"
        elif isinstance(message, SystemMessage):
            role = "系统"
        else:
            role = "老师"
        message_id = getattr(message, "id", None) or "unknown"
        lines.append(f"[{message_id}] {role}: {_message_text(message)}")
    return "\n".join(lines)


def _parse_episode(existing_summary: str) -> SessionEpisode:
    if not existing_summary:
        return SessionEpisode()
    text = existing_summary.strip()
    if text.startswith("[会话摘要]"):
        text = text.removeprefix("[会话摘要]").strip()
    try:
        return SessionEpisode.model_validate_json(text)
    except Exception:
        # Compatibility with the previous free-text summary.
        return SessionEpisode(
            decisions=[AnchoredItem(text=text[:1000], source_message_ids=[])]
        )


def _summary_message(episode: SessionEpisode) -> SystemMessage:
    return SystemMessage(
        content=(
            "[会话摘要]\n"
            f"{episode.prompt_text()}\n\n"
            "[结构化数据]\n"
            f"{episode.model_dump_json()}"
        )
    )


async def _summarize_episode(
    overflow_messages: list[BaseMessage],
    existing_summary: str,
) -> SessionEpisode:
    llm = get_node_llm("supervisor", streaming=False)
    structured_llm = llm.with_structured_output(SessionEpisode)
    old_episode = _parse_episode(existing_summary)
    source_text = _serialize_messages(overflow_messages)
    max_input_tokens = int(get_setting("memory.compact_input_tokens", 12000))

    # Keep complete messages where possible. trim_messages only applies a hard
    # fallback when a very large overflow would exceed the compressor budget.
    trimmed = trim_messages(
        overflow_messages,
        max_tokens=max_input_tokens,
        token_counter="approximate",
        strategy="last",
        allow_partial=False,
        start_on="human",
    )
    source_text = _serialize_messages(trimmed)

    prompt = (
        "将旧会话事件和新增对话合并成结构化摘要。严格遵守：\n"
        "1. 数字、日期、题号、公式、目标分数、学习时长不得改写或猜测；\n"
        "2. 仅记录对后续辅导有用的信息；\n"
        "3. 每条信息填写原消息方括号中的 message id；\n"
        "4. 已完成事项不要继续放在 open_loops；\n"
        "5. 没有证据的字段保持空，不得编造。\n\n"
        f"旧会话事件:\n{old_episode.model_dump_json()}\n\n"
        f"新增对话:\n{source_text}"
    )
    return await structured_llm.ainvoke(
        [
            SystemMessage(content="你负责高考辅导会话压缩，输出严格的结构化数据。"),
            HumanMessage(content=prompt),
        ]
    )


def _deterministic_trim(
    recent_messages: list[BaseMessage],
    *,
    max_tokens: int,
) -> list[BaseMessage]:
    """Always keep the latest user turn even when summarization is unavailable."""
    trimmed = trim_messages(
        recent_messages,
        max_tokens=max_tokens,
        token_counter="approximate",
        strategy="last",
        allow_partial=False,
        start_on="human",
    )
    if trimmed:
        return trimmed

    # A pathological tiny budget must not erase the current request. Return the
    # latest complete user turn even if it temporarily exceeds the target.
    for index in range(len(recent_messages) - 1, -1, -1):
        if isinstance(recent_messages[index], HumanMessage):
            return recent_messages[index:]
    return recent_messages[-1:]


async def compress_conversation(
    messages: list[BaseMessage],
    *,
    recent_turns: int | None = None,
    existing_summary: str = "",
    soft_limit_tokens: int | None = None,
) -> CompressionResult:
    """Compact old turns into a structured episode under a token budget."""
    recent_turns = recent_turns or int(get_setting("memory.recent_turns", 5))
    soft_limit_tokens = soft_limit_tokens or int(
        get_setting("memory.soft_limit_tokens", 18000)
    )
    min_overflow_tokens = int(
        get_setting("memory.compact_min_overflow_tokens", 3000)
    )
    before_tokens = estimate_message_tokens(messages)
    if before_tokens <= soft_limit_tokens:
        return CompressionResult(
            messages=list(messages),
            summary_json=existing_summary,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            compressed=False,
        )

    overflow, recent = _split_recent_window(messages, recent_turns=recent_turns)
    if not overflow or estimate_message_tokens(overflow) < min_overflow_tokens:
        trimmed = _deterministic_trim(recent, max_tokens=soft_limit_tokens)
        after_tokens = estimate_message_tokens(trimmed)
        return CompressionResult(
            messages=trimmed,
            summary_json=existing_summary,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            compressed=len(trimmed) < len(messages),
        )

    try:
        episode = await _summarize_episode(overflow, existing_summary)
        summary_json = episode.model_dump_json()
        summary_message = _summary_message(episode)
        remaining_budget = max(
            soft_limit_tokens - estimate_message_tokens([summary_message]),
            64,
        )
        compacted = [
            summary_message,
            *_deterministic_trim(recent, max_tokens=remaining_budget),
        ]
    except Exception:
        logger.warning(
            "Structured conversation compression failed; applying deterministic trim",
            exc_info=True,
        )
        summary_json = existing_summary
        compacted = recent

    if not summary_json:
        compacted = _deterministic_trim(compacted, max_tokens=soft_limit_tokens)
    after_tokens = estimate_message_tokens(compacted)
    logger.info(
        "Conversation compressed: messages=%d->%d tokens=%d->%d",
        len(messages),
        len(compacted),
        before_tokens,
        after_tokens,
    )
    return CompressionResult(
        messages=compacted,
        summary_json=summary_json,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        compressed=True,
    )
