"""Extract typed, evidence-aware long-term memories from a completed turn."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.graph.llm import get_node_llm
from src.memory.long_term import MemoryRecord, get_memory_store

logger = logging.getLogger(__name__)


class ExtractedMemory(BaseModel):
    type: Literal["profile", "progress", "episode"]
    subject: str = ""
    topic: str = ""
    content: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    ttl_days: int | None = Field(default=None, ge=1, le=3650)


class ExtractionResult(BaseModel):
    memories: list[ExtractedMemory] = Field(default_factory=list)


_EXTRACT_PROMPT = """从对话中提取值得跨会话保留的学生记忆。

类型：
- profile：年级、选科、长期目标、稳定学习偏好、通常可用时间
- progress：学科弱项、知识点掌握、典型错误、近期成绩
- episode：近期计划、需要后续跟进的事件、短期情绪状态

规则：
1. 只记录学生明确陈述或对话中有直接证据的信息；
2. 不把老师的建议当成学生事实；
3. 同一含义只输出一条；
4. 情绪状态和临时安排使用较短 ttl_days，稳定画像可不设置；
5. content 简洁但必须保留数字、日期、科目和知识点。"""


async def extract_and_store(
    user_id: str,
    conversation: str,
    *,
    source_thread_id: str = "",
) -> list[str]:
    if len(conversation) < 50:
        return []

    llm = get_node_llm("supervisor", streaming=False)
    structured_llm = llm.with_structured_output(ExtractionResult)
    try:
        result = await structured_llm.ainvoke(
            [
                SystemMessage(content=_EXTRACT_PROMPT),
                HumanMessage(content=f"对话内容：\n\n{conversation}"),
            ]
        )
    except Exception:
        logger.warning("Memory extraction LLM call failed", exc_info=True)
        return []

    store = get_memory_store()
    new_memories: list[str] = []
    now = datetime.now(UTC)
    for extracted in result.memories:
        valid_until = (
            (now + timedelta(days=extracted.ttl_days)).isoformat()
            if extracted.ttl_days
            else None
        )
        record = MemoryRecord(
            type=extracted.type,
            subject=extracted.subject,
            topic=extracted.topic,
            content=extracted.content,
            confidence=extracted.confidence,
            importance=extracted.importance,
            valid_until=valid_until,
            source_thread_id=source_thread_id,
        )
        if store.upsert_memory(user_id, record):
            new_memories.append(record.content)

    if new_memories:
        logger.info(
            "Memory extracted %d new records for user %s",
            len(new_memories),
            user_id[:8],
        )
    return new_memories
