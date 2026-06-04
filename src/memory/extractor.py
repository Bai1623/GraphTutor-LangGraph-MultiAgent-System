"""记忆提取器——从对话中自动识别用户关键信息

使用 Supervisor 小模型做轻量信息提取，零额外成本（不复用 DeepSeek 配额）。
每次对话后异步提取，不增加用户感知延迟。
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.llm import get_node_llm
from src.memory.long_term import get_memory_store

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """你是一个信息提取助手。从以下对话中提取关于学生的关键信息，每条信息不超过25个字。

重点关注：
1. 年级/考试年份（如"2025届高三"）
2. 选考科目（如"物化生"）
3. 学科强弱（如"数学导数较弱"）
4. 学习偏好（如"喜欢刷题"、"偏好视频讲解"）
5. 可用时间（如"每晚3小时"）
6. 情绪状态（如"考前焦虑"）
7. 目标院校或分数（如"想考浙大"）

只输出事实，每条一行，不要编号。如果没有可提取的信息，回复"无"。
不要编造信息，不要输出无关内容。"""


async def extract_and_store(user_id: str, conversation: str) -> list[str]:
    """从对话中提取关键事实并存储。

    参数：
        user_id: 用户标识（通常为 thread_id）
        conversation: 要提取信息的对话文本

    返回：
        新提取的 fact 列表
    """
    if len(conversation) < 50:
        return []

    llm = get_node_llm("supervisor")  # 用 Supervisor 小模型
    store = get_memory_store()

    try:
        response = await llm.ainvoke([
            SystemMessage(content=_EXTRACT_PROMPT),
            HumanMessage(content=f"对话内容：\n\n{conversation}"),
        ])
        content = response.content.strip()
    except Exception:
        logger.warning("Memory extraction LLM call failed", exc_info=True)
        return []

    if content == "无" or not content:
        return []

    # 分条存储
    new_facts: list[str] = []
    for line in content.split("\n"):
        fact = line.strip().lstrip("- ").strip()
        if fact and len(fact) >= 4:
            added = store.add_fact(user_id, fact)
            if added:
                new_facts.append(fact)

    if new_facts:
        logger.info("Memory extracted %d new facts for user %s", len(new_facts), user_id[:8])

    return new_facts
