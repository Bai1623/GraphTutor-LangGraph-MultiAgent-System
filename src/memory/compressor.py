"""对话记忆三层压缩——控制上下文膨胀，降低 Token 消耗

设计灵感来自 Claude Code 的五层压缩，简化为三层适配高考辅导场景：

Layer 1 — 窗口消息 (Window, 完整保留):
  最近 N 轮对话（默认 8 轮 = 16 条消息），一字不改保留。
  这是"工作记忆"——用户刚说了什么、系统刚回了什么。

Layer 2 — 会话摘要 (Session Summary, 增量压缩):
  超过窗口的旧消息 → LLM 压缩为一段摘要，追加到消息列表头部。
  每次触发压缩时，旧摘要 + 新溢出消息 → 新一轮 LLM → 新摘要。
  这是"中期记忆"——"之前聊了什么，关键信息是什么"。

Layer 3 — 长期事实 (Long-term Facts, 提取式):
  跨会话持久化的离散事实（年级、弱项、偏好等）。
  已由 MemoryStore + extractor.py 实现。

压缩触发条件:
- 对话轮数 > WINDOW_SIZE（默认 8 轮）
- 溢出部分累积到一定长度（> 500 chars）才触发，避免无意义压缩

为什么用 Qwen2.5-7B 做压缩？
- 摘要任务是轻量级 NLP 任务，小模型足够
- 零额外成本（复用 Supervisor 模型）
- 摘要约 200 tokens，比原始消息约 2000 tokens 省 90%

面试能讲的数字：
- 假设每轮 500 tokens × 20 轮 = 10K tokens 历史
- 压缩后：8 轮窗口(4K) + 摘要(200) = 4.2K，节省 58%
- 多轮对话越长效果越明显
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.llm import get_node_llm

logger = logging.getLogger(__name__)

# 默认配置
WINDOW_SIZE = 8       # 保留最近 N 轮对话（每轮 = 用户消息 + AI 回复）
MIN_OVERFLOW = 500    # 溢出内容至少 N 字符才触发压缩
SUMMARY_MAX = 400     # 摘要最大长度（字符）


async def compress_conversation(
    messages: list,
    *,
    window_size: int = WINDOW_SIZE,
    existing_summary: str = "",
) -> list:
    """压缩对话历史——将超出窗口的旧消息替换为摘要。

    参数:
        messages: 完整消息列表 [HumanMessage, AIMessage, ...]
        window_size: 保留最近几轮对话（默认 8）
        existing_summary: 已有的会话摘要（增量压缩时传入）

    返回:
        压缩后的消息列表:
        [SystemMessage(摘要), ...窗口内消息...]

    工作流程:
    1. 分离窗口内(最近 window_size 轮)和溢出(更早的)
    2. 如果溢出为空 → 直接返回原消息
    3. 将溢出消息 + 旧摘要拼成文本，调用小模型生成新摘要
    4. 返回 [SystemMessage(新摘要), ...窗口内消息]
    """
    # 计算轮数（每轮 = HumanMessage + AIMessage 对）
    pairs = []
    i = len(messages) - 1
    window_msgs = []
    window_count = 0

    # 从后往前取 window_size 轮
    while i >= 0 and window_count < window_size:
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            window_count += 1
        window_msgs.insert(0, msg)
        i -= 1

    # 溢出部分
    overflow_msgs = messages[:i + 1] if i >= 0 else []

    # 不需要压缩
    if not overflow_msgs:
        return list(messages)

    # 溢出太少也不压
    overflow_text = "\n".join(
        f"{'学生' if isinstance(m, HumanMessage) else '老师'}: {m.content[:200]}"
        for m in overflow_msgs
        if hasattr(m, "content") and m.content
    )
    if len(overflow_text) < MIN_OVERFLOW:
        return list(messages)

    # — 调用小模型生成新摘要 —
    try:
        new_summary = await _summarize(overflow_text, existing_summary)
    except Exception:
        logger.warning("Conversation compression failed, keeping full history", exc_info=True)
        return list(messages)

    # 构建压缩后的消息列表
    result = []
    if new_summary:
        result.append(SystemMessage(
            content=f"[会话摘要]\n{new_summary}\n\n--- 最近对话 ---"
        ))
    result.extend(window_msgs)

    logger.info(
        "Conversation compressed: %d msgs → %d (summary %d chars)",
        len(messages), len(result), len(new_summary) if new_summary else 0,
    )
    return result


async def _summarize(
    overflow_text: str,
    existing_summary: str = "",
) -> str:
    """调用 Supervisor 小模型生成增量摘要。

    增量摘要优于重建摘要：
    - 保留历史重要信息（旧摘要中的关键点）
    - 只追加新的要点（溢出部分的增量信息）
    - 避免每次压缩都从头总结，丢失早期关键事实
    """
    llm = get_node_llm("supervisor")  # Qwen2.5-7B，零成本

    if existing_summary:
        prompt = (
            f"以下是一段对话的旧摘要和新内容。请将新旧信息合并为一段约{SUMMARY_MAX}字的更新摘要，"
            f"保留所有关键信息（年级、科目弱项、偏好、目标、情绪状态等）。\n\n"
            f"## 旧摘要\n{existing_summary[:500]}\n\n"
            f"## 新对话内容\n{overflow_text[:2000]}\n\n"
            f"请用 2-5 句话输出更新后的摘要："
        )
    else:
        prompt = (
            f"从以下对话中提取关键信息，生成一段约{SUMMARY_MAX}字的摘要。"
            f"重点关注：年级/考试年份、学科弱项、学习偏好、情绪状态、重要讨论内容。\n\n"
            f"## 对话内容\n{overflow_text[:2000]}\n\n"
            f"请用 2-5 句话输出摘要："
        )

    messages = [
        SystemMessage(content="你是一个对话信息压缩助手。从对话中提取关键信息，输出简洁摘要。"),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    summary = response.content.strip()

    # 截断过长摘要
    if len(summary) > SUMMARY_MAX + 100:
        summary = summary[:SUMMARY_MAX] + "..."

    return summary
