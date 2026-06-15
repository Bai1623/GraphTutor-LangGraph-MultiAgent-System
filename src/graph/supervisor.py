"""Supervisor（意图路由）——系统的"前台调度员"。

Supervisor 是整个多智能体系统的入口节点，负责两项工作：
1. **意图分类**：判断用户输入属于 academic / planning / emotional / unknown
2. **关键词提取**：如果意图是 academic，提取关键知识点供 RAG 检索使用

两项工作在一次 LLM 调用中完成（通过 Structured Output），避免了
"先分类再提取"的两次调用延迟。

为什么 Supervisor 不用 DeepSeek 而用 Qwen2.5-7B？
- Qwen2.5-7B 通过 SiliconFlow 部署，延迟 ~200ms（vs DeepSeek ~1s）
- Classification 任务不需要大模型，7B 足够准确
- 节省 DeepSeek 配额（虽然便宜但能省则省）
- temperature=0.0 保证确定性路由，同一输入永远得到相同分类
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.llm import get_node_llm
from src.graph.state import TutorState
from src.memory.context_builder import build_memory_context
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


# ============================================================================
# 结构化输出模型
# ============================================================================

class SupervisorOutput(BaseModel):
    """Supervisor 的结构化输出。

    使用 Pydantic 的 Literal 类型限制 intent 只能取四个值之一，
    配合 with_structured_output() 确保 LLM 输出的 JSON 被精确解析。
    """
    intent: Literal["academic", "planning", "emotional", "unknown"]
    keywords: list[str]   # 学术问题时提取的关键知识点
    confidence: float     # 置信度 (0.0-1.0)


# 从配置加载有效意图列表（防御式编程：如果模型返回了不在列表中的值）
_VALID_INTENTS = set(get_setting(
    "supervisor.valid_intents",
    ["academic", "planning", "emotional", "unknown"],
))


# ============================================================================
# Supervisor 主节点
# ============================================================================

@traced_node
async def supervisor_node(state: TutorState) -> dict:
    """分类用户意图并提取关键词。

    工作流程：
    1. 获取最后一条用户消息
    2. 调用 Qwen2.5-7B（with_structured_output）进行意图分类
    3. 如果 intent=academic 且有 keywords，检测学科（math/chinese）
    4. 返回 {intent, subject, keypoints} 供后续路由使用

    异常处理：如果 LLM 调用失败，默认回退到 academic（宁可答非所问
    也不让用户感知到系统故障）。
    """
    llm = get_node_llm("supervisor")
    # with_structured_output 让 LLM 直接返回 Pydantic 对象
    structured_llm = llm.with_structured_output(SupervisorOutput)

    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # —— 加载长期记忆：首次对话时从 MemoryStore 读取 ——
    memory_context = build_memory_context(state)
    memory_section = f"\n\n{memory_context}" if memory_context else ""
    if memory_context:
        logger.info("Memory loaded: %d chars for this user", len(memory_context))

    temperature = get_setting("supervisor.temperature", 0.0)
    model_name = get_setting("supervisor.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    with traced_llm_call(
        model_name=model_name,
        node_name="supervisor",
        temperature=temperature,
    ):
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=load_prompt("supervisor_system") + memory_section),
                HumanMessage(content=user_text),
            ])
            intent = result.intent
            subject = "other"
            keypoints = result.keywords

            # —— 学科检测（基于关键词匹配，不额外调用 LLM）——
            # 为 RAG 检索提供 subject 过滤器，缩小检索范围
            if intent == "academic" and keypoints:
                query_lower = user_text.lower()
                math_keywords = {"数学", "函数", "方程", "几何", "代数", "概率", "向量",
                                 "导数", "积分", "椭圆", "双曲线", "抛物线", "三角"}
                chinese_keywords = {"语文", "作文", "文言文", "古诗", "阅读理解", "诗词",
                                    "鉴赏", "修辞", "散文", "小说"}
                if any(kw in query_lower for kw in math_keywords):
                    subject = "math"
                elif any(kw in query_lower for kw in chinese_keywords):
                    subject = "chinese"

        except Exception:
            # —— 容错：LLM 调用失败时路由到未知意图兜底 ——
            # 之前版本默认回退到 academic，但实际效果很糟糕：
            # 用户说"我好焦虑"→ LLM 崩了 → 系统当学术问题处理 →
            # 跑去 RAG 检索焦虑相关知识点 → 返回一段科普
            # 不如直接说"系统繁忙，请重试"来得诚实
            logger.warning("Supervisor structured output failed, routing to unknown")
            intent = "unknown"
            subject = "other"
            keypoints = []

    # 防御：确保返回的 intent 在有效范围内
    if intent not in _VALID_INTENTS:
        intent = "academic"

    return {"intent": intent, "subject": subject, "keypoints": keypoints}


# ============================================================================
# 兜底节点：处理与高考无关的问题
# ============================================================================

@traced_node
async def handle_unknown(state: TutorState) -> dict:
    """友好拒答非高考相关问题。

    只返回引导性文案，不调用任何 LLM（零成本）。
    引导用户回到辅导场景，而不是硬性拒绝。
    """
    return {
        "messages": [AIMessage(
            content=(
                "抱歉，我暂时无法理解你的问题，请重新描述一下。"
                "我是你的高考辅导助手，可以帮你解答学科知识、"
                "制定学习计划、或者聊聊学习中的烦恼。"
            ),
        )],
    }


# ============================================================================
# 条件路由函数
# ============================================================================

def route_by_intent(state: TutorState) -> str:
    """根据 Supervisor 分类的 intent 决定流向哪个子图。

    这是 add_conditional_edges 的回调函数，返回值必须匹配 builder.py
    中定义的路由表中某个 key。

    Returns:
        "academic"  → 学术子图（RAG + WebSearch + 幻觉检测）
        "planning"  → 规划子图（政策搜索 + 对抗性规划）
        "emotional" → 情绪支持（单节点）
        "unknown"   → 兜底拒答
    """
    return state.get("intent", "academic")
