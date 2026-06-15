"""情绪支持节点（Emotional Support）—— 班主任老师人格

这是系统四条分支中最简单的一条：一次 LLM 调用，直接结束。
不需要 RAG 检索、不需要对抗性审查、不需要人机交互。

设计理念：
- 学生来倾诉学习压力时，不需要"权威答案"，需要的是被理解
- 以经验丰富的班主任身份回应，温暖而实用
- temperature=0.8 保证回答有一定创造性，不像是模板回复

为什么没有 RAG 或联网搜索？
情绪支持不需要事实性知识。学生的核心需求是情感共鸣。
如果学生同时提到学习困境，Supervisor 应该分类到 academic/planning 分支，
而不是 emotional。

面试追问点：
- emotional 分支为什么不需要 fallback？
  因为这里只有一次 LLM 调用，且调用者有容灾（async_invoke_with_fallback）
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, SystemMessage

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.memory.context_builder import build_memory_context
from src.tracing import traced_llm_call, traced_node


@traced_node
async def emotional_response(state: TutorState) -> dict:
    """以班主任身份提供温暖、实用的情绪支持。

    完整对话历史传入 LLM，让它理解上下文和学生的情绪变化。
    不检索任何外部知识，完全依靠模型的共情能力。

    State 更新：
    - messages: 追加 AI 的情绪支持回复
    """
    llm = get_node_llm("emotional")

    # 加载长期记忆，注入系统提示词
    memory_context = build_memory_context(state)
    memory_text = f"\n\n{memory_context}" if memory_context else ""

    # 传入完整对话历史 + 长期记忆，让 LLM 理解上下文
    history = [SystemMessage(content=load_prompt("emotional_system") + memory_text)]
    for msg in state["messages"]:
        if (
            isinstance(msg, SystemMessage)
            and str(getattr(msg, "content", "")).startswith("[会话摘要]")
        ):
            continue
        history.append(msg)

    temperature = get_setting("emotional.temperature", 0.8)
    fallback = get_fallback_llm(temperature=temperature)

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="emotional_response",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, history, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}
