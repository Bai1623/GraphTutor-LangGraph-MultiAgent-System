"""TutorState: LangGraph 共享状态定义（单一数据源）

TutorState 是整个多智能体系统的"中央黑板"——所有 19 个图节点通过读写
这个 TypedDict 来协作，不依赖任何外部状态。

设计原则:
1. 单一数据源 (Single Source of Truth): 所有节点共享同一个 state
2. 增量更新: 每个节点只返回自己需要修改的字段，LangGraph 自动合并
3. 自定义 Reducer: 对于 list/dict 等复杂字段，指定合并策略（追加/覆盖/清空）
4. 扁平化: 将子图状态展平到父图，简化状态管理

面试追问点:
- context_reducer 为什么需要？因为并行分支 (RAG + Web Search)
  同时写入 context 字段，不加 reducer 会互相覆盖
- 为什么用 TypedDict 而不是 Pydantic？LangGraph 内部用 TypedDict
  做状态合并，性能更好且支持 Annotated reducer
"""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ============================================================================
# Sentinel 值：用于清空 context 字段
# ============================================================================
# 当节点返回 CONTEXT_CLEAR 时，context_reducer 会将整个 context 列表
# 重置为空列表。用于重试路径（改写查询后需要丢弃旧的检索结果）
CONTEXT_CLEAR: list[dict] = [{"__clear__": True}]


def context_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """并行分支的安全合并函数——LangGraph 的 Annotated Reducer 机制。

    **为什么需要这个函数？**
    在学术子图中，RAG 检索和 Web 搜索是两个并行分支（Fan-out），它们
    同时执行且都会向 state["context"] 写入检索结果。如果不指定 reducer，
    LangGraph 默认会用后写入的值覆盖先写入的值，导致其中一路的检索结果
    丢失。

    **工作逻辑：**
    1. 正常情况: 将新结果追加到已有结果后面（相当于 operator.add）
    2. 重试情况: 如果 update 是 CONTEXT_CLEAR 标记，清空整个列表
       ——用于查询改写后重新检索时丢弃旧的上下文

    面试追问: LangGraph 默认 reducer 是"覆盖"，对于 list 类型要用
    Annotated[list, reducer_function] 指定自定义合并逻辑。
    """
    if update and update[0].get("__clear__"):
        return []  # 重试路径：清空上下文，等新检索结果写入
    return existing + update  # 正常路径：追加合并


# ============================================================================
# TutorState — 26 个字段的共享状态
# ============================================================================

class TutorState(TypedDict):
    """高考辅导多智能体系统的统一状态对象。

    每个字段的含义及使用节点：

    ┌─────────────────────┬──────────┬────────────────────────────────┐
    │ 字段                │ 写入节点  │ 说明                           │
    ├─────────────────────┼──────────┼────────────────────────────────┤
    │ messages            │ 全部     │ 对话历史 (add_messages reducer) │
    │ intent              │supervisor│ 用户意图分类结果               │
    │ subject             │supervisor│ 学科识别 (math/chinese/other)  │
    │ keypoints           │supervisor│ 提取的关键知识点               │
    │ context             │rag/web   │ 检索到的上下文片段 (context_reducer)│
    │ context_insufficient│check_ctx │ RAG 与 Web 检索均为空      │
    │ search_results      │search_policy│ 政策搜索结果               │
    │ plan                │plan_output│ 最终生成的学习计划           │
    │ retry_count         │evaluate  │ 幻觉检测重试计数               │
    │ hallucination_detected│evaluate│ 是否检测到幻觉                 │
    │ rewritten_query     │rewrite   │ 改写后的查询文本               │
    │ hallucination_reason│evaluate  │ 幻觉评估的详细原因             │
    │ emotional_intel     │gather_intel│ 情绪状态情报               │
    │ resource_intel      │gather_intel│ 学习资源情报               │
    │ intel_summary       │gather_intel│ 合并后的情报摘要           │
    │ draft               │drafter   │ 当前学习计划草稿               │
    │ academic_verdict    │reviewer  │ 学术审查结论 (approve/reject)  │
    │ academic_reason     │reviewer  │ 学术审查理由                   │
    │ emotional_verdict   │reviewer  │ 情绪审查结论 (approve/reject)  │
    │ emotional_reason    │reviewer  │ 情绪审查理由                   │
    │ adv_round           │drafter   │ 当前对抗审查轮次               │
    │ consensus           │consensus │ 双审查员是否达成共识           │
    │ revision_notes      │consensus │ 打回重写的修改建议             │
    │ hil_action          │plan_output│ HIL 动作 (confirm/feedback)  │
    │ hil_feedback        │feedback_router│ 用户反馈文本           │
    │ hil_summary         │feedback_router│ 压缩后的反馈摘要       │
    │ feedback_route      │feedback_router│ 反馈路由 (tweak/rewrite)│
    └─────────────────────┴──────────┴────────────────────────────────┘
    """

    # ── 基础信息 ───────────────────────────────────────────────────
    # 对话历史。使用 add_messages reducer 自动处理消息追加和去重
    messages: Annotated[list, add_messages]

    # ── Supervisor 输出 ────────────────────────────────────────────
    intent: Literal["academic", "planning", "emotional", "unknown"]
    subject: str         # 主题/学科
    keypoints: list[str] # 提取的关键知识点

    # ── 学术子图 (RAG + 幻觉检测) ──────────────────────────────────
    # 检索上下文。使用 context_reducer 防止并行分支互相覆盖
    context: Annotated[list[dict], context_reducer]
    context_insufficient: bool      # RAG 和 Web 检索结果均为空（前置检查标记）
    retry_count: int                # 当前重试次数
    hallucination_detected: bool    # 是否检测到幻觉
    rewritten_query: str            # 改写后的查询（重试时使用）
    hallucination_reason: str       # 幻觉原因

    # ── 规划子图 (搜索 + 情报) ────────────────────────────────────
    search_results: list[dict]  # 政策/资源搜索结果
    emotional_intel: str        # 学生情绪状态情报
    resource_intel: str         # 学习资源情报
    intel_summary: str          # 合并情报摘要（传给 drafter）

    # ── 对抗性规划 (Drafter → 双Reviewer → Consensus → 重写循环) ─
    draft: str               # 当前计划草稿
    plan: str                # 最终确认的计划
    academic_verdict: str    # 学术审查结论 "approve" / "reject"
    academic_reason: str     # 学术审查详细理由
    emotional_verdict: str   # 情绪审查结论 "approve" / "reject"
    emotional_reason: str    # 情绪审查详细理由
    adv_round: int           # 当前审查轮次（从1开始）
    consensus: bool          # 双方是否达成共识
    revision_notes: str      # 合并的修改建议（打回时传给 drafter）

    # ── 人在回路 (HIL) 反馈 ──────────────────────────────────────
    hil_action: str      # plan_output 设置: "confirm" / "feedback"
    hil_feedback: str    # 用户原始反馈文本
    hil_summary: str     # 多轮反馈的压缩摘要（覆盖式，不追加）
    feedback_route: str  # 反馈路由器分类结果: "tweak" / "rewrite"

    # ── 长期记忆 ────────────────────────────────────────────
    # 跨会话持久化的用户关键信息（如年级、弱项、偏好等）
    # supervisor 启动时从 MemoryStore 加载，注入系统提示词
    # 格式：[关于该用户的记忆]\n- 事实1\n- 事实2\n
    long_term_memory: str
