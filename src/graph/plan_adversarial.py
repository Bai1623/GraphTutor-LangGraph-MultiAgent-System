"""对抗性规划节点集合 (Adversarial Planning Nodes)

核心思想：借鉴 GAN（生成对抗网络）的思路，用多个 Agent 互相博弈
来提高输出质量。一个 Drafter（起草者）生成计划，两个 Reviewer（审查员）
从不同维度审查，必须双方都同意才输出。

完整流程：
  Drafter → [Reviewer Academic ∥ Reviewer Emotional] → Consensus Check
       ↑                                                    ↓
       └────────── 打回重写 (adv_rewrite) ←────────────── 不通过
                                                           ↓
                                                  plan_output (HIL中断)
                                                           ↓
                                              用户反馈 → feedback_router
                                                    ├─ tweak → plan_tweak
                                                    └─ rewrite → drafter

面试追问点：
- 为什么两个 Reviewer 是并行而不是串行？
  答：两个维度（学术质量 vs 情绪关怀）互不依赖，并行可减少延迟约 40%
- 为什么不让一个 LLM 同时评估两个维度？
  答：角色分离强制 LLM 从不同视角思考，避免"确认偏误"
- 安全阀 (max_rounds) 的作用？
  答：防止死循环无限消耗 Token，强制收敛保证系统可用性
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


# ============================================================================
# 结构化输出模型 (Pydantic — 确保 LLM 返回可解析的 JSON)
# ============================================================================

class ReviewVerdict(BaseModel):
    """审查员的结构化评审结论。

    使用 Pydantic BaseModel + with_structured_output() 确保 LLM
    返回的 JSON 被可靠解析，避免字符串截断或格式错误。
    """
    verdict: Literal["approve", "reject"]   # 审批结论：通过或驳回
    reason: str                              # 详细理由（打回时必须填写）


class FeedbackClassification(BaseModel):
    """用户反馈的结构化分类结果。

    根据反馈内容判断是"局部微调"还是"根本性重写"。
    微调走快速通道（单次 LLM 调用），重写走完整对抗循环。
    """
    route: Literal["tweak", "rewrite"]  # 路由方向
    reason: str                          # 分类理由


# ============================================================================
# 工具函数
# ============================================================================

def _last_human_query(state: TutorState) -> str:
    """从对话历史中提取最后一条用户消息。

    遍历 messages 列表（倒序），找到最近的 HumanMessage。
    用于在 plan 分支的各个节点中获取用户的原始问题文本。
    """
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ============================================================================
# 节点 1: Drafter（计划起草者）
# ============================================================================

@traced_node
async def drafter_node(state: TutorState) -> dict[str, Any]:
    """起草或改写学习计划。

    根据 state 中是否有 revision_notes（修改建议）判断是首次起草还是打回重写：
    - 首次起草：使用 plan_drafter 提示词，基于情报摘要从头生成
    - 打回重写：使用 plan_rewrite 提示词，结合原始草稿和修改建议
    """
    llm = get_node_llm("planner")
    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)

    user_request = _last_human_query(state)
    intel_summary = state.get("intel_summary", "")
    revision_notes = state.get("revision_notes", "")

    if revision_notes:
        # —— 重写路径：审查员或用户要求修改 ——
        prompt_text = load_prompt("plan_rewrite").format(
            user_request=user_request,
            intel_summary=intel_summary,
            current_draft=state.get("draft", ""),
            revision_notes=revision_notes,
        )
    else:
        # —— 首次起草路径 ——
        prompt_text = load_prompt("plan_drafter").format(
            user_request=user_request,
            intel_summary=intel_summary,
        )

    messages = [
        SystemMessage(content=load_prompt("plan_drafter_system")),
        HumanMessage(content=prompt_text),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="drafter_node",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {
        "draft": response.content,               # 新草稿
        "adv_round": state.get("adv_round", 0) + 1,  # 轮次 +1
    }


# ============================================================================
# 审查员共享逻辑 (DRY — 学术审查员和情绪审查员用同一个模板函数)
# ============================================================================

async def _run_reviewer(
    state: TutorState,
    *,
    system_prompt_name: str,
    node_name: str,
) -> ReviewVerdict:
    """审查员通用逻辑——学术和情绪审查共用。

    面试可以讲的设计决策:
    - temperature=0.0：审查要求确定性，不允许随机性
    - with_structured_output：用 Pydantic 约束输出格式，避免解析失败
    - 异常时默认 approve：防止审查员崩溃阻塞整个流程（乐观策略）
    """
    reviewer_temp = get_setting("planner.reviewer_temperature", 0.0)
    llm = get_node_llm("planner", temperature=reviewer_temp)
    structured_primary = llm.with_structured_output(ReviewVerdict, method="json_mode")

    # 容灾链：primary → fallback 也配置 structured output
    fallback_llm = get_fallback_llm(temperature=reviewer_temp)
    structured_fallback = fallback_llm.with_structured_output(ReviewVerdict, method="json_mode")

    review_prompt = (
        f"## 学习计划\n\n{state.get('draft', '')}\n\n"
        f"## 学生情况\n\n{state.get('intel_summary', '')}\n\n"
        f"请以 json 格式返回你的审查结论。"
    )
    messages = [
        SystemMessage(content=load_prompt(system_prompt_name)),
        HumanMessage(content=review_prompt),
    ]

    with traced_llm_call(
        model_name=get_setting("planner.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        node_name=node_name,
        temperature=reviewer_temp,
    ) as span:
        try:
            verdict = await async_invoke_with_fallback(
                structured_primary, messages,
                fallback=structured_fallback, span=span,
            )
            return verdict
        except Exception:
            # —— 容错策略：审查员异常时默认通过 ——
            # 防止审查员崩溃阻塞整个 pipeline
            logger.warning("Reviewer %s failed, defaulting to approve", node_name, exc_info=True)
            return ReviewVerdict(verdict="approve", reason="审查异常，默认通过")


# ============================================================================
# 节点 2 & 3: 两名审查员（并行执行）
# ============================================================================

@traced_node
async def reviewer_academic_node(state: TutorState) -> dict[str, Any]:
    """学术质量审查员。

    评估维度：
    - 计划是否符合高考备考规律
    - 学科安排是否合理（数学要多练，语文要积累）
    - 时间分配是否科学
    """
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_academic_system",
        node_name="reviewer_academic",
    )
    return {"academic_verdict": verdict.verdict, "academic_reason": verdict.reason}


@traced_node
async def reviewer_emotional_node(state: TutorState) -> dict[str, Any]:
    """情绪关怀审查员。

    评估维度：
    - 计划是否过于激进（易导致挫败感）
    - 是否留有休息和弹性时间
    - 语言是否鼓励性、是否考虑了学生的心理状态
    """
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_emotional_system",
        node_name="reviewer_emotional",
    )
    return {"emotional_verdict": verdict.verdict, "emotional_reason": verdict.reason}


# ============================================================================
# 节点 4: Consensus Check（共识检查）
# ============================================================================

@traced_node
async def consensus_check_node(state: TutorState) -> dict[str, Any]:
    """检查双审查员是否达成共识。

    决策逻辑：
    1. 双方 approve → 通过，输出计划
    2. 任一方 reject → 收集驳回理由，触发重写
    3. 达到最大轮次 (adversarial_max_rounds) → 安全阀强制通过
       ——宁可输出一个不完美的计划，也不能让系统死循环

    面试追问：安全阀阈值设为多少？
    默认 3 轮。每轮约消耗 10K Token（drafter 2K + 2×reviewer 各 4K）。
    3 轮 = 约 30K Token ≈ ¥0.06（DeepSeek），对用户体验影响可控。
    """
    current_round = state.get("adv_round", 0)
    max_rounds = get_setting("planner.adversarial_max_rounds", 3)
    academic = state.get("academic_verdict", "")
    emotional = state.get("emotional_verdict", "")

    both_approve = academic == "approve" and emotional == "approve"

    # ── 安全阀：达到最大轮次强制输出 ──
    if current_round >= max_rounds:
        if not both_approve:
            logger.warning(
                "Max rounds (%d) reached with unresolved rejections, forcing output",
                max_rounds,
            )
        return {"consensus": True, "revision_notes": ""}

    # ── 双方通过 ──
    if both_approve:
        return {"consensus": True, "revision_notes": ""}

    # ── 有驳回：收集理由供 drafter 修改 ──
    notes_parts: list[str] = []
    if academic == "reject":
        reason = state.get("academic_reason", "未提供原因")
        notes_parts.append(f"[学术审查] {reason}")
    if emotional == "reject":
        reason = state.get("emotional_reason", "未提供原因")
        notes_parts.append(f"[情绪审查] {reason}")

    return {
        "consensus": False,
        "revision_notes": "; ".join(notes_parts) if notes_parts else "需要修改",
    }


# ============================================================================
# 节点 5: 打回重写前的状态清理
# ============================================================================

@traced_node
async def adv_rewrite_node(state: TutorState) -> dict[str, Any]:
    """进入新一轮起草前，清空上一轮的审查结果。

    不清空会导致旧 verdict 残留在 state 中，共识检查会用到脏数据。
    这是一个"纯状态清理"节点——不调用任何 LLM。
    """
    return {
        "academic_verdict": "",
        "academic_reason": "",
        "emotional_verdict": "",
        "emotional_reason": "",
    }


# ============================================================================
# 节点 6: Plan Output（计划输出 + HIL 中断）
# ============================================================================

@traced_node
async def plan_output_node(state: TutorState) -> dict:
    """输出最终计划并通过 LangGraph interrupt() 暂停执行。

    这是 Human-in-the-Loop 的核心节点：
    1. 调用 interrupt(plan_text) 暂停图表执行
    2. 前端收到 interrupt 事件后展示可编辑的计划
    3. 用户确认 → 返回字符串 → 结束
    4. 用户反馈 → 返回 dict{"action": "feedback"} → 进入反馈路由

    面试追问：interrupt() 的状态保存在哪里？
    PostgreSQL checkpointer。图表的状态快照通过 langgraph-checkpoint-postgres
    持久化到数据库，支持跨进程恢复。没有 checkpointer 时跳过 HIL。
    """
    plan_text = state.get("draft", "")

    # HIL: 暂停图表，等待用户审批
    try:
        user_response = interrupt(plan_text)
    except ValueError:
        # 没有 checkpointer → 跳过 HIL，直接用草稿
        logger.warning("interrupt() failed (no checkpointer?), skipping HIL review")
        user_response = plan_text

    # ── 用户反馈 (dict) → 进入反馈路由 ──
    if isinstance(user_response, dict) and user_response.get("action") == "feedback":
        return {
            "hil_action": "feedback",
            "hil_feedback": user_response.get("text", ""),
        }

    # ── 用户确认 (string) → 最终输出 ──
    final_plan = user_response if isinstance(user_response, str) and user_response else plan_text
    return {
        "plan": final_plan,
        "messages": [AIMessage(content=final_plan)],
        "hil_action": "confirm",
    }


# ============================================================================
# 节点 7: Feedback Router（反馈分类器）
# ============================================================================

@traced_node
async def feedback_router(state: TutorState) -> dict[str, Any]:
    """将用户反馈分类为 tweak（微调）或 rewrite（重写）。

    为什么要分类？
    - 用户说"把数学从周二换到周三" → 微调，单次 LLM 调用即可
    - 用户说"整个计划方向不对，我更喜欢刷题为主" → 重写，走完整对抗循环

    成本和延迟差异：
    - tweak: 1 次 LLM 调用，~2s，~2K Token
    - rewrite: 1-N 轮对抗循环，~10-30s，~10-50K Token

    使用 Supervisor 的模型 (Qwen2.5-7B) 做轻量分类，避免消耗主力模型配额。
    """
    llm = get_node_llm("supervisor")
    structured_llm = llm.with_structured_output(FeedbackClassification, method="json_mode")

    feedback = state.get("hil_feedback", "")
    draft = state.get("draft", "")
    old_summary = state.get("hil_summary", "")

    # ── 步骤 1: 分类反馈 ──
    classify_prompt = (
        f"学生对以下学习计划提出了修改意见。\n\n"
        f"## 当前计划（前500字）\n{draft[:500]}\n\n"
        f"## 学生反馈\n{feedback}\n\n"
        f"判断这个反馈需要的修改程度：\n"
        f"- tweak: 只需要局部微调（如调整某天科目、修改时间、增删某个小项）\n"
        f"- rewrite: 需要重新规划（如整体思路不对、完全不符合需求、需要换方向）\n\n"
        f"请以 json 格式返回你的分类结果。"
    )

    try:
        result = await structured_llm.ainvoke([
            SystemMessage(content="你是一个学习计划修改分类器。根据学生反馈判断需要微调还是重写。"),
            HumanMessage(content=classify_prompt),
        ])
        route = result.route
    except Exception:
        logger.warning("Feedback classification failed, defaulting to tweak")
        route = "tweak"

    # ── 步骤 2: 更新反馈摘要（覆盖式，防膨胀）──
    # 只保留压缩摘要而非追加完整历史，避免上下文无限增长
    if old_summary:
        new_summary = f"历史修改摘要: {old_summary[:200]}\n最新反馈: {feedback[:500]}"
    else:
        new_summary = f"用户反馈: {feedback[:500]}"

    if route == "rewrite":
        # 重写路径：清空所有对抗状态，当作新需求处理
        return {
            "feedback_route": "rewrite",
            "hil_summary": new_summary,
            "revision_notes": feedback,
            "adv_round": 0,        # 重置审查轮次
            "draft": "",            # 清空旧草稿
            "academic_verdict": "",
            "academic_reason": "",
            "emotional_verdict": "",
            "emotional_reason": "",
            "consensus": False,
        }
    else:
        # 微调路径：保留现有草稿，只传反馈给 plan_tweak
        return {
            "feedback_route": "tweak",
            "hil_summary": new_summary,
        }


# ============================================================================
# 节点 8: Plan Tweak（计划微调）
# ============================================================================

@traced_node
async def plan_tweak_node(state: TutorState) -> dict[str, Any]:
    """基于用户反馈对计划进行局部微调。

    与 drafter 的区别：
    - drafter: 从头生成或大幅重写（需要对抗审查）
    - plan_tweak: 只修改用户提及的部分（单次调用，不经审查）

    为什么微调不需要审查？
    微调只改变局部（如交换两天的科目），不会引入结构性风险。
    如果微调出错，用户可以再次反馈 —— 多轮微调而非一轮完美。
    """
    llm = get_node_llm("planner")
    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)

    draft = state.get("draft", "")
    feedback = state.get("hil_feedback", "")
    summary = state.get("hil_summary", "")

    prompt = (
        f"请根据学生的反馈对以下学习计划进行**局部微调**。\n"
        f"只修改学生提到的部分，保持其他内容不变。\n\n"
        f"## 当前计划\n{draft}\n\n"
        f"## 学生反馈\n{feedback}\n\n"
    )
    if summary:
        prompt += f"## 修改历史摘要\n{summary}\n\n"
    prompt += "请输出修改后的完整计划："

    messages = [
        SystemMessage(content=load_prompt("plan_drafter_system")),
        HumanMessage(content=prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="plan_tweak",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"draft": response.content}


# ============================================================================
# 条件路由函数（供 builder.py 中的 add_conditional_edges 使用）
# ============================================================================

def should_output_or_revise(state: TutorState) -> str:
    """共识检查后的路由：输出 or 打回重写。

    Returns:
        "output" — 共识达成或达到最大轮次
        "revise" — 需要新一轮起草
    """
    if state.get("consensus", False):
        return "output"
    return "revise"


def route_after_hil(state: TutorState) -> str:
    """HIL 中断后的路由：确认结束 or 进入反馈流程。

    Returns:
        "end" — 用户确认了计划
        "feedback" — 用户提交了反馈意见
    """
    return "feedback" if state.get("hil_action") == "feedback" else "end"


def route_feedback(state: TutorState) -> str:
    """反馈分类后的路由：微调 or 重写。

    Returns:
        "tweak" — 局部修改（快速通道）
        "rewrite" — 完整重写（对抗审查循环）
    """
    return state.get("feedback_route", "tweak")
