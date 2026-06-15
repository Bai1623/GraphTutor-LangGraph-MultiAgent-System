"""LangGraph StateGraph 构建器——组装 19 个节点、条件边、并行与循环。

本文件是整个多智能体系统的"调度中心"：定义所有节点、配置边（普通边、
条件边、并行 Fan-out/Fan-in），并编译成可执行的 StateGraph。

面试中描述图结构时可以按以下顺序讲解：
1. 入口 → Supervisor 意图分类
2. 四条分支：学术（含并行检索+幻觉检测）、规划（含对抗循环+HIL）、
   情绪（单节点）、兜底
3. 横切关注点：checkpointer（状态持久化）、tracing（全链路追踪）

图拓扑概览（20 节点）：
==============================
Supervisor (入口)
├─ academic_router → [rag_retrieve ∥ web_search]
│                          ↓
│                   check_context_sufficiency
│                     ├─ insufficient → END (诚实引导，零LLM)
│                     └─ generate_answer
│                          ↓
│     ┌── retry ← rewrite_query ← evaluate_hallucination
│     └── END
├─ search_policy → gather_intel → drafter
│                                   ↓
│     ┌── revise ←─ consensus_check ← [reviewer_academic ∥ reviewer_emotional]
│     └── plan_output → (HIL interrupt) → feedback_router
│                                              ├─ tweak → plan_tweak → plan_output
│                                              └─ rewrite → drafter
├─ emotional_response → END
└─ handle_unknown → END
==============================
"""

from __future__ import annotations

from langchain_core.messages import RemoveMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.graph.academic import (
    academic_router,
    check_context_sufficiency,
    evaluate_hallucination,
    generate_answer,
    rag_retrieve,
    rewrite_query,
    route_after_check,
    should_retry_or_end,
    web_search,
)
from src.graph.emotional import emotional_response
from src.graph.plan_adversarial import (
    adv_rewrite_node,
    consensus_check_node,
    drafter_node,
    feedback_router,
    plan_output_node,
    plan_tweak_node,
    reviewer_academic_node,
    reviewer_emotional_node,
    route_after_hil,
    route_feedback,
    should_output_or_revise,
)
from src.graph.planner import gather_intel, search_policy
from src.graph.state import TutorState
from src.graph.supervisor import handle_unknown, route_by_intent, supervisor_node
from src.tracing import traced_node


@traced_node
async def compress_messages(state: TutorState) -> dict:
    """三层压缩节点——在 supervisor 之前自动触发。

    每轮对话前检查消息数量，超出窗口时触发压缩：
    Layer 1 → 最近8轮完整保留
    Layer 2 → 更早内容压缩为 session_summary
    Layer 3 → 长期事实由 MemoryStore 管理（不在此节点处理）
    """
    try:
        from src.memory.compressor import compress_conversation
        messages = state.get("messages", [])
        old_summary = state.get("session_summary", "")
        result = await compress_conversation(
            list(messages),
            existing_summary=old_summary,
        )
        if not result.compressed:
            return {}
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *result.messages,
            ],
            "session_summary": result.summary_json,
            "compression_count": state.get("compression_count", 0) + 1,
            "compression_before_tokens": result.before_tokens,
            "compression_after_tokens": result.after_tokens,
        }
    except Exception:
        import logging
        logging.getLogger(__name__).warning("Message compression failed, using full history", exc_info=True)
        return {}


def build_graph() -> StateGraph:
    """构建完整的 LangGraph StateGraph（未编译）。

    返回未编译的 StateGraph 对象，调用方可以通过 .compile(checkpointer=...)
    生成可执行的 CompiledStateGraph。

    注意事项：
    - add_node 只注册节点，不定义执行顺序
    - add_edge 定义确定性连接（单输入单输出）
    - add_conditional_edges 定义条件分支（根据 state 动态路由）
    - 同一节点的多个 add_edge 自动形成 Fan-out（并行出边）
    - 多个节点指向同一个节点自动形成 Fan-in（汇聚）
    """

    # ── 步骤 1: 创建 StateGraph，绑定 TutorState 类型 ──────────────
    graph = StateGraph(TutorState)

    # ── 步骤 2: 注册全部 19 个节点 ─────────────────────────────────
    # 节点只是"注册"，执行顺序由边决定

    # 记忆压缩节点（入口——在 supervisor 之前触发）
    graph.add_node("compress_messages", compress_messages)

    # 入口节点：意图分类
    graph.add_node("supervisor", supervisor_node)

    # 子图 A：学术问答（并行检索 + 前置检查 + 幻觉检测重试）
    graph.add_node("academic_router", academic_router)      # 学术路由（提取查询意图）
    graph.add_node("rag_retrieve", rag_retrieve)            # 本地 RAG 检索
    graph.add_node("web_search", web_search)                # DuckDuckGo 联网搜索
    graph.add_node("check_context_sufficiency", check_context_sufficiency)  # 检索充分性前置检查
    graph.add_node("generate_answer", generate_answer)      # 融合上下文生成回答
    graph.add_node("evaluate_hallucination", evaluate_hallucination)  # 幻觉检测
    graph.add_node("rewrite_query", rewrite_query)          # 查询改写（重试时触发）

    # 子图 B：学习规划（对抗性起草 + HIL 人工审批）
    graph.add_node("search_policy", search_policy)          # 搜索高考政策
    graph.add_node("gather_intel", gather_intel)            # 收集学生情报（情绪+资源）
    graph.add_node("drafter", drafter_node)                 # 起草学习计划
    graph.add_node("reviewer_academic", reviewer_academic_node)    # 学术质量审查
    graph.add_node("reviewer_emotional", reviewer_emotional_node)  # 情绪关怀审查
    graph.add_node("consensus_check", consensus_check_node) # 共识检查（双方通过才放行）
    graph.add_node("adv_rewrite", adv_rewrite_node)         # 打回重写前清空旧审查结果
    graph.add_node("plan_output", plan_output_node)         # 计划输出 + HIL 中断
    graph.add_node("feedback_router", feedback_router)      # 用户反馈分类（微调/重写）
    graph.add_node("plan_tweak", plan_tweak_node)           # 基于反馈的局部微调

    # 情绪支持（单节点，无子图）
    graph.add_node("emotional_response", emotional_response)

    # 未知意图兜底
    graph.add_node("handle_unknown", handle_unknown)

    # ── 步骤 3: 定义边（图的拓扑结构）────────────────────────────

    # 3.1 设置入口节点——先压缩再分类
    graph.set_entry_point("compress_messages")
    graph.add_edge("compress_messages", "supervisor")

    # 3.2 Supervisor → 四条分支（条件路由）
    #     根据 supervisor_node 分类的 intent 字段，分发到不同子图
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,  # 条件函数：返回 "academic" / "planning" / "emotional" / "unknown"
        {
            "academic": "academic_router",
            "planning": "search_policy",
            "emotional": "emotional_response",
            "unknown": "handle_unknown",
        },
    )

    # ═══════════════════════════════════════════════════════════════
    # 子图 A：学术问答流（含并行检索 + 前置检查 + 幻觉检测重试循环）
    # ═══════════════════════════════════════════════════════════════

    # 3.3 Fan-out（并行出边）：academic_router 同时触发 RAG 和 Web
    #     两条边指向不同节点 = LangGraph 自动并行执行
    graph.add_edge("academic_router", "rag_retrieve")
    graph.add_edge("academic_router", "web_search")

    # 3.4 Fan-in（汇聚）：两条检索路径都完成后再进入前置检查
    #     两条边指向同一节点 = LangGraph 等待两者都完成后才执行
    graph.add_edge("rag_retrieve", "check_context_sufficiency")
    graph.add_edge("web_search", "check_context_sufficiency")

    # 3.5 检索充分性检查 → 生成回答 or 直接结束
    graph.add_conditional_edges(
        "check_context_sufficiency",
        route_after_check,
        {
            "continue": "generate_answer",  # 有检索结果：正常生成
            "end": END,                      # 都为空：已返回诚实引导
        },
    )

    # 3.6 幻觉评估 → 重试循环或结束
    graph.add_edge("generate_answer", "evaluate_hallucination")
    graph.add_conditional_edges(
        "evaluate_hallucination",
        should_retry_or_end,  # 条件函数：返回 "retry" 或 "end"
        {
            "retry": "rewrite_query",  # 有幻觉：改写查询重新检索
            "end": END,                # 无幻觉或超过最大重试：结束
        },
    )
    # 重试回路：rewrite_query → academic_router（走回检索+生成流程）
    graph.add_edge("rewrite_query", "academic_router")

    # ═══════════════════════════════════════════════════════════════
    # 子图 B：学习规划流（含对抗性审查循环 + HIL 人机交互）
    # ═══════════════════════════════════════════════════════════════

    # 3.6 规划准备：政策搜索 → 情报收集 → 起草计划
    graph.add_edge("search_policy", "gather_intel")
    graph.add_edge("gather_intel", "drafter")

    # 3.7 对抗性审查：Drafter → 并行双审 → 共识检查
    #     Fan-out: drafter 同时触发学术审查员和情绪审查员
    graph.add_edge("drafter", "reviewer_academic")
    graph.add_edge("drafter", "reviewer_emotional")
    #     Fan-in: 两个审查员都完成后进入共识检查
    graph.add_edge("reviewer_academic", "consensus_check")
    graph.add_edge("reviewer_emotional", "consensus_check")

    # 3.8 共识检查 → 通过输出 or 打回重写
    graph.add_conditional_edges(
        "consensus_check",
        should_output_or_revise,  # 条件函数：返回 "output" 或 "revise"
        {
            "output": "plan_output",  # 通过：输出计划
            "revise": "adv_rewrite",  # 不通过：清空审查结果后重写
        },
    )
    # 打回回路：adv_rewrite → drafter（重新起草，保留 revision_notes）
    graph.add_edge("adv_rewrite", "drafter")

    # 3.9 HIL 人机交互：plan_output 中断 → 用户反馈 → 分类处理
    graph.add_conditional_edges(
        "plan_output",
        route_after_hil,  # 条件函数：返回 "end" 或 "feedback"
        {
            "end": END,               # 用户确认：结束
            "feedback": "feedback_router",  # 用户有意见：进入反馈路由
        },
    )

    # 3.10 反馈路由器 → 微调 or 重写
    graph.add_conditional_edges(
        "feedback_router",
        route_feedback,  # 条件函数：返回 "tweak" 或 "rewrite"
        {
            "tweak": "plan_tweak",     # 局部修改：走微调节点
            "rewrite": "drafter",      # 大改：回到起草（重置审查状态）
        },
    )
    # 微调回路：plan_tweak → plan_output（微调后再次展示给用户）
    graph.add_edge("plan_tweak", "plan_output")

    # ═══════════════════════════════════════════════════════════════
    # 情绪分支 & 兜底分支（直接结束）
    # ═══════════════════════════════════════════════════════════════

    graph.add_edge("emotional_response", END)
    graph.add_edge("handle_unknown", END)

    return graph


def get_compiled_graph(checkpointer=None):
    """构建并编译图，可直接用于调用。

    面试补充说明:
    - checkpointer 为 None 时，系统降级为无状态模式（无 PostgreSQL）
    - 无状态模式下无法使用 interrupt()（HIL），但其他功能正常
    - 有 checkpointer 时，支持多轮对话记忆 + HIL 中断恢复

    Args:
        checkpointer: LangGraph Checkpointer 实例（如 AsyncPostgresSaver）。
                      为 None 时运行在无状态模式。

    Returns:
        编译后的 CompiledStateGraph，可调用 .ainvoke() / .astream_events()
    """
    return build_graph().compile(checkpointer=checkpointer)
