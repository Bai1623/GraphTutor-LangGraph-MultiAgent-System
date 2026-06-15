"""子图 A：学科问答（Academic Tutor）

这是四条分支中最复杂的一条，包含：
1. 并行检索（Fan-out/Fan-in）：RAG + Web 同时查询，结果自动合并
2. 答案生成：基于合并上下文 + LLM 生成最终回答
3. 幻觉评估：判断回答是否基于实际检索内容
4. 条件重试循环：检测到幻觉 → 改写查询 → 重新检索 → 重新生成

完整数据流：
  academic_router → [rag_retrieve ∥ web_search]
         ↑                ↓              ↓
         │            generate_answer (Fan-in 汇聚)
         │                ↓
         │         evaluate_hallucination
         │              ↓           ↓
         └── rewrite_query ←── 检测到幻觉
                             ↓
                       未检测到 → END

面试追问点：
- 为什么 Fan-out/Fan-in 不需要显式 sync？
  LangGraph 自动管理：add_edge(A, C) + add_edge(B, C) = 等待 A 和 B 都完成
- 幻觉检测为什么不直接用 LangChain 的 HallucinationChecker？
  自建评估更灵活，可以用中文提示词适配高考场景，且支持容灾降级
- 重试上限为什么是 2 次？
  每次重试增加 ~5s 延迟和 ~3K Token。2 次在延迟和可靠性之间平衡
"""

from __future__ import annotations

import asyncio
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import CONTEXT_CLEAR, TutorState
from src.memory.context_builder import build_memory_context
from src.rag.retriever import retrieve
from src.tools.agent_tools import search_knowledge_base, search_web
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_retrieval, traced_search

logger = logging.getLogger(__name__)

# 最大重试次数——从 YAML 配置读取，默认 2
MAX_RETRIES = get_setting("academic.max_retries", 2)


# ============================================================================
# 结构化输出：幻觉评估
# ============================================================================

class HallucinationEvaluation(BaseModel):
    """LLM 驱动的答案忠实度评估结果。

    用于 evaluate_hallucination 节点的结构化输出。
    is_faithful=False 时触发查询改写 + 重新检索。

    为什么不用简单的相似度判断？
    语义相似度高 ≠ 没有幻觉。例如：
    - 检索内容："三角函数 sin²θ + cos²θ = 1"
    - AI 回答："此外，tan²θ + 1 = sec²θ" ← 正确但不在检索结果中
    需要 LLM 判断是否"超出检索内容的合理推断"vs"凭空捏造"
    """
    is_faithful: bool = Field(
        description="True 表示回答基于检索上下文，没有编造事实"
    )
    reason: str = Field(
        description="评估的简要说明（如果判定为幻觉，需详细解释原因）"
    )


# ============================================================================
# 工具函数
# ============================================================================

def _last_human_query(state: TutorState) -> str:
    """从对话历史中提取最近一条用户消息。

    对于重试循环特别重要：state["messages"] 中可能包含多轮
    对话，但我们需要的是最初的问题（或改写后的问题）。
    倒序遍历找到最近的 HumanMessage。
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ============================================================================
# 节点 0: Academic Router（并行检索的触发点）
# ============================================================================

@traced_node
async def academic_router(state: TutorState) -> dict:
    """学术路由节点——决定是首次检索还是重试检索。

    这个节点本身不做任何业务逻辑，只做状态管理：
    - 首次执行：retry_count=0 → 返回空字典，直接进入并行检索
    - 重试执行：retry_count>0 → 清空旧检索结果（CONTEXT_CLEAR），
      因为旧上下文导致了幻觉，需要全新检索

    为什么需要清空？
    state["context"] 使用 context_reducer（追加合并），
    如果不先清空，新检索结果会追加在"导致幻觉的旧结果"后面。
    """
    if state.get("retry_count", 0) > 0:
        # 重试路径：清空旧的检索上下文
        return {"context": CONTEXT_CLEAR}
    # 首次执行：不做任何修改
    return {}


# ============================================================================
# 节点 0b: Query Rewrite（查询改写——仅在重试时触发）
# ============================================================================

@traced_node
async def rewrite_query(state: TutorState) -> dict:
    """根据幻觉评估的反馈改写用户查询。

    为什么要改写而不是直接重试？
    原始查询可能本身就是模糊的（如"函数怎么学"），如果直接重试，
    RAG 可能返回同样的不相关结果。改写的目的是让查询更精确、
    更具体，从而获取更相关的上下文。

    使用 Superisor 模型（Qwen2.5-7B）做改写，不消耗主力模型配额。
    失败时回退到原始查询，不阻塞主流程。
    """
    original_query = _last_human_query(state)
    reason = state.get("hallucination_reason", "")

    llm = get_node_llm("supervisor")
    rewrite_prompt = load_prompt("rewrite_query").format(
        original_query=original_query,
        hallucination_reason=reason,
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content="你是一个查询改写助手。根据反馈改进用户的搜索查询。"),
            HumanMessage(content=rewrite_prompt),
        ])
        rewritten = response.content.strip()
    except Exception:
        # —— 容错：改写失败时使用原始查询 ——
        logger.warning("Query rewrite failed, using original query")
        rewritten = original_query

    return {"rewritten_query": rewritten}


# ============================================================================
# 节点 1: RAG 检索（并行分支 A——本地知识库）
# ============================================================================

@traced_node
async def rag_retrieve(state: TutorState) -> dict:
    """从本地 ChromaDB 知识库检索相关内容。

    查询策略（优先级从高到低）：
    1. rewritten_query：如果有改写后的查询（重试路径），优先使用
    2. keypoints：Supervisor 提取的关键知识点（首次查询）
    3. 原始用户消息：兜底

    使用 asyncio.to_thread 将同步检索逻辑放到线程池执行，
    避免阻塞事件循环。

    检索结果通过 traced_retrieval 记录到 OpenTelemetry，
    包含检索数量、命中状态和最高相关度分数。
    """
    rewritten = state.get("rewritten_query", "")
    keypoints = state.get("keypoints", [])
    subject = state.get("subject")

    # —— 构建最优查询 ——
    if rewritten:
        query = rewritten
    elif keypoints:
        query = " ".join(keypoints)
    else:
        query = _last_human_query(state)

    # subject 过滤器：缩小检索范围（如只查语文或数学相关文档）
    subj = subject if subject != "other" else None

    with traced_retrieval(query=query, subject=subj) as span:
        result = await asyncio.to_thread(retrieve, query=query, subject=subj)
        span.set_attribute("rag.doc_count", len(result.get("docs", [])))
        span.set_attribute("rag.is_hit", result.get("is_hit", False))
        if result.get("docs"):
            span.set_attribute("rag.top_score", result["docs"][0].get("score", 0))

    docs = result["docs"]
    # 标记来源类型，方便后续 generate_answer 区分 RAG 和 Web 结果
    return {"context": [{"type": "rag", **doc} for doc in docs]}


# ============================================================================
# 节点 2: Web 搜索（并行分支 B——联网搜索）
# ============================================================================

_SEARCH_TIMEOUT = get_setting("academic.search_timeout", 15)  # 搜索超时（秒）


@traced_node
async def web_search(state: TutorState) -> dict:
    """通过 DuckDuckGo 搜索网络内容，与 RAG 检索并行执行。

    为什么需要联网搜索？
    - 高考政策可能每年变化（如2025年新高考改革）
    - 实时新闻、录取分数线等信息本地知识库不可能全覆盖
    - 为学生提供课本之外的拓展视角

    超时保护：asyncio.wait_for 限制搜索时间。
    超时或异常时返回空列表，不阻塞主流程（RAG 结果仍然可用）。
    """
    rewritten = state.get("rewritten_query", "")
    query = rewritten if rewritten else _last_human_query(state)

    with traced_search(query=query, timeout=_SEARCH_TIMEOUT) as span:
        try:
            search_results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
                timeout=_SEARCH_TIMEOUT,
            )
            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute("search.timed_out", False)
        except asyncio.TimeoutError:
            # —— 超时：返回空结果，不阻塞主流程 ——
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", True)
        except Exception:
            # —— 其他异常：静默处理 ——
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", False)

    # 同样标记来源类型
    return {"context": [{"type": "web", **r} for r in search_results]}


# ============================================================================
# 节点 2b: Context Sufficiency Check（检索充分性前置检查——2026-06-04 新增）
# ============================================================================

@traced_node
async def check_context_sufficiency(state: TutorState) -> dict:
    """在进入 LLM 生成之前，检查 RAG 和 Web 检索是否均无结果。

    为什么需要这个节点？
    generate_answer 的 system prompt 要求 LLM"结合参考资料给出详尽解答"，
    但当两边检索都为空时，LLM 被置于一个矛盾境地：
    — 遵守 prompt → 需要编造内容（幻觉）
    — 诚实 → 违反"详尽解答"的指令

    这个前置检查的根本目的不是拦截幻觉，而是**不让 LLM 面对这个矛盾**。
    当信息客观上不存在时（如"2025 年高考作文题目"在考前被提问）：
    1. 不调用主力模型（generate_answer 用的 DeepSeek）
    2. 改用约束型 LLM 调用（Qwen2.5-7B + temperature=0.0 + 专用 prompt）
       — 生成诚实引导回答（说明原因 + 给出替代方向）

    为什么这里又调用了 LLM？之前不是说零 LLM 吗？
    硬编码模板无法给出查询特定的「为什么不可得」——用户问"2025 作文题"
    和问"同桌叫什么名字"，信息不可得的原因完全不同。需要 LLM 理解查询
    内容才能写出有针对性的诚实回复。

    但这里的 LLM 调用和 generate_answer 有本质区别：
    — generate_answer: 「结合参考资料给出详尽解答」→ 空 context 时矛盾
    — honest_response: 「信息不存在，请诚实说明原因」→ 任务本身就是诚实
    LLM 不需要编造——它的工作是解释「为什么回答不了」，不是「强行回答」。

    安全措施：
    - temperature=0.0：确定性输出，不"发挥"
    - 用 Supervisor 模型（Qwen2.5-7B）：便宜、快、与 generate_answer 隔离
    - LLM 异常时回退到硬编码兜底模板：保证可用性

    与 is_hit 的关系（面试可能追问）：
    is_hit 判断的是「检索到的文档和 query 相关吗」（相似度阈值），
    本节点判断的是「检索结果为空吗」（空/非空）。两者是正交维度：
    — is_hit=False + 空结果 → 本节点拦截
    — is_hit=False + 有结果（质量差）→ 本节点放行，靠幻觉评估兜底
    """
    context = state.get("context", [])

    # 按来源类型分离
    rag_docs = [c for c in context if c.get("type") == "rag"]
    web_results = [c for c in context if c.get("type") == "web"]

    # 至少有一方有结果 → 放行到 generate_answer
    if rag_docs or web_results:
        return {"context_insufficient": False}

    # 两者均为空 → 用约束型 LLM 生成诚实应答
    user_query = _last_human_query(state)
    logger.info(
        "Context insufficient for query: %s (RAG=0, Web=0), "
        "calling honest-response LLM",
        user_query[:80],
    )

    # —— 使用 Supervisor 模型（便宜 + 快）+ temperature=0.0 ——
    llm = get_node_llm("supervisor", temperature=0.0)
    system_prompt = load_prompt("honest_response_system")

    with traced_llm_call(
        model_name=get_setting("supervisor.model", "Qwen/Qwen2.5-7B-Instruct"),
        node_name="check_context_sufficiency",
        temperature=0.0,
    ):
        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"学生的问题是：{user_query}"),
            ])
            honest_reply = response.content.strip()
        except Exception:
            # —— 兜底：LLM 异常时用硬编码模板（保可用性）——
            logger.warning("Honest-response LLM failed, using static fallback")
            honest_reply = (
                f"抱歉，关于「{user_query[:80]}」，我目前无法找到可靠的参考信息。\n\n"
                "可能的原因：\n"
                "- 相关信息尚未公布或超出了我的知识覆盖范围\n"
                "- 网络搜索也未找到相关内容\n\n"
                "**你可以尝试：**\n"
                "- 换一个相关角度提问，比如备考策略或知识点梳理\n"
                "- 询问近几年的类似内容，我可以帮你分析规律\n\n"
                "我很乐意从其他角度继续帮你备考！"
            )

    return {
        "context_insufficient": True,
        "messages": [AIMessage(content=honest_reply)],
    }


def route_after_check(state: TutorState) -> str:
    """检索充分性检查后的路由决策。

    - context_insufficient=True  → 直接结束（已在 check 中返回回答）
    - context_insufficient=False → 进入 generate_answer 正常生成

    Returns:
        "end" 还是 "continue"
    """
    if state.get("context_insufficient", False):
        return "end"
    return "continue"


# ============================================================================
# 节点 3: Generate Answer（融合生成——Fan-in 汇聚点）
# ============================================================================

def _format_retrieved(docs: list[dict]) -> str:
    """格式化 RAG 检索结果为提示词用的文本块。

    为每个文档标注序号和来源，方便 LLM 在回答中引用。
    """
    if not docs:
        return "无相关参考资料。"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(
            f"[{i}] 来源：{d.get('source', '未知')}"
            f"（相关度：{d.get('score', 'N/A')}）\n{d.get('content', '')}"
        )
    return "\n\n".join(parts)


def _format_search(results: list[dict]) -> str:
    """格式化 Web 搜索结果为提示词用的文本块。

    与 RAG 格式不同，Web 结果包含 URL（供用户核实）。
    """
    if not results:
        return "无网络搜索结果。"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.get('title', '无标题')} ({r.get('url', '')})\n{r.get('content', '')}"
        )
    return "\n\n".join(parts)


# ============================================================================
# Agent 工具配置 —— 为 Function Calling 准备
# ============================================================================

# 工具列表：LLM 在生成过程中可以自主调用的两把"查询工具"
_TOOLS = [search_knowledge_base, search_web]
_TOOL_BY_NAME = {t.name: t for t in _TOOLS}
_MAX_TOOL_ROUNDS = get_setting("academic.max_tool_rounds", 3)


async def _execute_tool(tool_call: dict) -> str:
    """在线程池中执行同步工具调用，返回格式化结果。

    工具函数（search_knowledge_base / search_web）内部调用了
    retrieve() 和 search()，这些是同步 I/O 操作，需要放到线程池
    执行以避免阻塞 FastAPI 事件循环。
    """
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    tool_fn = _TOOL_BY_NAME.get(tool_name)
    if tool_fn is None:
        logger.warning("Unknown tool requested: %s", tool_name)
        return f"错误：未知工具 '{tool_name}'"
    try:
        result = await asyncio.to_thread(tool_fn.invoke, tool_args)
        return str(result)
    except Exception:
        logger.exception("Tool '%s' execution failed", tool_name)
        return f"工具 '{tool_name}' 调用失败，请基于已有信息继续回答。"


# ============================================================================
# 节点 3: Generate Answer（Agent 循环 + Function Calling）
# ============================================================================

@traced_node
async def generate_answer(state: TutorState) -> dict:
    """融合 RAG + Web 检索结果，通过 Agent 循环自主决定是否需要补充查询。

    这是学术分支的 Fan-in 汇聚点 —— rag_retrieve 和 web_search
    都完成后才触发。

    与旧版的区别（Function Calling 升级）：
    — 旧版：LLM 一次性生成，拿到什么上下文用什么，不够就硬编
    — 新版：LLM 生成过程中发现信息不足时，主动调用 search_knowledge_base
      或 search_web 工具补充查询，工具结果实时追加到对话上下文

    安全措施：
    — max_tool_rounds=3：最多 3 轮工具调用，防止死循环
    — 容灾 fallback LLM 同样 bind_tools，保障工具调用不因主模型故障中断
    — 幻觉评估（evaluate_hallucination）仍在此节点之后兜底
    """
    llm = get_node_llm("academic")
    temperature = get_setting("academic.temperature", 0.7)
    question = _last_human_query(state)

    # —— 按来源类型分离上下文（Fan-in 汇聚的结果）——
    context = state.get("context", [])
    rag_docs = [c for c in context if c.get("type") == "rag"]
    web_results = [c for c in context if c.get("type") == "web"]

    # —— 绑定工具到主/副模型 ——
    primary_with_tools = llm.bind_tools(_TOOLS)
    fallback_llm = get_fallback_llm(temperature=temperature)
    fallback_with_tools = fallback_llm.bind_tools(_TOOLS)

    # —— 构建初始消息（初始上下文 + 用户问题）——
    user_prompt = load_prompt("academic_answer").format(
        retrieved_context=_format_retrieved(rag_docs),
        search_context=_format_search(web_results),
        question=question,
    )
    memory_context = build_memory_context(state)
    if memory_context:
        user_prompt = f"{memory_context}\n\n{user_prompt}"
    messages: list = [
        SystemMessage(content=load_prompt("academic_system")),
        HumanMessage(content=user_prompt),
    ]

    # —— Agent 循环：LLM 自主决定调用工具 or 输出最终回答 ——
    tool_rounds = 0
    final_content = ""

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="generate_answer",
        temperature=temperature,
    ) as span:
        for _ in range(_MAX_TOOL_ROUNDS):
            response = await async_invoke_with_fallback(
                primary_with_tools, messages,
                fallback=fallback_with_tools, span=span,
            )
            messages.append(response)

            # 没有工具调用 → LLM 认为信息够了，输出最终回答
            if not response.tool_calls:
                final_content = response.content or ""
                break

            # 有工具调用 → 执行工具，结果追加到对话，继续循环
            tool_rounds += 1
            span.set_attribute("agent.tool_rounds", tool_rounds)

            for tc in response.tool_calls:
                tool_result = await _execute_tool(tc)
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tc["id"],
                ))

        else:
            # 达到最大轮次仍在调工具 → 用最后一轮响应作为最终回答
            logger.warning(
                "Agent reached max tool rounds (%d), forcing final answer",
                _MAX_TOOL_ROUNDS,
            )
            final_content = response.content or "抱歉，我暂时无法完整回答这个问题，请换个方式提问。"

    return {"messages": [AIMessage(content=final_content)]}


# ============================================================================
# 节点 4: Hallucination Evaluation（幻觉评估——反思环节）
# ============================================================================

@traced_node
async def evaluate_hallucination(state: TutorState) -> dict:
    """评估生成回答是否基于检索上下文（vs 模型幻觉）。

    这是学术分支的"质量关卡"——通过的答案才输出给用户，
    不通过的触发查询改写 + 重新检索 + 重新生成。

    为什么 temperature=0.0 很重要？
    幻觉检测必须确定性——同一输入不应有时判"通过"有时判"不通过"。

    容错策略：LLM 调用失败时默认 is_faithful=True（乐观策略），
    避免因为评估节点故障而阻塞用户获取回答。

    State 更新：
    - hallucination_detected=True → retry_count += 1（触发条件重试）
    - hallucination_detected=False → 不做额外修改（流程结束）
    """
    eval_temp = get_setting("academic.hallucination_eval_temperature", 0.0)
    llm = get_node_llm("academic", temperature=eval_temp)
    structured_primary = llm.with_structured_output(HallucinationEvaluation)

    # 容灾链
    fallback_llm = get_fallback_llm(temperature=eval_temp)
    structured_fallback = fallback_llm.with_structured_output(HallucinationEvaluation)

    # 提取回答和原始问题
    answer = state["messages"][-1].content
    question = _last_human_query(state)

    # 拼接所有检索上下文
    docs = state.get("context", [])
    context = "\n".join(d.get("content", "") for d in docs) if docs else ""

    eval_prompt = load_prompt("hallucination_eval").format(
        question=question, context=context, answer=answer,
    )

    retry_count = state.get("retry_count", 0)

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="evaluate_hallucination",
        temperature=eval_temp,
    ) as span:
        try:
            evaluation = await async_invoke_with_fallback(
                structured_primary,
                [
                    SystemMessage(content=load_prompt("hallucination_system")),
                    HumanMessage(content=eval_prompt),
                ],
                fallback=structured_fallback,
                span=span,
            )
            is_faithful = evaluation.is_faithful
        except Exception:
            # —— 容错：评估失败时默认通过 ——
            logger.warning("Hallucination evaluation failed, defaulting to valid")
            is_faithful = True

    hallucination_detected = not is_faithful

    result: dict = {"hallucination_detected": hallucination_detected}
    if hallucination_detected:
        result["retry_count"] = retry_count + 1
        result["hallucination_reason"] = evaluation.reason

    return result


# ============================================================================
# 条件路由：重试 or 结束
# ============================================================================

def should_retry_or_end(state: TutorState) -> str:
    """判断幻觉检测后的路由方向。

    决策逻辑：
    1. 幻觉被检测到 AND 重试次数 ≤ MAX_RETRIES → "retry"（改写查询重试）
    2. 无幻觉 OR 超过最大重试次数 → "end"（结束流程）

    为什么达到上限后即使有幻觉也结束？
    无限重试会陷入死循环——有些"幻觉"可能是评估节点误判，
    应该把最终判断权交给用户。同时控制 Token 消耗。

    Returns:
        "retry" → 触发 rewrite_query → academic_router → 重新检索+生成
        "end"   → 流程结束，返回当前回答
    """
    if (
        state.get("hallucination_detected", False)
        and state.get("retry_count", 0) <= MAX_RETRIES
    ):
        return "retry"
    return "end"
