"""子图 B 上半部分：学习规划的情报收集阶段（Planner Intel Gathering）

完整规划子图分为三个阶段：
1. search_policy（本文件）：搜索最新高考政策
2. gather_intel（本文件）：并行情报收集（情绪 + 资源）
3. adversarial planning（plan_adversarial.py）：对抗性起草+审查+HIL

本文件是规划子图的"情报收集阶段"——在起草计划之前，
先了解学生的情况（情绪状态）和可用资源（知识库 + 网络信息）。

数据流：
  search_policy → gather_intel
                    ├─ _gather_emotional_intel（分析情绪）
                    └─ _gather_resource_intel
                         ├─ RAG 检索（本地知识库）
                         └─ Web 搜索（联网信息）

面试追问点：
- 为什么情绪分析和资源分析要并行？
  答：两者互不依赖，并行执行节省约 50% 的 wall-clock 延迟。
  用 asyncio.gather() 实现"同时等待两个结果"。
- gather_intel 为什么清空对抗状态（adv_round=0 等）？
  答：这是"新规划"的开始，需要确保上一次对抗循环的残留状态
  不会影响新计划。plan_adversarial.py 的 consensus_check_node
  依赖这些初始值做正确的判断。
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.memory.artifacts import compact_with_artifact
from src.rag.retriever import retrieve
from src.tools.policy_search import format_policy_results, search_official_policy
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_search
from src.tracing.metrics import record_rag_retrieval

logger = logging.getLogger(__name__)


# ============================================================================
# 节点 1: Search Policy（搜索最新高考政策）
# ============================================================================

_SEARCH_TIMEOUT = get_setting("planner.search_timeout", 15)


@traced_node
async def search_policy(state: TutorState) -> dict:
    """搜索最新高考政策信息作为规划参考。

    高考政策每年都在变化（新高考改革、考试时间调整、录取规则等），
    本地知识库更新不及时，必须通过联网搜索获取最新信息。

    搜索策略：
    - 查询模板："{当前年份}年高考最新政策 考试时间安排 科目改革"
    - 超时 15 秒（通过 settings.yaml 配置）
    - 失败时返回空列表，不阻塞下游节点

    State 更新：
    - search_results: DuckDuckGo 搜索到的政策信息列表
    """
    year = datetime.now().year
    query = f"{year}年高考最新政策 考试时间安排 科目改革"

    with traced_search(query=query, timeout=_SEARCH_TIMEOUT) as span:
        policy_source = "none"
        try:
            search_results = await asyncio.wait_for(
                asyncio.to_thread(
                    search_official_policy,
                    query,
                    topic="高考政策",
                    limit=5,
                ),
                timeout=_SEARCH_TIMEOUT,
            )
            if search_results:
                policy_source = "official_mcp"
                span.set_attribute("search.policy_source", policy_source)
                span.set_attribute("search.result_count", len(search_results))
                span.set_attribute("search.timed_out", False)
                compact_results = [
                    compact_with_artifact(
                        result,
                        kind="official_policy_result",
                        text_key="content",
                        preview_chars=900,
                        metadata={"query": query, "policy_source": policy_source},
                    )
                    for result in search_results
                ]
                return {
                    "search_results": compact_results,
                    "policy_source": policy_source,
                    "policy_query": query,
                }
        except asyncio.TimeoutError:
            span.set_attribute("search.official_timed_out", True)
        except Exception:
            logger.warning("Official policy MCP search failed, falling back to web search", exc_info=True)
            span.set_attribute("search.official_failed", True)

        try:
            search_results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
                timeout=_SEARCH_TIMEOUT,
            )
            policy_source = "web_fallback" if search_results else "none"
            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute("search.policy_source", policy_source)
            span.set_attribute("search.timed_out", False)
        except asyncio.TimeoutError:
            search_results = []
            policy_source = "none"
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.policy_source", policy_source)
            span.set_attribute("search.timed_out", True)
        except Exception:
            search_results = []
            policy_source = "none"
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.policy_source", policy_source)
            span.set_attribute("search.timed_out", False)

    compact_results = [
        compact_with_artifact(
            result,
            kind="policy_web_fallback_result",
            text_key="content",
            preview_chars=700,
            metadata={"query": query, "policy_source": policy_source},
        )
        for result in search_results
    ]

    return {
        "search_results": compact_results,
        "policy_source": policy_source,
        "policy_query": query,
    }


# ============================================================================
# 节点 2: Gather Intel（并行情报收集）
# ============================================================================

def _last_human_query(state: TutorState) -> str:
    """从对话历史中提取用户最新的问题。

    规划分支需要用户的具体需求文本（"帮我制定一个数学复习计划"），
    而不是上课内容（"导数是什么"）。
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


async def _gather_emotional_intel(state: TutorState) -> str:
    """分析学生的情绪状态——为对抗性审查员提供输入。

    在起草计划之前，先了解学生的当前状态：
    - 最近的学习状态如何？（积极/焦虑/疲惫）
    - 有没有特别担心或排斥的科目？
    - 目前的学习压力水平？

    这条信息会传给 planner 的 emotional_reviewer，
    让审查员从"学生是否能承受"的角度把关计划。

    失败容错：LLM 调用失败时返回降级文案，不阻塞流程。
    """
    llm = get_node_llm("emotional")
    fallback = get_fallback_llm(temperature=get_setting("emotional.temperature", 0.8))

    # 拼接对话历史供 LLM 分析
    history_text = "\n".join(
        f"{'学生' if isinstance(m, HumanMessage) else '老师'}: {m.content}"
        for m in state["messages"]
        if hasattr(m, "content")
    )

    messages = [
        SystemMessage(content=load_prompt("gather_emotional_intel")),
        HumanMessage(content=history_text),
    ]

    try:
        with traced_llm_call(
            model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            node_name="gather_emotional_intel",
            temperature=get_setting("emotional.temperature", 0.8),
        ) as span:
            response = await async_invoke_with_fallback(
                llm, messages, fallback=fallback, span=span,
            )
        return response.content.strip()
    except Exception:
        logger.warning("Emotional intel LLM call failed, using fallback", exc_info=True)
        return "无法获取情绪分析，建议按常规方式安排计划。"


async def _gather_resource_intel(state: TutorState) -> str:
    """并行检索 RAG 和 Web，汇总可用学习资源。

    与 search_policy 不同，这一步的检索是针对学生具体问题的
    （如"数学怎么考到 120 分"），而不是笼统的政策搜索。

    使用 asyncio.gather 同时执行 RAG 和 Web 两个异步任务：
    - RAG 检索：从本地知识库找相关的学习方法、知识点
    - Web 搜索：从网络找最新的学习资料和技巧

    两个任务互相独立，任何一个失败不会影响另一个。
    """
    query = _last_human_query(state)
    subject = state.get("subject")
    subj = subject if subject and subject != "other" else None

    async def _rag():
        try:
            result = await asyncio.to_thread(retrieve, query=query, subject=subj)
            docs = result.get("docs", [])
            top_score = None
            if docs:
                top_score = docs[0].get("rerank_score", docs[0].get("score", 0))
            record_rag_retrieval(len(docs), result.get("is_hit", False), top_score)
            if not docs:
                return ""
            parts = [f"- {d.get('content', '')[:200]}" for d in docs[:3]]
            return "【知识库资源】\n" + "\n".join(parts)
        except Exception:
            logger.warning("RAG retrieval failed in gather_intel", exc_info=True)
            return ""

    async def _web():
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
                timeout=_SEARCH_TIMEOUT,
            )
            if not results:
                return ""
            parts = [f"- {r.get('title', '')}: {r.get('content', '')[:200]}" for r in results[:3]]
            return "【网络搜索】\n" + "\n".join(parts)
        except Exception:
            logger.warning("Web search failed in gather_intel", exc_info=True)
            return ""

    # 并行执行两个检索任务
    rag_text, web_text = await asyncio.gather(_rag(), _web())

    combined = "\n\n".join(part for part in [rag_text, web_text] if part)
    return combined if combined else "未获取到相关资源信息。"


@traced_node
async def gather_intel(state: TutorState) -> dict:
    """并行收集情绪情报和资源情报，汇总为 intel_summary。

    这是规划分支的最关键节点——没有准确的情报，后续的对抗性规划
    就是在"闭着眼睛写计划"。

    同时初始化对抗规划状态的字段（adv_round、draft、审查员字段等），
    确保对抗循环从一个干净的状态开始。

    State 更新：
    - emotional_intel: 情绪分析结果
    - resource_intel: 资源检索结果
    - intel_summary: 合并摘要（直接传给 drafter）
    - 对抗状态字段: 全部初始化为空/0/False

    输出到下游（plan_adversarial.py）：
    intel_summary → drafter_node 的输入
    """
    # 并行执行情绪分析和资源检索
    emotional_intel, resource_intel = await asyncio.gather(
        _gather_emotional_intel(state),
        _gather_resource_intel(state),
    )

    policy_info = format_policy_results(state.get("search_results", []))
    policy_source = state.get("policy_source", "none")
    if policy_info:
        policy_section = f"【政策信息】\n来源: {policy_source}\n{policy_info}"
    else:
        policy_section = "【政策信息】\n未获取到官方政策信息。"

    intel_summary = f"{policy_section}\n\n【情绪分析】\n{emotional_intel}\n\n{resource_intel}"

    return {
        "emotional_intel": emotional_intel,
        "resource_intel": resource_intel,
        "intel_summary": intel_summary,
        # 初始化对抗规划状态（确保无残留）
        "adv_round": 0,
        "draft": "",
        "academic_verdict": "",
        "academic_reason": "",
        "emotional_verdict": "",
        "emotional_reason": "",
        "consensus": False,
        "revision_notes": "",
    }
