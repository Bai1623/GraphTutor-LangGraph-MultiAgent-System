"""Measure adversarial convergence rounds and hallucination retry success rate.

Usage:
    python scripts/measure_convergence.py           # run both suites
    python scripts/measure_convergence.py --planning-only
    python scripts/measure_convergence.py --academic-only
    python scripts/measure_convergence.py --verbose # show per-query results
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ── Path setup ──────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from langchain_core.messages import HumanMessage

from src.database.checkpointer import make_thread_config
from src.graph.builder import build_graph

# ══════════════════════════════════════════════════════════════════════
# Test query suites
# ══════════════════════════════════════════════════════════════════════

PLANNING_QUERIES = [
    # ── 纯 planning 意图 ── 不同变体确保 supervisor 正确路由 ──
    "帮我制定一个三个月的数学复习计划",
    "我数学很差英语还行，帮我规划一下高考前60天的复习",
    "距离高考还有两个月，怎么安排各科复习时间",
    "帮我做一个理科综合的冲刺计划",
    "我每天只有3小时学习时间，怎么高效安排",
    "高考倒计时100天，帮我制定一个全科复习方案",
    "物理和化学都不太好，帮我规划一下怎么提升",
    "帮我制定一个适合基础薄弱学生的语文复习计划",
    "我是艺考生，文化课落下很多，怎么安排复习",
    "高三下学期怎么平衡各科学习时间",
    "英语单词总是记不住，帮我规划一个背单词计划",
    "帮我做一个每周的刷题计划，数学和理综为主",
    "基础很差想从零开始复习，帮我制定一个半年计划",
    "帮我制定高考前最后一个月的冲刺安排",
    "生物和化学靠背的多，帮我规划一个记忆型复习计划",
    "每天晚自习三个小时，帮我安排数学和英语的轮换学习",
    "周末两天怎么利用，帮我做一个周末强化计划",
    "物理大题总是丢分，帮我规划一个专题突破方案",
    "语文作文和阅读怎么分配时间训练",
    "帮我制定一个理科生的一轮复习完整计划",
    # ── 边界 case（supervisor 可能误判为 academic）──
    "高考数学函数的复习计划怎么制定",
    "英语完形填空的复习时间怎么安排",
    "三角函数部分怎么安排学习和刷题",
]

ACADEMIC_QUERIES = [
    # ── 精确查询（RAG 容易命中，预期不触发幻觉）──
    "2024年高考语文作文题是什么",
    "二次函数的判别式公式是什么",
    "勾股定理的表达式是什么",
    "椭圆的标准方程是什么",
    "高考英语作文一般要求多少字",
    "三角函数中 sin30° 等于多少",
    "导数求极值的步骤是什么",
    "等比数列求和公式是什么",
    "文言文中'之'的用法有哪些",
    "动量守恒定律的公式是什么",
    # ── 模糊/宽泛查询（RAG 可能返回不相关内容，易触发幻觉）──
    "函数怎么学才能提高",
    "高考数学怎么考高分",
    "阅读理解有什么技巧",
    "作文怎么写才能得高分",
    "物理大题解题有什么思路",
    "怎么快速提高英语成绩",
    "化学方程式有什么记忆技巧",
    "数学压轴题有什么通用解法",
    "怎么培养数学思维",
    "高考前最后两周怎么有效利用",
    # ── 精确但可能超出知识库范围的问题 ──
    "2026年高考数学考试大纲有什么变化",
    "新高考3+1+2模式对选科有什么影响",
    "高考志愿填报应该注意什么",
    "强基计划对高考录取有什么影响",
    "高考生物实验题有什么新趋势",
]

# ══════════════════════════════════════════════════════════════════════
# Measurement logic
# ══════════════════════════════════════════════════════════════════════


def _extract_state_value(result: dict, key: str, default: Any = 0) -> Any:
    """Safely extract a value from LangGraph's ainvoke result.

    ``graph.ainvoke()`` returns the full TutorState dict when the graph
    reaches END, or the last state snapshot before interrupt.
    """
    return result.get(key, default)


async def _run_single_query(
    graph,
    query: str,
    thread_id: str,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one query through the graph and return extracted metrics."""
    config = make_thread_config(thread_id)
    state_input = {"messages": [HumanMessage(content=query)]}

    t0 = time.monotonic()

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(state_input, config=config),
            timeout=120,  # 2 min max per query
        )
    except asyncio.TimeoutError:
        return {"query": query, "error": "timeout", "intent": "unknown"}
    except Exception as exc:
        return {"query": query, "error": str(exc), "intent": "unknown"}

    elapsed = round(time.monotonic() - t0, 1)
    intent = _extract_state_value(result, "intent", "unknown")

    metrics: dict[str, Any] = {
        "query": query,
        "intent": intent,
        "elapsed_s": elapsed,
    }

    if intent == "planning":
        metrics.update({
            "adv_round": _extract_state_value(result, "adv_round", 0),
            "consensus": _extract_state_value(result, "consensus", True),
            "draft_len": len(_extract_state_value(result, "draft", "")),
        })
    elif intent == "academic":
        metrics.update({
            "retry_count": _extract_state_value(result, "retry_count", 0),
            "hallucination_detected": _extract_state_value(
                result, "hallucination_detected", False
            ),
            "hallucination_reason": _extract_state_value(
                result, "hallucination_reason", ""
            ),
        })

    if verbose:
        print(f"  [{intent}] {query[:40]:<40} → {elapsed}s", end="")
        if intent == "planning":
            adv = metrics["adv_round"]
            ok = "✓" if metrics["consensus"] else "✗"
            print(f"  rounds={adv} consensus={ok}")
        elif intent == "academic":
            rc = metrics["retry_count"]
            halluc = "HALLUC" if metrics["hallucination_detected"] else "OK"
            print(f"  retries={rc} {halluc}")
        else:
            print(f"  (misclassified)")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Planning suite
# ──────────────────────────────────────────────────────────────────────


async def measure_planning(
    graph,
    queries: list[str],
    *,
    verbose: bool = False,
    delay_s: float = 1.0,
) -> dict[str, Any]:
    """Run planning queries and collect convergence statistics.

    Returns a dict with:
        total, misclassified, avg_rounds, first_round_rate, forced_rate,
        round_distribution, per_query_details
    """
    print(f"\n{'='*60}")
    print(f"  Planning Convergence Measurement — {len(queries)} queries")
    print(f"{'='*60}")

    rounds: list[int] = []
    details: list[dict] = []

    for i, query in enumerate(queries, 1):
        tid = f"measure-planning-{i}"
        m = await _run_single_query(graph, query, tid, verbose=verbose)
        details.append(m)

        if m.get("intent") == "planning" and "error" not in m:
            rounds.append(m["adv_round"])

        # Rate-limit friendliness
        if i < len(queries):
            await asyncio.sleep(delay_s)

    # ── Statistics ────────────────────────────────────────────────
    planning_count = len([d for d in details if d.get("intent") == "planning"])
    misclassified = len(details) - planning_count

    if rounds:
        avg = sum(rounds) / len(rounds)
        first_round = sum(1 for r in rounds if r == 1)
        first_round_rate = first_round / len(rounds) * 100
        forced = sum(1 for d in details if d.get("adv_round", 0) >= 3 and d.get("consensus"))
        forced_rate = forced / len(rounds) * 100
        dist = Counter(rounds)
    else:
        avg = first_round_rate = forced_rate = 0
        dist = {}

    return {
        "total": len(queries),
        "planning_routed": planning_count,
        "misclassified": misclassified,
        "avg_rounds": round(avg, 2),
        "first_round_rate": round(first_round_rate, 1),
        "forced_rate": round(forced_rate, 1),
        "round_distribution": dict(sorted(dist.items())),
        "details": details,
    }


# ──────────────────────────────────────────────────────────────────────
# Academic suite
# ──────────────────────────────────────────────────────────────────────


async def measure_academic(
    graph,
    queries: list[str],
    *,
    verbose: bool = False,
    delay_s: float = 1.0,
) -> dict[str, Any]:
    """Run academic queries and collect hallucination-retry statistics.

    Returns a dict with:
        total, misclassified, hallucination_triggered_count,
        retry_recovery_rate, retry_exhausted_count,
        per_query_details
    """
    print(f"\n{'='*60}")
    print(f"  Hallucination Retry Measurement — {len(queries)} queries")
    print(f"{'='*60}")

    details: list[dict] = []
    triggered: list[dict] = []   # queries that needed retry
    exhausted: list[dict] = []   # queries where retries ran out

    for i, query in enumerate(queries, 1):
        tid = f"measure-academic-{i}"
        m = await _run_single_query(graph, query, tid, verbose=verbose)
        details.append(m)

        if m.get("intent") == "academic" and "error" not in m:
            if m["retry_count"] > 0:
                triggered.append(m)
                # Exhausted = still hallucinating after all retries
                if m["hallucination_detected"]:
                    exhausted.append(m)

        if i < len(queries):
            await asyncio.sleep(delay_s)

    # ── Statistics ────────────────────────────────────────────────
    academic_count = len([d for d in details if d.get("intent") == "academic"])
    misclassified = len(details) - academic_count

    triggered_count = len(triggered)
    exhausted_count = len(exhausted)

    if triggered_count > 0:
        recovered = triggered_count - exhausted_count
        recovery_rate = recovered / triggered_count * 100
    else:
        recovery_rate = 100.0

    # No-retry rate (answers that passed hallucination eval on first try)
    if academic_count > 0:
        no_retry_rate = (academic_count - triggered_count) / academic_count * 100
    else:
        no_retry_rate = 0

    # Average retry count
    all_retries = [d.get("retry_count", 0) for d in details if d.get("intent") == "academic"]
    avg_retries = sum(all_retries) / len(all_retries) if all_retries else 0

    return {
        "total": len(queries),
        "academic_routed": academic_count,
        "misclassified": misclassified,
        "hallucination_triggered": triggered_count,
        "hallucination_trigger_rate": (
            round(triggered_count / academic_count * 100, 1) if academic_count else 0
        ),
        "recovered": triggered_count - exhausted_count,
        "exhausted": exhausted_count,
        "recovery_rate": round(recovery_rate, 1),
        "no_retry_rate": round(no_retry_rate, 1),
        "avg_retries": round(avg_retries, 2),
        "details": details,
    }


# ══════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════


def print_report(planning_stats: dict | None, academic_stats: dict | None) -> None:
    """Pretty-print measurement results suitable for resume writing."""
    print(f"\n{'='*60}")
    print("  MEASUREMENT REPORT")
    print(f"{'='*60}")

    if planning_stats and planning_stats["planning_routed"] > 0:
        s = planning_stats
        print(f"\n── 对抗循环收敛 ──")
        print(f"  测试查询: {s['total']} 条（成功路由 {s['planning_routed']} 条）")
        print(f"  误分类:   {s['misclassified']} 条")
        print(f"  平均收敛轮次:     {s['avg_rounds']} 轮")
        print(f"  首轮通过率:       {s['first_round_rate']}%")
        print(f"  安全阀触发率:     {s['forced_rate']}%")
        print(f"  轮次分布:         {s['round_distribution']}")
        print(f"")
        print(f"  → 简历可写: 对抗循环平均 {s['avg_rounds']} 轮收敛，"
              f"{s['first_round_rate']}% 计划首轮即通过双审共识")

    if academic_stats and academic_stats["academic_routed"] > 0:
        s = academic_stats
        print(f"\n── 幻觉检测与重试 ──")
        print(f"  测试查询: {s['total']} 条（成功路由 {s['academic_routed']} 条）")
        print(f"  误分类:   {s['misclassified']} 条")
        print(f"  首次生成即通过:   {s['no_retry_rate']}%")
        print(f"  触发重试:         {s['hallucination_triggered']} 次 "
              f"({s['hallucination_trigger_rate']}%)")
        print(f"  重试后恢复:       {s['recovered']} 次")
        print(f"  重试耗尽:         {s['exhausted']} 次")
        print(f"  幻觉自愈率:       {s['recovery_rate']}%")
        print(f"  平均重试次数:     {s['avg_retries']}")
        print(f"")
        if s['hallucination_triggered'] > 0:
            print(f"  → 简历可写: 幻觉自愈率 {s['recovery_rate']}%，"
                  f"触发重试的问题中 {s['recovery_rate']}% 在 1-2 轮内修正")
        else:
            print(f"  → 简历可写: {s['no_retry_rate']}% 回答首次生成即通过幻觉评估，"
                  f"自建评估节点配合自动重试闭环确保内容可靠性")

    print(f"\n{'='*60}")
    print("  Resume-ready summary:")
    print(f"{'='*60}")
    parts = []
    if planning_stats and planning_stats["planning_routed"] > 0:
        ps = planning_stats
        parts.append(
            f"对抗循环平均 {ps['avg_rounds']} 轮收敛，"
            f"首轮通过率 {ps['first_round_rate']}%"
        )
    if academic_stats and academic_stats["academic_routed"] > 0:
        a_s = academic_stats
        if a_s["hallucination_triggered"] > 0:
            parts.append(
                f"幻觉自愈率 {a_s['recovery_rate']}%，"
                f"{a_s['hallucination_trigger_rate']}% 回答触发自动重试闭环"
            )
        else:
            parts.append(
                f"{a_s['no_retry_rate']}% 回答首次通过幻觉评估"
            )
    print("  " + "；".join(parts))
    print()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Measure LangGraph convergence metrics")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--academic-only", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Seconds between queries (default: 1.0)")
    parser.add_argument("--planning-count", type=int, default=0,
                       help="Limit planning queries to first N (0 = all)")
    parser.add_argument("--academic-count", type=int, default=0,
                       help="Limit academic queries to first N (0 = all)")
    args = parser.parse_args()

    run_planning = not args.academic_only
    run_academic = not args.planning_only

    # ── Build graph (stateless — no checkpointer needed for measurement) ──
    print("Building graph (stateless mode)...")
    graph = build_graph().compile()
    print(f"Graph compiled. Nodes: {len(graph.nodes)}")

    planning_stats = None
    academic_stats = None

    if run_planning:
        queries = PLANNING_QUERIES[:args.planning_count] if args.planning_count else PLANNING_QUERIES
        planning_stats = await measure_planning(
            graph, queries, verbose=args.verbose, delay_s=args.delay,
        )

    if run_academic:
        queries = ACADEMIC_QUERIES[:args.academic_count] if args.academic_count else ACADEMIC_QUERIES
        academic_stats = await measure_academic(
            graph, queries, verbose=args.verbose, delay_s=args.delay,
        )

    print_report(planning_stats, academic_stats)


if __name__ == "__main__":
    asyncio.run(main())
