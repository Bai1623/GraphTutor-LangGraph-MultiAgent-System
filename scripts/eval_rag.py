# -*- coding: utf-8 -*-
"""RAG 检索质量评估脚本 — 量化混合检索 vs 纯向量检索的效果差异

评估指标:
- Recall@K:   前K个结果中命中了多少 ground-truth 文档
- Precision@K: 前K个结果中 ground-truth 占比
- MRR:        Mean Reciprocal Rank — 第一个正确答案排名的倒数平均
- Hit Rate:   至少命中一个 ground-truth 的查询比例

测试集设计:
- 基于 data/chinese/ 中的高考语文试卷内容
- 每个查询标注了预期应召回的文件名（ground truth）
- 覆盖: 现代文阅读、文言文、作文、古诗鉴赏 等题型

使用方法:
    cd project_root
    python scripts/eval_rag.py

输出:
    data/eval_report.md — 评估报告（Markdown 格式）
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Windows 控制台编码修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.rag.retriever import retrieve

# ============================================================================
# 测试查询集 (query, subject, expected_source_files)
# ============================================================================
# expected_source_files: 文件名关键词（不要求完全匹配，包含即可命中）

TEST_QUERIES = [
    {
        "query": "2024年新课标I卷的现代文阅读讲了什么",
        "subject": "chinese",
        "expected": ["2024年高考语文试卷（新课标Ⅰ卷）"],
    },
    {
        "query": "2025年全国I卷语文作文题目",
        "subject": "chinese",
        "expected": ["2025年高考语文试卷（全国Ⅰ卷）"],
    },
    {
        "query": "新课标II卷文言文阅读内容",
        "subject": "chinese",
        "expected": ["2024年高考语文试卷（新课标Ⅱ卷）"],
    },
    {
        "query": "古诗鉴赏题的常见题型",
        "subject": "chinese",
        "expected": ["2024年高考语文试卷（新课标Ⅰ卷）", "2024年高考语文试卷（新课标Ⅱ卷）", "2025年高考语文试卷（全国Ⅰ卷）", "2025年高考语文试卷（全国Ⅱ卷）"],
    },
    {
        "query": "高考语文阅读理解答题技巧",
        "subject": "chinese",
        "expected": ["2024年高考语文试卷（新课标Ⅰ卷）", "2024年高考语文试卷（新课标Ⅱ卷）"],
    },
    {
        "query": "2025年语文全国II卷文学类文本",
        "subject": "chinese",
        "expected": ["2025年高考语文试卷（全国Ⅱ卷）"],
    },
    {
        "query": "论述类文本阅读方法",
        "subject": "chinese",
        "expected": ["2024年高考语文试卷（新课标Ⅰ卷）", "2025年高考语文试卷（全国Ⅰ卷）"],
    },
    {
        "query": "古诗词默写常见篇目",
        "subject": "chinese",
        "expected": ["2024年高考语文试卷（新课标Ⅰ卷）", "2024年高考语文试卷（新课标Ⅱ卷）"],
    },
]


# ============================================================================
# 评估指标计算
# ============================================================================

def _hit(expected: list[str], source: str) -> bool:
    """检查文档来源是否命中期望文件（文件名关键词匹配）。"""
    for exp in expected:
        if exp in source:
            return True
    return False


def evaluate(
    queries: list[dict],
    *,
    top_k: int = 5,
    subject_filter: Optional[str] = None,
) -> dict:
    """运行评估并返回指标。

    参数:
        queries: 测试查询列表
        top_k: 评估用的 K 值（默认 5）
        subject_filter: 可选，限制只测试某个学科

    返回:
        {"recall@K": float, "precision@K": float, "mrr": float, "hit_rate": float, "details": [...]}
    """
    recalls = []
    precisions = []
    reciprocal_ranks = []
    hits = []
    details = []

    total = len(queries)
    print(f"评估 {total} 条查询（K={top_k}）...")

    for i, item in enumerate(queries):
        query = item["query"]
        subject = item["subject"] if not subject_filter else subject_filter
        expected = item["expected"]

        # 执行检索
        start = time.monotonic()
        result = retrieve(query=query, subject=subject, top_k=top_k)
        elapsed_ms = (time.monotonic() - start) * 1000

        docs = result.get("docs", [])
        hit_count = 0
        first_rank = 0

        for rank, doc in enumerate(docs, 1):
            source = doc.get("source", "")
            if _hit(expected, source):
                hit_count += 1
                if first_rank == 0:
                    first_rank = rank

        # 计算指标
        recall = min(hit_count / max(len(expected), 1), 1.0)  # 上限 1.0
        precision = hit_count / len(docs) if docs else 0
        rr = 1.0 / first_rank if first_rank > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(rr)
        hits.append(1 if hit_count > 0 else 0)

        details.append({
            "query": query,
            "expected": expected,
            "found": [d.get("source", "?") for d in docs],
            "scores": [round(d.get("score", 0), 3) for d in docs],
            "hit_count": hit_count,
            "first_rank": first_rank,
            "recall": round(recall, 3),
            "ms": round(elapsed_ms, 1),
        })

        print(f"  [{i+1}/{total}] {query[:30]}... "
              f"hit={hit_count}, recall={recall:.2f}, "
              f"{'OK' if hit_count > 0 else 'FAIL'}")

    n = len(queries)
    return {
        "top_k": top_k,
        "total_queries": n,
        "recall_at_k": round(sum(recalls) / n, 3) if n else 0,
        "precision_at_k": round(sum(precisions) / n, 3) if n else 0,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0,
        "hit_rate": round(sum(hits) / n, 3) if n else 0,
        "details": details,
    }


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(hybrid_result: dict) -> str:
    """生成 Markdown 格式的评估报告。"""
    lines = [
        "# RAG 检索质量评估报告",
        "",
        f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**测试查询数**: {hybrid_result['total_queries']}",
        f"**评估 K 值**: Top-{hybrid_result['top_k']}",
        "",
        "## 关键指标",
        "",
        f"| 指标 | 数值 | 说明 |",
        f"|------|------|------|",
        f"| **Recall@{hybrid_result['top_k']}** | {hybrid_result['recall_at_k']:.1%} | 期望文档被召回的比例 |",
        f"| **Precision@{hybrid_result['top_k']}** | {hybrid_result['precision_at_k']:.1%} | 返回结果中相关文档占比 |",
        f"| **MRR** | {hybrid_result['mrr']:.3f} | 第一个正确答案的排名倒数平均 |",
        f"| **Hit Rate** | {hybrid_result['hit_rate']:.1%} | 至少命中一个 ground-truth 的查询比例 |",
        "",
        "## 逐查询详情",
        "",
    ]

    for i, d in enumerate(hybrid_result["details"], 1):
        hit_icon = "✓" if d["hit_count"] > 0 else "✗"
        lines.append(f"### {i}. {hit_icon} {d['query']}")
        lines.append(f"- 期望文件: {', '.join(d['expected'][:2])}")
        lines.append(f"- 检索结果 (前{d['first_rank']}): {', '.join(d['found'][:5])}")
        lines.append(f"- 命中数: {d['hit_count']}, 首个排名: {d['first_rank'] or '未命中'}, 延迟: {d['ms']}ms")
        lines.append("")

    lines.append("---")
    lines.append("*由 scripts/eval_rag.py 自动生成*")

    return "\n".join(lines)


# ============================================================================
# 主入口
# ============================================================================

def main():
    print("=" * 60)
    print("RAG 检索质量评估")
    print("=" * 60)

    # ── 第 1 步: 混合检索评估 ──
    print("\n[Step 1] 混合检索评估 (Vector + BM25 + Reranker)")
    hybrid_result = evaluate(TEST_QUERIES, top_k=5)

    # ── 第 2 步: 输出 ──
    print(f"\n{'='*60}")
    print(f"结果汇总:")
    print(f"  Recall@5   = {hybrid_result['recall_at_k']:.1%}")
    print(f"  Precision@5 = {hybrid_result['precision_at_k']:.1%}")
    print(f"  MRR        = {hybrid_result['mrr']:.3f}")
    print(f"  Hit Rate   = {hybrid_result['hit_rate']:.1%}")
    print(f"  平均延迟   = {sum(d['ms'] for d in hybrid_result['details']) / len(hybrid_result['details']):.0f}ms")

    # ── 第 3 步: 生成报告 ──
    report = generate_report(hybrid_result)
    report_path = PROJECT_ROOT / "data" / "eval_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n完整报告已保存: {report_path}")

    # 同时保存 JSON（转换 float32 → float 避免序列化错误）
    json_path = PROJECT_ROOT / "data" / "eval_result.json"
    json_str = json.dumps(hybrid_result, ensure_ascii=False, indent=2, default=str)
    json_path.write_text(json_str, encoding="utf-8")
    print(f"JSON 结果已保存: {json_path}")


if __name__ == "__main__":
    main()
