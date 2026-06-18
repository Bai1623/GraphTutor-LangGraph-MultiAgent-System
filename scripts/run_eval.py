# -*- coding: utf-8 -*-
"""Unified evaluation harness for Gaokao Tutor.

Examples:
    python scripts/run_eval.py --suite rag
    python scripts/run_eval.py --suite routing --output artifacts/eval/
    python scripts/run_eval.py --suite planning --no-fail
    python scripts/run_eval.py --suite all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eval"

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

SUITE_FILES = {
    "rag": "rag_retrieval.yaml",
    "rag_retrieval": "rag_retrieval.yaml",
    "routing": "routing.yaml",
    "planning": "planning_quality.yaml",
    "planning_quality": "planning_quality.yaml",
}


def load_suite(suite: str, golden_dir: Path = GOLDEN_DIR) -> dict[str, Any]:
    """Load a golden suite by short name or YAML file path."""
    suite_path = Path(suite)
    if suite_path.suffix in {".yaml", ".yml"}:
        path = suite_path if suite_path.is_absolute() else PROJECT_ROOT / suite_path
    else:
        file_name = SUITE_FILES.get(suite)
        if not file_name:
            known = ", ".join(sorted(SUITE_FILES))
            raise ValueError(f"Unknown suite '{suite}'. Known suites: {known}, all")
        path = golden_dir / file_name

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Suite file must contain a mapping: {path}")
    data["_path"] = str(path)
    return data


def _hit(expected_sources: list[str], source: str) -> bool:
    return any(expected in source for expected in expected_sources)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _threshold_results(metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for name, expected in thresholds.items():
        if name.endswith("_max"):
            metric_name = name[:-4]
            actual = metrics.get(metric_name)
            passed = actual is not None and actual <= expected
            op = "<="
        elif name.endswith("_min"):
            metric_name = name[:-4]
            actual = metrics.get(metric_name)
            passed = actual is not None and actual >= expected
            op = ">="
        else:
            metric_name = name
            actual = metrics.get(metric_name)
            passed = actual is not None and actual >= expected
            op = ">="
        results.append({
            "metric": metric_name,
            "actual": actual,
            "expected": expected,
            "op": op,
            "passed": bool(passed),
        })
    return results


def _with_thresholds(result: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    checks = _threshold_results(result["metrics"], thresholds)
    result["thresholds"] = checks
    result["passed"] = all(check["passed"] for check in checks)
    return result


def run_rag_suite(suite: dict[str, Any]) -> dict[str, Any]:
    from src.rag.retriever import retrieve

    defaults = suite.get("defaults", {})
    top_k = int(defaults.get("top_k", 5))
    cases = suite.get("cases", [])

    recalls = []
    precisions = []
    reciprocal_ranks = []
    hits = []
    details = []

    print(f"Running RAG suite '{suite['suite']}' ({len(cases)} cases, top_k={top_k})")

    for i, case in enumerate(cases, 1):
        query = case["query"]
        subject = case.get("subject")
        expected_sources = case["expected_sources"]

        start = time.monotonic()
        result = retrieve(query=query, subject=subject, top_k=top_k)
        elapsed_ms = (time.monotonic() - start) * 1000

        docs = result.get("docs", [])
        hit_count = 0
        first_rank = 0
        for rank, doc in enumerate(docs, 1):
            if _hit(expected_sources, doc.get("source", "")):
                hit_count += 1
                if first_rank == 0:
                    first_rank = rank

        recall = min(hit_count / max(len(expected_sources), 1), 1.0)
        precision = hit_count / len(docs) if docs else 0.0
        rr = 1.0 / first_rank if first_rank else 0.0

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(rr)
        hits.append(1 if hit_count > 0 else 0)

        details.append({
            "id": case.get("id", f"case_{i}"),
            "query": query,
            "subject": subject,
            "expected_sources": expected_sources,
            "found_sources": [d.get("source", "?") for d in docs],
            "scores": [round(_to_float(d.get("score")), 3) for d in docs],
            "hit_count": hit_count,
            "first_rank": first_rank,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "elapsed_ms": round(elapsed_ms, 1),
        })
        print(f"  [{i}/{len(cases)}] {case.get('id')} hit={hit_count} recall={recall:.2f}")

    n = len(cases)
    metrics = {
        "top_k": top_k,
        "total_cases": n,
        "recall_at_k": round(sum(recalls) / n, 3) if n else 0,
        "precision_at_k": round(sum(precisions) / n, 3) if n else 0,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0,
        "hit_rate": round(sum(hits) / n, 3) if n else 0,
        "avg_latency_ms": round(sum(d["elapsed_ms"] for d in details) / n, 1) if n else 0,
    }
    return _with_thresholds(_base_result(suite, metrics, details), suite.get("thresholds", {}))


async def run_routing_suite(suite: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    from src.graph.supervisor import supervisor_node

    cases = suite.get("cases", [])
    details = []
    correct = 0

    print(f"Running routing suite '{suite['suite']}' ({len(cases)} cases)")

    for i, case in enumerate(cases, 1):
        start = time.monotonic()
        output = await supervisor_node({"messages": [HumanMessage(content=case["query"])]})
        elapsed_ms = (time.monotonic() - start) * 1000

        expected_intent = case["expected_intent"]
        expected_subject = case.get("expected_subject")
        actual_intent = output.get("intent")
        actual_subject = output.get("subject")
        passed = actual_intent == expected_intent
        if expected_subject is not None:
            passed = passed and actual_subject == expected_subject
        if passed:
            correct += 1

        details.append({
            "id": case.get("id", f"case_{i}"),
            "query": case["query"],
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "expected_subject": expected_subject,
            "actual_subject": actual_subject,
            "keypoints": output.get("keypoints", []),
            "passed": passed,
            "elapsed_ms": round(elapsed_ms, 1),
        })
        print(f"  [{i}/{len(cases)}] {case.get('id')} {actual_intent} {'OK' if passed else 'FAIL'}")

    n = len(cases)
    metrics = {
        "total_cases": n,
        "correct": correct,
        "accuracy": round(correct / n, 3) if n else 0,
        "avg_latency_ms": round(sum(d["elapsed_ms"] for d in details) / n, 1) if n else 0,
    }
    return _with_thresholds(_base_result(suite, metrics, details), suite.get("thresholds", {}))


async def run_planning_suite(suite: dict[str, Any]) -> dict[str, Any]:
    from collections import Counter

    from langchain_core.messages import HumanMessage

    from src.database.checkpointer import make_thread_config
    from src.graph.builder import build_graph

    defaults = suite.get("defaults", {})
    delay_s = float(defaults.get("delay_s", 1.0))
    timeout_s = float(defaults.get("timeout_s", 120))
    cases = suite.get("cases", [])
    graph = build_graph().compile()

    details = []
    rounds = []
    draft_lengths = []

    print(f"Running planning suite '{suite['suite']}' ({len(cases)} cases)")

    for i, case in enumerate(cases, 1):
        thread_id = f"eval-planning-{case.get('id', i)}"
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {"messages": [HumanMessage(content=case["query"])]},
                    config=make_thread_config(thread_id),
                ),
                timeout=timeout_s,
            )
            error = None
        except asyncio.TimeoutError:
            result = {}
            error = "timeout"
        except Exception as exc:
            result = {}
            error = str(exc)
        elapsed_s = time.monotonic() - start

        intent = result.get("intent", "unknown")
        adv_round = int(result.get("adv_round", 0) or 0)
        draft_len = len(result.get("draft", "") or result.get("plan", "") or "")
        if intent == "planning" and not error:
            rounds.append(adv_round)
            draft_lengths.append(draft_len)

        details.append({
            "id": case.get("id", f"case_{i}"),
            "query": case["query"],
            "intent": intent,
            "adv_round": adv_round,
            "consensus": result.get("consensus"),
            "draft_len": draft_len,
            "error": error,
            "elapsed_s": round(elapsed_s, 1),
        })
        status = "ERR" if error else intent
        print(f"  [{i}/{len(cases)}] {case.get('id')} {status} rounds={adv_round}")

        if i < len(cases) and delay_s > 0:
            await asyncio.sleep(delay_s)

    n = len(cases)
    planning_routed = sum(1 for d in details if d["intent"] == "planning" and not d["error"])
    first_round = sum(1 for r in rounds if r == 1)
    metrics = {
        "total_cases": n,
        "planning_routed": planning_routed,
        "planning_routed_rate": round(planning_routed / n, 3) if n else 0,
        "misclassified": n - planning_routed,
        "avg_rounds": round(sum(rounds) / len(rounds), 2) if rounds else 0,
        "first_round_rate": round(first_round / len(rounds), 3) if rounds else 0,
        "round_distribution": dict(sorted(Counter(rounds).items())),
        "min_draft_len": min(draft_lengths) if draft_lengths else 0,
        "avg_latency_s": round(sum(d["elapsed_s"] for d in details) / n, 1) if n else 0,
    }
    return _with_thresholds(_base_result(suite, metrics, details), suite.get("thresholds", {}))


def _base_result(suite: dict[str, Any], metrics: dict[str, Any], details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "suite": suite["suite"],
        "kind": suite["kind"],
        "description": suite.get("description", ""),
        "golden_path": suite.get("_path"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "details": details,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Eval Report: {result['suite']}",
        "",
        f"- Kind: `{result['kind']}`",
        f"- Generated: {result['generated_at']}",
        f"- Golden: `{result.get('golden_path')}`",
        f"- Passed: {'yes' if result.get('passed') else 'no'}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for name, value in result["metrics"].items():
        lines.append(f"| {name} | {value} |")

    if result.get("thresholds"):
        lines.extend(["", "## Thresholds", "", "| Metric | Actual | Expected | Result |", "| --- | --- | --- | --- |"])
        for check in result["thresholds"]:
            status = "PASS" if check["passed"] else "FAIL"
            expected = f"{check['op']} {check['expected']}"
            lines.append(f"| {check['metric']} | {check['actual']} | {expected} | {status} |")

    lines.extend(["", "## Cases", ""])
    for detail in result["details"]:
        status = "PASS"
        if detail.get("passed") is False or detail.get("error"):
            status = "FAIL"
        label = detail.get("id", detail.get("query", "case"))
        lines.append(f"### {label} ({status})")
        for key, value in detail.items():
            if key == "id":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")

    return "\n".join(lines)


def _resolve_output_paths(output: Path, suite_name: str) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output.suffix.lower() == ".json":
        json_path = output if output.is_absolute() else PROJECT_ROOT / output
        md_path = json_path.with_suffix(".md")
    else:
        out_dir = output if output.is_absolute() else PROJECT_ROOT / output
        json_path = out_dir / f"{suite_name}_{timestamp}.json"
        md_path = out_dir / f"{suite_name}_{timestamp}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return json_path, md_path


async def run_suite(suite_name: str) -> dict[str, Any]:
    suite = load_suite(suite_name)
    kind = suite["kind"]
    if kind == "rag":
        return run_rag_suite(suite)
    if kind == "routing":
        return await run_routing_suite(suite)
    if kind == "planning":
        return await run_planning_suite(suite)
    raise ValueError(f"Unsupported suite kind: {kind}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gaokao Tutor golden evaluations")
    parser.add_argument(
        "--suite",
        default="rag",
        help="Suite to run: rag, routing, planning, all, or a YAML path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory, or a JSON file path when running one suite",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 even when thresholds fail",
    )
    args = parser.parse_args()

    suite_names = ["rag", "routing", "planning"] if args.suite == "all" else [args.suite]
    if len(suite_names) > 1 and args.output.suffix.lower() == ".json":
        print("--output must be a directory when running multiple suites", file=sys.stderr)
        return 2

    results = []
    for suite_name in suite_names:
        result = await run_suite(suite_name)
        json_path, md_path = _resolve_output_paths(args.output, result["suite"])
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(result), encoding="utf-8")
        print(f"Saved JSON: {json_path}")
        print(f"Saved report: {md_path}")
        results.append(result)

    failed = [r["suite"] for r in results if not r.get("passed")]
    if failed:
        print(f"Threshold failures: {', '.join(failed)}")
        return 0 if args.no_fail else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
