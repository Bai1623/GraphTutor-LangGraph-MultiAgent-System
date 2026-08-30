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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eval"

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.evaluation.golden_dataset import load_golden_suite

SUITE_FILES = {
    "gate": "quality_gate.yaml",
    "hallucination": "hallucination.yaml",
    "hallucination_quality": "hallucination.yaml",
    "quality": "quality_gate.yaml",
    "quality_gate": "quality_gate.yaml",
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

    return load_golden_suite(path)


def _hit(expected_sources: list[str], source: str) -> bool:
    return any(expected in source for expected in expected_sources)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _empty_cost_latency() -> dict[str, Any]:
    return {
        "total_tokens": 0,
        "node_tokens": {},
        "wall_time_ms": 0.0,
        "node_latency_ms": {},
        "fallback_used": False,
        "tool_rounds": 0,
        "retry_count": 0,
        "adv_round": 0,
    }


def _merge_numeric_maps(items: list[dict[str, Any]], key: str) -> dict[str, int | float]:
    merged: dict[str, float] = defaultdict(float)
    has_float = False
    for item in items:
        for name, value in item.get(key, {}).items():
            number = _to_float(value)
            has_float = has_float or not float(number).is_integer()
            merged[str(name)] += number
    return {
        name: round(value, 1) if has_float else int(value)
        for name, value in sorted(merged.items())
    }


def _cost_latency_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [d.get("cost_latency", {}) for d in details]
    total = len(costs)
    wall_times = [_to_float(c.get("wall_time_ms")) for c in costs]
    retry_counts = [_to_int(c.get("retry_count")) for c in costs]
    adv_rounds = [_to_int(c.get("adv_round")) for c in costs]
    return {
        "total_tokens": sum(_to_int(c.get("total_tokens")) for c in costs),
        "node_tokens": _merge_numeric_maps(costs, "node_tokens"),
        "wall_time_ms": round(sum(wall_times), 1),
        "avg_wall_time_ms": round(sum(wall_times) / total, 1) if total else 0,
        "node_latency_ms": _merge_numeric_maps(costs, "node_latency_ms"),
        "fallback_used": any(bool(c.get("fallback_used")) for c in costs),
        "tool_rounds": sum(_to_int(c.get("tool_rounds")) for c in costs),
        "retry_count": sum(retry_counts),
        "max_retry_count": max(retry_counts) if retry_counts else 0,
        "avg_retry_count": round(sum(retry_counts) / total, 2) if total else 0,
        "adv_round": sum(adv_rounds),
        "max_adv_round": max(adv_rounds) if adv_rounds else 0,
        "avg_adv_round": round(sum(adv_rounds) / total, 2) if total else 0,
    }


class EvalTelemetry:
    """Collect cost and latency signals from LangGraph stream events."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self._node_starts: dict[str, float] = {}
        self.node_latency_ms: dict[str, float] = defaultdict(float)
        self.node_tokens: dict[str, int] = defaultdict(int)
        self.total_tokens = 0
        self.fallback_used = False
        self.tool_rounds = 0
        self.retry_count = 0
        self.adv_round = 0

    def observe_state(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        self.retry_count = max(self.retry_count, _to_int(value.get("retry_count")))
        self.adv_round = max(self.adv_round, _to_int(value.get("adv_round")))
        self.fallback_used = self.fallback_used or bool(value.get("fallback_used", False))

    def observe_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event")
        metadata = event.get("metadata", {}) or {}
        node_name = metadata.get("langgraph_node") or event.get("name")

        if event_type == "on_tool_start":
            self.tool_rounds += 1

        if event_type in {"on_chain_start", "on_chain_end"}:
            event_name = event.get("name")
            meta_node = metadata.get("langgraph_node")
            if event_name and event_name == meta_node:
                if event_type == "on_chain_start":
                    self._node_starts[event_name] = time.monotonic()
                else:
                    start = self._node_starts.pop(event_name, None)
                    if start is not None:
                        self.node_latency_ms[event_name] += (time.monotonic() - start) * 1000

            output = event.get("data", {}).get("output")
            self.observe_state(output)

        if event_type == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            usage = getattr(output, "usage_metadata", None)
            if isinstance(usage, dict):
                total = _to_int(usage.get("total_tokens"))
                if total == 0:
                    total = _to_int(usage.get("input_tokens")) + _to_int(usage.get("output_tokens"))
                self.total_tokens += total
                if node_name:
                    self.node_tokens[str(node_name)] += total

    def finish(self, *, wall_time_ms: float | None = None) -> dict[str, Any]:
        if wall_time_ms is None:
            wall_time_ms = (time.monotonic() - self.started_at) * 1000
        return {
            "total_tokens": self.total_tokens,
            "node_tokens": dict(sorted(self.node_tokens.items())),
            "wall_time_ms": round(wall_time_ms, 1),
            "node_latency_ms": {
                name: round(value, 1)
                for name, value in sorted(self.node_latency_ms.items())
            },
            "fallback_used": self.fallback_used,
            "tool_rounds": self.tool_rounds,
            "retry_count": self.retry_count,
            "adv_round": self.adv_round,
        }


async def _run_graph_with_telemetry(
    graph: Any,
    input_data: dict[str, Any],
    config: dict[str, Any],
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    telemetry = EvalTelemetry()
    state: dict[str, Any] = {}
    error: str | None = None

    async def _consume() -> None:
        nonlocal state
        async for event in graph.astream_events(input_data, config=config, version="v2"):
            telemetry.observe_event(event)
            output = event.get("data", {}).get("output")
            if isinstance(output, dict):
                state.update(output)

    try:
        await asyncio.wait_for(_consume(), timeout=timeout_s)
    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as exc:
        error = str(exc)

    telemetry.observe_state(state)
    return state, telemetry.finish(), error


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


def _rag_metric_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    if not total:
        return {
            "total_cases": 0,
            "recall_at_k": 0,
            "precision_at_k": 0,
            "mrr": 0,
            "hit_rate": 0,
        }
    return {
        "total_cases": total,
        "recall_at_k": round(sum(d["recall"] for d in details) / total, 3),
        "precision_at_k": round(sum(d["precision"] for d in details) / total, 3),
        "mrr": round(sum(d["reciprocal_rank"] for d in details) / total, 3),
        "hit_rate": round(sum(1 for d in details if d["hit_count"] > 0) / total, 3),
    }


def _rag_breakdown(details: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    for field in fields:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for detail in details:
            value = detail.get(field)
            if value is None:
                continue
            grouped.setdefault(str(value), []).append(detail)
        breakdown[field] = {
            value: _rag_metric_summary(group_details)
            for value, group_details in sorted(grouped.items())
        }
    return breakdown


def _classification_metric_summary(
    details: list[dict[str, Any]],
    *,
    expected_key: str,
    actual_key: str,
    positive_value: Any,
) -> dict[str, Any]:
    total = len(details)
    correct = sum(1 for d in details if d.get("passed"))
    positives = [d for d in details if d.get(expected_key) == positive_value]
    negatives = [d for d in details if d.get(expected_key) != positive_value]
    true_positives = sum(1 for d in positives if d.get(actual_key) == positive_value)
    true_negatives = sum(1 for d in negatives if d.get(actual_key) != positive_value)
    predicted_positives = [d for d in details if d.get(actual_key) == positive_value]
    false_positives = sum(1 for d in predicted_positives if d.get(expected_key) != positive_value)
    return {
        "total_cases": total,
        "correct": correct,
        "pass_rate": round(correct / total, 3) if total else 0,
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "positive_recall": round(true_positives / len(positives), 3) if positives else 0,
        "negative_recall": round(true_negatives / len(negatives), 3) if negatives else 0,
        "positive_precision": round(
            true_positives / (true_positives + false_positives),
            3,
        ) if (true_positives + false_positives) else 0,
    }


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
        cost_latency = _empty_cost_latency()
        cost_latency.update({
            "wall_time_ms": round(elapsed_ms, 1),
            "node_latency_ms": {"rag_retrieve": round(elapsed_ms, 1)},
            "tool_rounds": 1,
        })

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
            "topic": case.get("topic"),
            "query_type": case.get("query_type"),
            "difficulty": case.get("difficulty"),
            "expected_sources": expected_sources,
            "found_sources": [d.get("source", "?") for d in docs],
            "scores": [round(_to_float(d.get("score")), 3) for d in docs],
            "hit_count": hit_count,
            "first_rank": first_rank,
            "reciprocal_rank": round(rr, 3),
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "elapsed_ms": round(elapsed_ms, 1),
            "cost_latency": cost_latency,
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
        "cost_latency": _cost_latency_summary(details),
        "breakdown": _rag_breakdown(
            details,
            fields=("subject", "topic", "query_type", "difficulty"),
        ),
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
        cost_latency = _empty_cost_latency()
        cost_latency.update({
            "wall_time_ms": round(elapsed_ms, 1),
            "node_latency_ms": {"supervisor": round(elapsed_ms, 1)},
        })

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
            "cost_latency": cost_latency,
        })
        print(f"  [{i}/{len(cases)}] {case.get('id')} {actual_intent} {'OK' if passed else 'FAIL'}")

    n = len(cases)
    metrics = {
        "total_cases": n,
        "correct": correct,
        "accuracy": round(correct / n, 3) if n else 0,
        "avg_latency_ms": round(sum(d["elapsed_ms"] for d in details) / n, 1) if n else 0,
        "cost_latency": _cost_latency_summary(details),
    }
    return _with_thresholds(_base_result(suite, metrics, details), suite.get("thresholds", {}))


async def run_hallucination_suite(suite: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import AIMessage, HumanMessage

    from src.graph.academic import evaluate_hallucination

    defaults = suite.get("defaults", {})
    timeout_s = float(defaults.get("timeout_s", 60))
    cases = suite.get("cases", [])
    details = []

    print(f"Running hallucination suite '{suite['suite']}' ({len(cases)} cases)")

    for i, case in enumerate(cases, 1):
        question = case["question"]
        answer = case["answer"]
        context = case.get("context", [])
        expected = bool(case["expected_hallucination"])
        state = {
            "messages": [HumanMessage(content=question), AIMessage(content=answer)],
            "context": [
                {
                    "type": "rag",
                    "content": item.get("content", ""),
                    "source": item.get("source", "golden"),
                }
                for item in context
            ],
            "retry_count": int(case.get("retry_count", 0)),
        }

        start = time.monotonic()
        error = None
        try:
            output = await asyncio.wait_for(evaluate_hallucination(state), timeout=timeout_s)
        except TimeoutError:
            output = {"hallucination_detected": False}
            error = "timeout"
        except Exception as exc:
            output = {"hallucination_detected": False}
            error = str(exc)
        elapsed_ms = (time.monotonic() - start) * 1000

        actual = bool(output.get("hallucination_detected", False))
        passed = actual == expected and error is None
        cost_latency = _empty_cost_latency()
        cost_latency.update({
            "wall_time_ms": round(elapsed_ms, 1),
            "node_latency_ms": {"evaluate_hallucination": round(elapsed_ms, 1)},
            "retry_count": _to_int(output.get("retry_count")),
        })
        detail = {
            "id": case.get("id", f"case_{i}"),
            "question": question,
            "category": case.get("category"),
            "expected_hallucination": expected,
            "actual_hallucination": actual,
            "reason": output.get("hallucination_reason", ""),
            "passed": passed,
            "error": error,
            "elapsed_ms": round(elapsed_ms, 1),
            "cost_latency": cost_latency,
        }
        details.append(detail)
        status = "OK" if passed else "FAIL"
        print(f"  [{i}/{len(cases)}] {detail['id']} hallucination={actual} {status}")

    summary = _classification_metric_summary(
        details,
        expected_key="expected_hallucination",
        actual_key="actual_hallucination",
        positive_value=True,
    )
    metrics = {
        "total_cases": summary["total_cases"],
        "correct": summary["correct"],
        "pass_rate": summary["pass_rate"],
        "hallucination_recall": summary["positive_recall"],
        "faithful_recall": summary["negative_recall"],
        "hallucination_precision": summary["positive_precision"],
        "avg_latency_ms": round(
            sum(d["elapsed_ms"] for d in details) / len(details),
            1,
        ) if details else 0,
        "cost_latency": _cost_latency_summary(details),
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
        result, cost_latency, error = await _run_graph_with_telemetry(
            graph,
            {"messages": [HumanMessage(content=case["query"])]},
            make_thread_config(thread_id),
            timeout_s,
        )
        elapsed_s = time.monotonic() - start
        cost_latency["wall_time_ms"] = round(elapsed_s * 1000, 1)

        intent = result.get("intent", "unknown")
        adv_round = int(result.get("adv_round", 0) or 0)
        cost_latency["adv_round"] = max(_to_int(cost_latency.get("adv_round")), adv_round)
        cost_latency["retry_count"] = max(
            _to_int(cost_latency.get("retry_count")),
            _to_int(result.get("retry_count")),
        )
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
            "cost_latency": cost_latency,
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
        "cost_latency": _cost_latency_summary(details),
    }
    return _with_thresholds(_base_result(suite, metrics, details), suite.get("thresholds", {}))


def _quality_gate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "total_suites": len(results),
        "passed_suites": sum(1 for result in results if result.get("passed")),
    }
    metrics["overall_pass_rate"] = (
        round(metrics["passed_suites"] / metrics["total_suites"], 3)
        if metrics["total_suites"]
        else 0
    )

    for result in results:
        kind = result.get("kind")
        suite_metrics = result.get("metrics", {})
        if kind == "routing":
            metrics["routing_accuracy"] = suite_metrics.get("accuracy", 0)
        elif kind == "rag":
            metrics["rag_recall_at_k"] = suite_metrics.get("recall_at_k", 0)
            metrics["rag_mrr"] = suite_metrics.get("mrr", 0)
            metrics["rag_hit_rate"] = suite_metrics.get("hit_rate", 0)
            metrics["rag_precision_at_k"] = suite_metrics.get("precision_at_k", 0)
        elif kind == "hallucination":
            metrics["hallucination_pass_rate"] = suite_metrics.get("pass_rate", 0)
            metrics["hallucination_recall"] = suite_metrics.get("hallucination_recall", 0)
            metrics["faithful_recall"] = suite_metrics.get("faithful_recall", 0)

    metrics["cost_latency"] = _cost_latency_summary([
        {"cost_latency": result.get("metrics", {}).get("cost_latency", {})}
        for result in results
    ])
    return metrics


async def run_quality_gate_suite(suite: dict[str, Any]) -> dict[str, Any]:
    sub_suites = suite.get("sub_suites", ["routing", "rag", "hallucination"])
    details = []

    print(f"Running quality gate '{suite['suite']}' ({len(sub_suites)} suites)")

    for suite_name in sub_suites:
        result = await run_suite(str(suite_name))
        details.append({
            "id": result["suite"],
            "kind": result["kind"],
            "passed": bool(result.get("passed")),
            "metrics": result.get("metrics", {}),
            "thresholds": result.get("thresholds", []),
        })
        print(f"  [{result['suite']}] {'PASS' if result.get('passed') else 'FAIL'}")

    metrics = _quality_gate_metrics(details)
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
        if isinstance(value, dict):
            continue
        lines.append(f"| {name} | {value} |")

    cost_latency = result["metrics"].get("cost_latency")
    if isinstance(cost_latency, dict) and cost_latency:
        lines.extend(["", "## Cost And Latency", ""])
        lines.extend([
            "| Metric | Value |",
            "| --- | --- |",
            f"| total_tokens | {cost_latency.get('total_tokens', 0)} |",
            f"| wall_time_ms | {cost_latency.get('wall_time_ms', 0)} |",
            f"| avg_wall_time_ms | {cost_latency.get('avg_wall_time_ms', 0)} |",
            f"| fallback_used | {cost_latency.get('fallback_used', False)} |",
            f"| tool_rounds | {cost_latency.get('tool_rounds', 0)} |",
            f"| retry_count | {cost_latency.get('retry_count', 0)} |",
            f"| max_retry_count | {cost_latency.get('max_retry_count', 0)} |",
            f"| adv_round | {cost_latency.get('adv_round', 0)} |",
            f"| max_adv_round | {cost_latency.get('max_adv_round', 0)} |",
        ])

        node_tokens = cost_latency.get("node_tokens", {})
        if isinstance(node_tokens, dict) and node_tokens:
            lines.extend(["", "### Node Tokens", "", "| Node | Tokens |", "| --- | ---: |"])
            for node, tokens in node_tokens.items():
                lines.append(f"| {node} | {tokens} |")

        node_latency = cost_latency.get("node_latency_ms", {})
        if isinstance(node_latency, dict) and node_latency:
            lines.extend(["", "### Node Latency", "", "| Node | Latency ms |", "| --- | ---: |"])
            for node, latency in node_latency.items():
                lines.append(f"| {node} | {latency} |")

    breakdown = result["metrics"].get("breakdown")
    if isinstance(breakdown, dict) and breakdown:
        lines.extend(["", "## Breakdown", ""])
        for field, groups in breakdown.items():
            lines.extend([
                f"### {field}",
                "",
                "| Group | Cases | Recall@K | Precision@K | MRR | Hit Rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ])
            for group_name, metrics in groups.items():
                lines.append(
                    f"| {group_name} | {metrics['total_cases']} | "
                    f"{metrics['recall_at_k']} | {metrics['precision_at_k']} | "
                    f"{metrics['mrr']} | {metrics['hit_rate']} |"
                )
            lines.append("")

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
    if kind == "hallucination":
        return await run_hallucination_suite(suite)
    if kind == "quality_gate":
        return await run_quality_gate_suite(suite)
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

    suite_names = (
        ["rag", "routing", "hallucination", "planning"]
        if args.suite == "all"
        else [args.suite]
    )
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
