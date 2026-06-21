"""Tests for the file-backed evaluation harness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import run_eval


def test_loads_rag_golden_suite() -> None:
    suite = run_eval.load_suite("rag")

    assert suite["suite"] == "rag_retrieval"
    assert suite["kind"] == "rag"
    assert suite["cases"]
    assert "thresholds" in suite


def test_threshold_results_support_min_max_and_default_minimum() -> None:
    metrics = {
        "accuracy": 0.95,
        "avg_rounds": 2.0,
        "first_round_rate": 0.3,
    }
    thresholds = {
        "accuracy": 0.9,
        "avg_rounds_max": 3.0,
        "first_round_rate_min": 0.2,
    }

    checks = run_eval._threshold_results(metrics, thresholds)

    assert all(check["passed"] for check in checks)
    assert [check["metric"] for check in checks] == [
        "accuracy",
        "avg_rounds",
        "first_round_rate",
    ]


def test_threshold_results_report_failures() -> None:
    checks = run_eval._threshold_results({"mrr": 0.5}, {"mrr": 0.75})

    assert checks == [
        {
            "metric": "mrr",
            "actual": 0.5,
            "expected": 0.75,
            "op": ">=",
            "passed": False,
        }
    ]


def test_rag_breakdown_groups_metrics() -> None:
    details = [
        {
            "subject": "math",
            "query_type": "formula",
            "recall": 1.0,
            "precision": 0.4,
            "reciprocal_rank": 1.0,
            "hit_count": 1,
        },
        {
            "subject": "math",
            "query_type": "formula",
            "recall": 0.0,
            "precision": 0.0,
            "reciprocal_rank": 0.0,
            "hit_count": 0,
        },
        {
            "subject": "english",
            "query_type": "method",
            "recall": 1.0,
            "precision": 0.2,
            "reciprocal_rank": 0.5,
            "hit_count": 1,
        },
    ]

    breakdown = run_eval._rag_breakdown(details, fields=("subject", "query_type"))

    assert breakdown["subject"]["math"] == {
        "total_cases": 2,
        "recall_at_k": 0.5,
        "precision_at_k": 0.2,
        "mrr": 0.5,
        "hit_rate": 0.5,
    }
    assert breakdown["subject"]["english"]["hit_rate"] == 1.0
    assert breakdown["query_type"]["formula"]["total_cases"] == 2


def test_eval_telemetry_collects_usage_latency_and_rounds() -> None:
    telemetry = run_eval.EvalTelemetry()

    telemetry.observe_event({
        "event": "on_chain_start",
        "name": "generate_answer",
        "metadata": {"langgraph_node": "generate_answer"},
        "data": {},
    })
    telemetry.observe_event({
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": "generate_answer"},
        "data": {
            "output": SimpleNamespace(usage_metadata={
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            })
        },
    })
    telemetry.observe_event({
        "event": "on_tool_start",
        "name": "rag_tool",
        "metadata": {"langgraph_node": "generate_answer"},
        "data": {},
    })
    telemetry.observe_event({
        "event": "on_chain_end",
        "name": "generate_answer",
        "metadata": {"langgraph_node": "generate_answer"},
        "data": {"output": {"retry_count": 2, "adv_round": 1}},
    })

    cost_latency = telemetry.finish(wall_time_ms=42.0)

    assert cost_latency["total_tokens"] == 150
    assert cost_latency["node_tokens"] == {"generate_answer": 150}
    assert cost_latency["node_latency_ms"]["generate_answer"] >= 0
    assert cost_latency["tool_rounds"] == 1
    assert cost_latency["retry_count"] == 2
    assert cost_latency["adv_round"] == 1
    assert cost_latency["wall_time_ms"] == 42.0


def test_cost_latency_summary_merges_case_costs() -> None:
    details = [
        {
            "cost_latency": {
                "total_tokens": 100,
                "node_tokens": {"a": 40},
                "wall_time_ms": 10.0,
                "node_latency_ms": {"a": 5.0},
                "fallback_used": False,
                "tool_rounds": 1,
                "retry_count": 0,
                "adv_round": 1,
            }
        },
        {
            "cost_latency": {
                "total_tokens": 50,
                "node_tokens": {"a": 10, "b": 40},
                "wall_time_ms": 30.0,
                "node_latency_ms": {"a": 3.0, "b": 7.0},
                "fallback_used": True,
                "tool_rounds": 2,
                "retry_count": 1,
                "adv_round": 2,
            }
        },
    ]

    summary = run_eval._cost_latency_summary(details)

    assert summary["total_tokens"] == 150
    assert summary["node_tokens"] == {"a": 50, "b": 40}
    assert summary["wall_time_ms"] == 40.0
    assert summary["avg_wall_time_ms"] == 20.0
    assert summary["node_latency_ms"] == {"a": 8.0, "b": 7.0}
    assert summary["fallback_used"] is True
    assert summary["tool_rounds"] == 3
    assert summary["retry_count"] == 1
    assert summary["max_adv_round"] == 2


def test_resolve_output_directory_paths(tmp_path: Path) -> None:
    json_path, md_path = run_eval._resolve_output_paths(tmp_path, "rag_retrieval")

    assert json_path.parent == tmp_path
    assert md_path.parent == tmp_path
    assert json_path.name.startswith("rag_retrieval_")
    assert json_path.suffix == ".json"
    assert md_path.suffix == ".md"


def test_resolve_output_json_file_paths(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    json_path, md_path = run_eval._resolve_output_paths(output, "rag_retrieval")

    assert json_path == output
    assert md_path == tmp_path / "result.md"
