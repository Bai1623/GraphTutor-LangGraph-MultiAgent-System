"""Tests for the file-backed evaluation harness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_eval


def test_loads_rag_golden_suite() -> None:
    suite = run_eval.load_suite("rag")

    assert suite["suite"] == "rag_retrieval"
    assert suite["kind"] == "rag"
    assert suite["cases"]
    assert "thresholds" in suite
    assert suite["metadata"]["version"] == "v1.0.0"
    assert suite["metadata"]["research_goal"]


def test_loads_hallucination_and_quality_gate_suites() -> None:
    hallucination = run_eval.load_suite("hallucination")
    gate = run_eval.load_suite("quality_gate")

    assert hallucination["kind"] == "hallucination"
    assert hallucination["cases"]
    assert gate["kind"] == "quality_gate"
    assert gate["sub_suites"] == ["routing", "rag", "hallucination"]


def test_all_golden_suites_have_valid_schemas() -> None:
    for path in sorted(run_eval.GOLDEN_DIR.glob("*.yaml")):
        suite = run_eval.load_suite(str(path))

        assert suite["suite"]


def test_load_suite_rejects_missing_kind_specific_field(tmp_path: Path) -> None:
    path = tmp_path / "invalid-routing.yaml"
    path.write_text(
        """
suite: routing
kind: routing
description: Invalid routing fixture.
metadata:
  dataset_name: fixture
  version: v1.0.0
  source: test fixture
  updated_at: "2026-09-05"
  research_goal: Validate schema behavior.
thresholds:
  accuracy: 0.9
cases:
  - id: missing_expected_intent
    query: 帮我制定复习计划
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"cases\.0\.expected_intent"):
        run_eval.load_suite(str(path))


def test_load_suite_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-routing.yaml"
    path.write_text(
        """
suite: routing
kind: routing
description: Invalid routing fixture.
metadata:
  dataset_name: fixture
  version: v1.0.0
  source: test fixture
  updated_at: "2026-09-05"
  research_goal: Validate schema behavior.
thresholds:
  accuracy: 0.9
cases:
  - id: duplicate
    query: 第一个问题
    expected_intent: academic
  - id: duplicate
    query: 第二个问题
    expected_intent: planning
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case id 'duplicate'"):
        run_eval.load_suite(str(path))


def test_load_suite_rejects_missing_required_threshold(tmp_path: Path) -> None:
    path = tmp_path / "missing-routing-threshold.yaml"
    path.write_text(
        """
suite: routing
kind: routing
description: Invalid routing fixture.
metadata:
  dataset_name: fixture
  version: v1.0.0
  source: test fixture
  updated_at: "2026-09-05"
  research_goal: Validate schema behavior.
thresholds:
  unrelated_metric: 0.9
cases:
  - id: valid_case
    query: 帮我制定复习计划
    expected_intent: planning
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"thresholds\.accuracy"):
        run_eval.load_suite(str(path))


def test_load_suite_rejects_coerced_boolean(tmp_path: Path) -> None:
    path = tmp_path / "invalid-hallucination.yaml"
    path.write_text(
        """
suite: hallucination
kind: hallucination
description: Invalid hallucination fixture.
metadata:
  dataset_name: fixture
  version: v1.0.0
  source: test fixture
  updated_at: "2026-09-05"
  research_goal: Validate schema behavior.
defaults:
  timeout_s: 60
thresholds:
  pass_rate: 0.9
  faithful_recall: 0.9
  hallucination_recall: 0.9
cases:
  - id: invalid_boolean
    category: faithful
    question: 导数是什么？
    context:
      - source: math
        content: 导数表示变化率。
    answer: 导数表示变化率。
    expected_hallucination: "false"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"cases\.0\.expected_hallucination"):
        run_eval.load_suite(str(path))


def test_load_suite_rejects_non_string_schema_name(tmp_path: Path) -> None:
    path = tmp_path / "invalid-schema-name.yaml"
    path.write_text(
        """
suite: invalid
kind:
  - routing
description: Invalid discriminator fixture.
metadata:
  dataset_name: fixture
  version: v1.0.0
  source: test fixture
  updated_at: "2026-09-05"
  research_goal: Validate schema behavior.
thresholds:
  accuracy: 0.9
cases: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema name must be a string"):
        run_eval.load_suite(str(path))


def test_load_suite_rejects_invalid_dataset_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid-metadata.yaml"
    path.write_text(
        """
suite: routing
kind: routing
description: Invalid metadata fixture.
metadata:
  dataset_name: routing
  version: 1.0
  source: hand-written
  updated_at: 2026-09-05
  research_goal: Validate routing.
thresholds:
  accuracy: 0.9
cases:
  - id: valid_case
    query: 帮我制定复习计划
    expected_intent: planning
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"metadata\.version"):
        run_eval.load_suite(str(path))


def test_base_result_and_markdown_include_dataset_version() -> None:
    suite = {
        "suite": "routing",
        "kind": "routing",
        "description": "Test suite.",
        "metadata": {
            "dataset_name": "routing-fixture",
            "version": "v2.1.0",
        },
        "_path": "routing.yaml",
    }

    result = run_eval._base_result(suite, {"accuracy": 1.0}, [])

    assert result["dataset"]["version"] == "v2.1.0"
    markdown = run_eval.render_markdown(result)
    assert "Dataset version: `v2.1.0`" in markdown


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


def test_classification_metric_summary_tracks_positive_and_negative_recall() -> None:
    details = [
        {"expected": True, "actual": True, "passed": True},
        {"expected": True, "actual": False, "passed": False},
        {"expected": False, "actual": False, "passed": True},
        {"expected": False, "actual": True, "passed": False},
    ]

    metrics = run_eval._classification_metric_summary(
        details,
        expected_key="expected",
        actual_key="actual",
        positive_value=True,
    )

    assert metrics["pass_rate"] == 0.5
    assert metrics["positive_recall"] == 0.5
    assert metrics["negative_recall"] == 0.5
    assert metrics["positive_precision"] == 0.5


def test_quality_gate_metrics_flattens_core_signals() -> None:
    results = [
        {
            "id": "routing",
            "kind": "routing",
            "passed": True,
            "metrics": {"accuracy": 0.95, "cost_latency": {"total_tokens": 10}},
        },
        {
            "id": "rag_retrieval",
            "kind": "rag",
            "passed": True,
            "metrics": {
                "recall_at_k": 0.9,
                "mrr": 0.8,
                "hit_rate": 1.0,
                "precision_at_k": 0.3,
                "cost_latency": {"total_tokens": 0},
            },
        },
        {
            "id": "hallucination",
            "kind": "hallucination",
            "passed": False,
            "metrics": {
                "pass_rate": 0.75,
                "hallucination_recall": 0.5,
                "faithful_recall": 1.0,
                "cost_latency": {"total_tokens": 20},
            },
        },
    ]

    metrics = run_eval._quality_gate_metrics(results)

    assert metrics["overall_pass_rate"] == 0.667
    assert metrics["routing_accuracy"] == 0.95
    assert metrics["rag_recall_at_k"] == 0.9
    assert metrics["rag_mrr"] == 0.8
    assert metrics["hallucination_pass_rate"] == 0.75
    assert metrics["cost_latency"]["total_tokens"] == 30


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
