"""Tests for the file-backed evaluation harness."""

from __future__ import annotations

from pathlib import Path

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
