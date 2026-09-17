"""Tests for baseline/variant evaluation comparisons."""

import pytest

from scripts.compare_eval_reports import compare_reports, render_markdown


def report(commit: str, version: str = "v1.0.0") -> dict:
    return {
        "suite": "routing",
        "dataset": {"version": version},
        "metrics": {"accuracy": 0.75, "cost_latency": {"total_tokens": 100}},
        "experiment": {"git_commit": commit, "seed": 42},
    }


def test_compare_reports_calculates_variant_delta() -> None:
    variant = report("new")
    variant["metrics"]["accuracy"] = 0.875
    variant["metrics"]["cost_latency"]["total_tokens"] = 120

    comparison = compare_reports(report("old"), variant)

    rows = {row["metric"]: row for row in comparison["metrics"]}
    assert rows["metrics.accuracy"]["delta"] == 0.125
    assert rows["metrics.cost_latency.total_tokens"]["delta"] == 20.0
    assert "Baseline commit: `old`" in render_markdown(comparison)


def test_compare_reports_rejects_different_dataset_versions() -> None:
    with pytest.raises(ValueError, match="dataset version"):
        compare_reports(report("old"), report("new", version="v2.0.0"))
