"""Tests for cross-report cost aggregation."""

from scripts.aggregate_eval_costs import aggregate_reports, render_markdown


def make_report(suite: str, latency: float, fallback: bool = False) -> dict:
    return {
        "suite": suite,
        "metrics": {
            "cost_latency": {
                "total_tokens": latency * 10,
                "wall_time_ms": latency,
                "tool_rounds": 1,
                "retry_count": 0,
                "adv_round": 1,
                "fallback_used": fallback,
            }
        },
    }


def test_aggregate_reports_groups_cost_signals_and_fallback_rate() -> None:
    summary = aggregate_reports([make_report("routing", 100), make_report("routing", 200, True)])

    data = summary["suites"]["routing"]
    assert data["reports"] == 2
    assert data["fallback_rate"] == 0.5
    assert data["cost_latency"]["wall_time_ms"] == {
        "total": 300.0,
        "average": 150.0,
        "p95": 195.0,
    }
    assert "Fallback rate: `0.5`" in render_markdown(summary)
