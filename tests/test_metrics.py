"""Tests for process-local observability metrics."""

from src.tracing.metrics import record_search, reset_metrics, snapshot


def test_search_metrics_expose_result_and_error_rates() -> None:
    reset_metrics()
    record_search(result_count=2)
    record_search(result_count=0)
    record_search(error_type="TimeoutError")

    search = snapshot()["web_search"]

    assert search["requests_total"] == 3
    assert search["results_total"] == 2
    assert search["empty_total"] == 1
    assert search["errors_total"] == 1
    assert search["by_error"] == {"TimeoutError": 1}
    assert search["empty_rate"] == 0.3333
    assert search["error_rate"] == 0.3333
