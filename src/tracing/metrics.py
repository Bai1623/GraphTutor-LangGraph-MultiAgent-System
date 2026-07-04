"""In-process observability metrics.

The app already emits OpenTelemetry spans.  This module keeps a compact
process-local snapshot for health dashboards and tests without requiring a
Prometheus client dependency.
"""

from __future__ import annotations

import time
from copy import deepcopy
from threading import Lock
from typing import Any

_lock = Lock()
_started_at = time.time()

_http: dict[str, Any] = {
    "requests_total": 0,
    "errors_total": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_max": 0.0,
    "by_status": {},
}

_llm: dict[str, Any] = {
    "calls_total": 0,
    "errors_total": 0,
    "fallback_total": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_max": 0.0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "by_node": {},
}

_rag: dict[str, Any] = {
    "retrievals_total": 0,
    "hits_total": 0,
    "docs_total": 0,
    "top_score_sum": 0.0,
}

_cache: dict[str, Any] = {
    "lookups_total": 0,
    "hits_total": 0,
    "stores_total": 0,
}


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _avg(total: int | float, count: int | float) -> float:
    if count <= 0:
        return 0.0
    return round(float(total) / float(count), 2)


def record_http_request(status_code: int, duration_ms: float) -> None:
    with _lock:
        _http["requests_total"] += 1
        if status_code >= 500:
            _http["errors_total"] += 1
        _http["latency_ms_sum"] += duration_ms
        _http["latency_ms_max"] = max(_http["latency_ms_max"], duration_ms)
        by_status = _http["by_status"]
        key = str(status_code)
        by_status[key] = by_status.get(key, 0) + 1


def record_llm_call(
    node_name: str,
    latency_ms: float,
    *,
    error: bool = False,
    fallback_used: bool = False,
) -> None:
    with _lock:
        _llm["calls_total"] += 1
        if error:
            _llm["errors_total"] += 1
        if fallback_used:
            _llm["fallback_total"] += 1
        _llm["latency_ms_sum"] += latency_ms
        _llm["latency_ms_max"] = max(_llm["latency_ms_max"], latency_ms)

        by_node = _llm["by_node"]
        node = by_node.setdefault(
            node_name,
            {
                "calls_total": 0,
                "errors_total": 0,
                "fallback_total": 0,
                "latency_ms_sum": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        node["calls_total"] += 1
        if error:
            node["errors_total"] += 1
        if fallback_used:
            node["fallback_total"] += 1
        node["latency_ms_sum"] += latency_ms


def record_llm_tokens(
    node_name: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    with _lock:
        _llm["input_tokens"] += input_tokens
        _llm["output_tokens"] += output_tokens
        _llm["total_tokens"] += total_tokens

        by_node = _llm["by_node"]
        node = by_node.setdefault(
            node_name,
            {
                "calls_total": 0,
                "errors_total": 0,
                "fallback_total": 0,
                "latency_ms_sum": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        node["input_tokens"] += input_tokens
        node["output_tokens"] += output_tokens
        node["total_tokens"] += total_tokens


def record_rag_retrieval(doc_count: int, is_hit: bool, top_score: float | None = None) -> None:
    with _lock:
        _rag["retrievals_total"] += 1
        if is_hit:
            _rag["hits_total"] += 1
        _rag["docs_total"] += doc_count
        if top_score is not None:
            _rag["top_score_sum"] += top_score


def record_cache_lookup(hit: bool) -> None:
    with _lock:
        _cache["lookups_total"] += 1
        if hit:
            _cache["hits_total"] += 1


def record_cache_store() -> None:
    with _lock:
        _cache["stores_total"] += 1


def snapshot() -> dict[str, Any]:
    with _lock:
        http = deepcopy(_http)
        llm = deepcopy(_llm)
        rag = deepcopy(_rag)
        cache = deepcopy(_cache)

    http["latency_ms_avg"] = _avg(http["latency_ms_sum"], http["requests_total"])
    http["latency_ms_max"] = round(http["latency_ms_max"], 2)
    http["error_rate"] = _rate(http["errors_total"], http["requests_total"])
    http.pop("latency_ms_sum", None)

    llm["latency_ms_avg"] = _avg(llm["latency_ms_sum"], llm["calls_total"])
    llm["latency_ms_max"] = round(llm["latency_ms_max"], 2)
    llm["error_rate"] = _rate(llm["errors_total"], llm["calls_total"])
    llm["fallback_rate"] = _rate(llm["fallback_total"], llm["calls_total"])
    llm.pop("latency_ms_sum", None)
    for node in llm["by_node"].values():
        node["latency_ms_avg"] = _avg(node["latency_ms_sum"], node["calls_total"])
        node["error_rate"] = _rate(node["errors_total"], node["calls_total"])
        node["fallback_rate"] = _rate(node["fallback_total"], node["calls_total"])
        node.pop("latency_ms_sum", None)

    rag["hit_rate"] = _rate(rag["hits_total"], rag["retrievals_total"])
    rag["docs_avg"] = _avg(rag["docs_total"], rag["retrievals_total"])
    rag["top_score_avg"] = _avg(rag["top_score_sum"], rag["retrievals_total"])
    rag.pop("top_score_sum", None)

    cache["hit_rate"] = _rate(cache["hits_total"], cache["lookups_total"])

    return {
        "process": {
            "started_at": _started_at,
            "uptime_seconds": round(time.time() - _started_at, 2),
        },
        "http": http,
        "llm": llm,
        "rag": rag,
        "semantic_cache": cache,
    }


def reset_metrics() -> None:
    """Reset counters for unit tests."""
    global _started_at
    with _lock:
        _started_at = time.time()
        _http.update({
            "requests_total": 0,
            "errors_total": 0,
            "latency_ms_sum": 0.0,
            "latency_ms_max": 0.0,
            "by_status": {},
        })
        _llm.update({
            "calls_total": 0,
            "errors_total": 0,
            "fallback_total": 0,
            "latency_ms_sum": 0.0,
            "latency_ms_max": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "by_node": {},
        })
        _rag.update({
            "retrievals_total": 0,
            "hits_total": 0,
            "docs_total": 0,
            "top_score_sum": 0.0,
        })
        _cache.update({
            "lookups_total": 0,
            "hits_total": 0,
            "stores_total": 0,
        })
