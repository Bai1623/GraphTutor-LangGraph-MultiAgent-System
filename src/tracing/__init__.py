"""OpenTelemetry tracing for Gaokao Tutor."""

from src.tracing.collector import get_tracer, setup_tracing, shutdown_tracing
from src.tracing.decorators import (
    traced_llm_call,
    traced_node,
    traced_retrieval,
    traced_search,
)
from src.tracing.logging import (
    REQUEST_ID_HEADER,
    configure_logging,
    get_request_id,
    reset_request_id,
    set_request_id,
)
from src.tracing.metrics import snapshot as metrics_snapshot

__all__ = [
    "setup_tracing",
    "shutdown_tracing",
    "get_tracer",
    "traced_node",
    "traced_llm_call",
    "traced_retrieval",
    "traced_search",
    "REQUEST_ID_HEADER",
    "configure_logging",
    "get_request_id",
    "set_request_id",
    "reset_request_id",
    "metrics_snapshot",
]
