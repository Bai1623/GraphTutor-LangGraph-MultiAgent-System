"""Schema validation for file-backed golden evaluation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _GoldenSchema(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


_Rate = Annotated[int | float, Field(ge=0, le=1)]
_NonNegativeNumber = Annotated[int | float, Field(ge=0)]


class _GoldenCase(_GoldenSchema):
    id: str = Field(min_length=1)


class _RoutingCase(_GoldenCase):
    query: str = Field(min_length=1)
    expected_intent: Literal["academic", "planning", "emotional", "unknown"]
    expected_subject: str | None = None


class _RagCase(_GoldenCase):
    query: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    expected_sources: list[str] = Field(min_length=1)


class _HallucinationContext(_GoldenSchema):
    source: str = Field(min_length=1)
    content: str = Field(min_length=1)


class _HallucinationCase(_GoldenCase):
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    context: list[_HallucinationContext] = Field(min_length=1)
    answer: str = Field(min_length=1)
    expected_hallucination: bool


class _PlanningCase(_GoldenCase):
    query: str = Field(min_length=1)


class _CompressionArtifact(_GoldenSchema):
    artifact_id: str = Field(min_length=1)
    kind: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    payload: Any = None


class _CompressionMessage(_GoldenSchema):
    id: str = Field(min_length=1)
    role: Literal["human", "user", "student", "system", "assistant", "ai"]
    content: str = Field(min_length=1)


class _CompressionCase(_GoldenCase):
    artifacts: list[_CompressionArtifact] = Field(default_factory=list)
    messages: list[_CompressionMessage] = Field(min_length=1)
    recent_message_count: int = Field(ge=1)
    expected_constraints: list[str]
    answer_terms: list[str]
    expected_artifact_ids: list[str]
    expected_episode: dict[str, Any] = Field(min_length=1)


class _RoutingThresholds(_GoldenSchema):
    accuracy: _Rate


class _RagThresholds(_GoldenSchema):
    recall_at_k: _Rate
    mrr: _Rate
    hit_rate: _Rate


class _RagDefaults(_GoldenSchema):
    top_k: int = Field(ge=1)


class _HallucinationThresholds(_GoldenSchema):
    pass_rate: _Rate
    faithful_recall: _Rate
    hallucination_recall: _Rate


class _HallucinationDefaults(_GoldenSchema):
    timeout_s: _NonNegativeNumber


class _PlanningThresholds(_GoldenSchema):
    planning_routed_rate: _Rate
    avg_rounds_max: _NonNegativeNumber
    first_round_rate_min: _Rate
    min_draft_len: int = Field(ge=1)


class _PlanningDefaults(_GoldenSchema):
    delay_s: _NonNegativeNumber
    timeout_s: _NonNegativeNumber


class _QualityGateThresholds(_GoldenSchema):
    overall_pass_rate: _Rate
    routing_accuracy: _Rate
    rag_recall_at_k: _Rate
    rag_mrr: _Rate
    rag_hit_rate: _Rate
    hallucination_pass_rate: _Rate


class _CompressionThresholds(_GoldenSchema):
    token_reduction: _Rate
    constraint_retention: _Rate
    answer_consistency: _Rate
    artifact_recoverability: _Rate


class _SuiteSchema(_GoldenSchema):
    suite: str = Field(min_length=1)
    description: str = Field(min_length=1)


class _RoutingSuite(_SuiteSchema):
    suite: Literal["routing"]
    kind: Literal["routing"]
    thresholds: _RoutingThresholds
    cases: list[_RoutingCase] = Field(min_length=1)


class _RagSuite(_SuiteSchema):
    suite: Literal["rag_retrieval"]
    kind: Literal["rag"]
    thresholds: _RagThresholds
    defaults: _RagDefaults
    cases: list[_RagCase] = Field(min_length=1)


class _HallucinationSuite(_SuiteSchema):
    suite: Literal["hallucination"]
    kind: Literal["hallucination"]
    thresholds: _HallucinationThresholds
    defaults: _HallucinationDefaults
    cases: list[_HallucinationCase] = Field(min_length=1)


class _PlanningSuite(_SuiteSchema):
    suite: Literal["planning_quality"]
    kind: Literal["planning"]
    thresholds: _PlanningThresholds
    defaults: _PlanningDefaults
    cases: list[_PlanningCase] = Field(min_length=1)


class _QualityGateSuite(_SuiteSchema):
    suite: Literal["quality_gate"]
    kind: Literal["quality_gate"]
    thresholds: _QualityGateThresholds
    sub_suites: list[str] = Field(min_length=1)


class _CompressionSuite(_SuiteSchema):
    suite: Literal["context_compression"]
    thresholds: _CompressionThresholds
    cases: list[_CompressionCase] = Field(min_length=1)


_SCHEMAS: dict[str, type[BaseModel]] = {
    "routing": _RoutingSuite,
    "rag": _RagSuite,
    "hallucination": _HallucinationSuite,
    "planning": _PlanningSuite,
    "quality_gate": _QualityGateSuite,
    "context_compression": _CompressionSuite,
}


def _format_validation_error(error: ValidationError) -> str:
    messages = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        messages.append(f"{location}: {item['msg']}")
    return "; ".join(messages)


def _validate_unique_case_ids(data: dict[str, Any], path: Path) -> None:
    seen: set[str] = set()
    for case in data.get("cases", []):
        case_id = case["id"]
        if case_id in seen:
            raise ValueError(f"Invalid golden suite '{path}': duplicate case id '{case_id}'")
        seen.add(case_id)


def validate_golden_suite(data: Any, path: Path) -> dict[str, Any]:
    """Validate one parsed suite and retain its original dictionary shape."""
    if not isinstance(data, dict):
        raise ValueError(f"Golden suite must contain a mapping: {path}")

    schema_name = data.get("kind") or data.get("suite")
    if not isinstance(schema_name, str):
        raise ValueError(
            f"Invalid golden suite '{path}': schema name must be a string"
        )
    schema = _SCHEMAS.get(schema_name)
    if schema is None:
        known = ", ".join(sorted(_SCHEMAS))
        raise ValueError(
            f"Invalid golden suite '{path}': unsupported schema '{schema_name}'. "
            f"Known schemas: {known}"
        )

    try:
        schema.model_validate(data)
    except ValidationError as exc:
        details = _format_validation_error(exc)
        raise ValueError(f"Invalid golden suite '{path}': {details}") from exc

    _validate_unique_case_ids(data, path)
    return data


def load_golden_suite(path: Path) -> dict[str, Any]:
    """Load and validate a YAML golden suite before evaluation starts."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in golden suite '{path}': {exc}") from exc

    suite = validate_golden_suite(data, path)
    suite["_path"] = str(path)
    return suite
