"""Tests for compression harness metrics."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from src.memory.artifacts import ContextArtifactStore
from src.memory.compression_harness import (
    episode_from_case,
    evaluate_compression_result,
    messages_from_case,
    result_from_static_episode,
)


def test_compression_harness_scores_constraints_and_artifacts(tmp_path):
    store = ContextArtifactStore(tmp_path)
    artifact_id = "ctx_harness_exam_001"
    day_dir = tmp_path / "2026-06-29"
    day_dir.mkdir()
    (day_dir / f"{artifact_id}.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "kind": "document_parse",
                "payload": {"recognized_text": "第17题 导数 函数零点"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    before_messages = messages_from_case(
        [
            {
                "id": "h0",
                "role": "human",
                "content": (
                    "广东高三，数学目标120分，导数和函数零点很弱，"
                    "artifact_id 是 ctx_harness_exam_001。"
                    + "这是一段旧讲解过程，需要压缩但不能丢约束。 " * 80
                ),
            },
            {
                "id": "a0",
                "role": "assistant",
                "content": "第17题涉及导数、函数零点。" + "旧推导步骤可以被摘要。 " * 80,
            },
            {"id": "h1", "role": "human", "content": "继续讲知识点。"},
        ]
    )
    episode = episode_from_case(
        {
            "gaokao_state": {
                "grade": "高三",
                "province": "广东",
                "target_score": "数学120分",
                "weak_points": [{"text": "导数和函数零点很弱"}],
            },
            "artifact_refs": [
                {
                    "artifact_id": artifact_id,
                    "kind": "document_parse",
                    "preview": "第17题 导数 函数零点",
                }
            ],
            "constraints": [{"text": "数学目标120分"}],
            "knowledge_progress": [{"text": "第17题涉及导数、函数零点"}],
        }
    )
    result = result_from_static_episode(
        before_messages=before_messages,
        episode=episode,
        recent_messages=[HumanMessage(content="继续讲知识点。")],
    )

    metrics = evaluate_compression_result(
        case_id="case",
        before_messages=before_messages,
        result=result,
        expected_constraints=["广东", "高三", "数学目标120分"],
        answer_terms=["导数", "函数零点", "第17题"],
        expected_artifact_ids=[artifact_id],
        thresholds={
            "token_reduction": 0.0,
            "constraint_retention": 1.0,
            "answer_consistency": 1.0,
            "artifact_recoverability": 1.0,
        },
        store=store,
    )

    assert metrics.constraint_retention == 1.0
    assert metrics.answer_consistency == 1.0
    assert metrics.artifact_recoverability == 1.0
    assert metrics.passed is True
