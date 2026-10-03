"""Tests for compression harness metrics."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from scripts import run_compression_harness
from src.memory.artifacts import ContextArtifactStore
from src.memory.compression_harness import (
    LiveAnswerComparison,
    answer_generation_messages,
    answer_judge_messages,
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


def test_load_compression_suite_rejects_missing_expected_episode(tmp_path):
    path = tmp_path / "invalid-compression.yaml"
    path.write_text(
        """
suite: context_compression
description: Invalid compression fixture.
thresholds:
  token_reduction: 0.2
cases:
  - id: missing_expected_episode
    messages:
      - id: h0
        role: human
        content: 请继续讲题
    recent_message_count: 1
    expected_constraints: []
    answer_terms: []
    expected_artifact_ids: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"cases\.0\.expected_episode"):
        run_compression_harness._load_suite(path)


def test_live_answer_prompts_keep_question_and_rubric_terms():
    generation = answer_generation_messages(
        context_text="学生要求先讲思路",
        question="第17题考什么？",
    )
    judge = answer_judge_messages(
        question="第17题考什么？",
        expected_constraints=["先讲思路"],
        answer_terms=["导数", "函数零点"],
        baseline_answer="涉及导数和函数零点。",
        compressed_answer="先讲思路：涉及导数和函数零点。",
    )

    assert "学生要求先讲思路" in generation[-1].content
    assert "第17题考什么" in generation[-1].content
    assert "约束遵循" in judge[-1].content
    assert "函数零点" in judge[-1].content


@pytest.mark.asyncio
async def test_compare_live_answers_returns_answers_and_structured_score():
    answer_llm = MagicMock()
    answer_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="基线回答：导数和函数零点。"),
            AIMessage(content="压缩后回答：先讲思路，再讲导数和函数零点。"),
        ]
    )
    structured_judge = MagicMock()
    structured_judge.ainvoke = AsyncMock(
        return_value=LiveAnswerComparison(
            factual_consistency=1.0,
            constraint_adherence=1.0,
            usefulness_retention=0.9,
            overall_score=0.95,
            regression_detected=False,
            reason="关键信息和回答约束均保留。",
        )
    )
    judge_llm = MagicMock()
    judge_llm.with_structured_output.return_value = structured_judge

    messages = [HumanMessage(content="第17题考什么？")]
    with patch(
        "scripts.run_compression_harness.get_node_llm",
        side_effect=[answer_llm, judge_llm],
    ):
        result = await run_compression_harness._compare_live_answers(
            before_messages=messages,
            after_messages=messages,
            expected_constraints=["先讲思路"],
            answer_terms=["导数", "函数零点"],
        )

    assert result["overall_score"] == 0.95
    assert result["regression_detected"] is False
    assert "基线回答" in result["baseline_answer"]
    assert "压缩后回答" in result["compressed_answer"]
    judge_llm.with_structured_output.assert_called_once_with(LiveAnswerComparison)


@pytest.mark.asyncio
async def test_live_judge_regression_fails_case_even_with_high_score(tmp_path):
    case = {
        "id": "judge_regression",
        "messages": [
            {"id": "h0", "role": "human", "content": "请记住必须先讲思路。"},
            {"id": "h1", "role": "human", "content": "继续回答。"},
        ],
        "recent_message_count": 1,
        "expected_constraints": ["先讲思路"],
        "answer_terms": [],
        "expected_artifact_ids": [],
        "expected_episode": {
            "constraints": [{"text": "必须先讲思路", "source_message_ids": ["h0"]}]
        },
    }
    comparison = {
        "factual_consistency": 1.0,
        "constraint_adherence": 0.5,
        "usefulness_retention": 1.0,
        "overall_score": 0.9,
        "regression_detected": True,
        "reason": "压缩后回答没有先讲思路。",
        "question": "继续回答。",
        "baseline_answer": "先讲思路。",
        "compressed_answer": "直接给答案。",
    }

    with patch(
        "scripts.run_compression_harness._compare_live_answers",
        new=AsyncMock(return_value=comparison),
    ):
        result = await run_compression_harness._run_case(
            case,
            thresholds={
                "token_reduction": 0.0,
                "constraint_retention": 1.0,
                "answer_consistency": 1.0,
                "artifact_recoverability": 1.0,
                "live_answer_quality": 0.8,
            },
            store=ContextArtifactStore(tmp_path),
            use_llm=False,
            compare_answers=True,
        )

    assert result["live_answer_quality"] == 0.9
    assert result["passed"] is False
