"""Regression tests for token-budgeted conversation compression."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from src.graph.builder import compress_messages
from src.memory.compressor import (
    AnchoredItem,
    ArtifactReference,
    CompressionResult,
    CurrentDocument,
    CurrentQuestion,
    GaokaoLearningState,
    SessionEpisode,
    SessionTask,
    compress_conversation,
)
from src.memory.context_builder import build_memory_context, build_node_context


def _long_history(turns: int = 8) -> list:
    messages = []
    for index in range(turns):
        messages.extend(
            [
                HumanMessage(
                    id=f"h{index}",
                    content=f"第{index}轮：我的目标是数学120分。" + "导数题分析。" * 40,
                ),
                AIMessage(
                    id=f"a{index}",
                    content=f"第{index}轮讲解。" + "先看定义域和单调区间。" * 40,
                ),
            ]
        )
    return messages


@pytest.mark.asyncio
async def test_compression_keeps_recent_turns_and_structured_summary():
    episode = SessionEpisode(
        task=SessionTask(intent="academic", subject="math", topic="导数"),
        constraints=[
            AnchoredItem(text="目标数学120分", source_message_ids=["h0"])
        ],
    )
    with patch(
        "src.memory.compressor._summarize_episode",
        new=AsyncMock(return_value=episode),
    ):
        result = await compress_conversation(
            _long_history(),
            recent_turns=2,
            soft_limit_tokens=100,
        )

    assert result.compressed is True
    assert result.summary_json
    assert isinstance(result.messages[0], SystemMessage)
    assert "[会话摘要]" in result.messages[0].content
    assert any("第7轮" in message.content for message in result.messages)
    assert not any("第0轮" in message.content for message in result.messages[1:])


@pytest.mark.asyncio
async def test_graph_node_deletes_old_messages_before_replacement():
    replacement = [
        SystemMessage(id="summary", content="[会话摘要]\n压缩结果"),
        HumanMessage(id="latest", content="继续讲"),
    ]
    compression_result = CompressionResult(
        messages=replacement,
        summary_json='{"task":{"intent":"academic"}}',
        before_tokens=1000,
        after_tokens=100,
        compressed=True,
    )
    state = {
        "messages": _long_history(3),
        "session_summary": "",
        "compression_count": 0,
    }
    with patch(
        "src.memory.compressor.compress_conversation",
        new=AsyncMock(return_value=compression_result),
    ):
        update = await compress_messages(state)

    assert isinstance(update["messages"][0], RemoveMessage)
    assert update["messages"][0].id == REMOVE_ALL_MESSAGES
    merged = add_messages(state["messages"], update["messages"])
    assert [message.id for message in merged] == ["summary", "latest"]
    assert update["compression_count"] == 1
    assert update["compression_before_tokens"] == 1000
    assert update["compression_after_tokens"] == 100


@pytest.mark.asyncio
async def test_no_update_below_budget():
    state = {
        "messages": [HumanMessage(content="你好")],
        "session_summary": "",
    }
    result = await compress_messages(state)
    assert result == {}


def test_session_episode_keeps_gaokao_state_and_artifact_refs():
    episode = SessionEpisode(
        task=SessionTask(intent="academic", subject="math", topic="导数"),
        gaokao_state=GaokaoLearningState(
            grade="高三",
            province="广东",
            exam_track="物化生",
            target_score="数学120分",
            weak_points=[AnchoredItem(text="导数和函数零点很弱", source_message_ids=["h0"])],
        ),
        artifact_refs=[
            ArtifactReference(
                artifact_id="ctx_test_exam_001",
                kind="document_parse",
                source="exam.pdf",
                preview="第17题导数",
            )
        ],
        current_documents=[
            CurrentDocument(
                artifact_id="ctx_test_exam_001",
                filename="exam.pdf",
                parser="document_parse",
                summary="第17题导数与函数零点",
                question_numbers=["17"],
                knowledge_points=["导数", "函数零点"],
            )
        ],
        current_questions=[
            CurrentQuestion(
                number="17",
                subject="math",
                stem_preview="讨论函数零点个数",
                knowledge_points=["导数"],
                status="needs_followup",
                artifact_id="ctx_test_exam_001",
            )
        ],
    )

    restored = SessionEpisode.model_validate_json(episode.model_dump_json())
    prompt_text = restored.prompt_text()

    assert restored.gaokao_state.province == "广东"
    assert restored.artifact_refs[0].artifact_id == "ctx_test_exam_001"
    assert "高考学习状态" in prompt_text
    assert "ctx_test_exam_001" in prompt_text
    assert "第17题导数与函数零点" in prompt_text


def test_build_memory_context_includes_extended_session_fields():
    episode = SessionEpisode(
        gaokao_state=GaokaoLearningState(
            province="广东",
            target_score="数学120分",
            study_preferences=[AnchoredItem(text="先思路后步骤")],
        ),
        artifact_refs=[
            ArtifactReference(
                artifact_id="ctx_test_exam_002",
                kind="document_parse",
                preview="第17题",
            )
        ],
        current_questions=[
            CurrentQuestion(
                number="17",
                subject="math",
                stem_preview="函数零点",
                artifact_id="ctx_test_exam_002",
            )
        ],
    )

    context = build_memory_context({"session_summary": episode.model_dump_json()})

    assert "广东" in context
    assert "数学120分" in context
    assert "先思路后步骤" in context
    assert "ctx_test_exam_002" in context
    assert "函数零点" in context


def test_build_node_context_projects_sections_by_consumer():
    episode = SessionEpisode(
        task=SessionTask(intent="academic", subject="math", topic="导数"),
        gaokao_state=GaokaoLearningState(
            province="广东",
            target_score="数学120分",
            weak_points=[AnchoredItem(text="导数零点")],
            study_preferences=[AnchoredItem(text="先思路后步骤")],
        ),
        artifact_refs=[
            ArtifactReference(
                artifact_id="ctx_projection_001",
                kind="document_parse",
                preview="第17题",
            )
        ],
        current_questions=[
            CurrentQuestion(
                number="17",
                subject="math",
                stem_preview="函数零点",
                artifact_id="ctx_projection_001",
            )
        ],
        student_state=[AnchoredItem(text="最近有些焦虑")],
        constraints=[AnchoredItem(text="每天最多学习两小时")],
        knowledge_progress=[AnchoredItem(text="已掌握单调性")],
    )
    state = {
        "long_term_memory": "[长期记忆]\n- 喜欢例题驱动",
        "session_summary": episode.model_dump_json(),
    }

    emotional = build_node_context(state, "emotional_response")
    assert "最近有些焦虑" in emotional
    assert "先思路后步骤" in emotional
    assert "ctx_projection_001" not in emotional
    assert "已掌握单调性" not in emotional

    academic = build_node_context(state, "generate_answer")
    assert "ctx_projection_001" in academic
    assert "函数零点" in academic
    assert "已掌握单调性" in academic
    assert "最近有些焦虑" not in academic

    assert build_node_context(state, "unregistered_node") == build_memory_context(state)


def test_build_node_context_keeps_legacy_free_text_summary():
    context = build_node_context(
        {"session_summary": "旧版摘要：学生希望先讲基础。"},
        "generate_answer",
    )

    assert "旧版摘要：学生希望先讲基础。" in context
