"""Regression tests for token-budgeted conversation compression."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from src.graph.builder import compress_messages
from src.memory.compressor import (
    AnchoredItem,
    CompressionResult,
    SessionEpisode,
    SessionTask,
    compress_conversation,
)


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
