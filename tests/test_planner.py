"""Unit tests for SubGraph B — Study Planner nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.planner import search_policy


class TestSearchPolicy:

    @patch("src.graph.planner.web_search_fn")
    async def test_returns_search_results(self, mock_search):
        mock_search.return_value = [
            {"content": "2026年高考6月7日", "title": "高考时间", "url": "https://example.com"},
        ]

        state = {"messages": [HumanMessage(content="帮我做复习计划")]}
        result = await search_policy(state)

        assert "search_results" in result
        assert len(result["search_results"]) == 1
        assert result["policy_source"] == "web_fallback"
        mock_search.assert_called_once()

    @patch("src.graph.planner.web_search_fn", side_effect=Exception("timeout"))
    async def test_returns_empty_on_exception(self, mock_search):
        state = {"messages": [HumanMessage(content="test")]}
        result = await search_policy(state)

        assert result["search_results"] == []
        assert result["policy_source"] == "none"

    @patch("src.graph.planner.web_search_fn")
    async def test_query_contains_current_year(self, mock_search):
        mock_search.return_value = []
        from datetime import datetime

        state = {"messages": [HumanMessage(content="test")]}
        await search_policy(state)

        call_args = mock_search.call_args[0][0]
        assert str(datetime.now().year) in call_args

    @patch("src.graph.planner.web_search_fn")
    @patch("src.graph.planner.search_official_policy")
    async def test_prefers_official_policy_mcp(self, mock_official, mock_web):
        mock_official.return_value = [
            {
                "source": "广东省教育考试院",
                "url": "https://eea.gd.gov.cn/policy",
                "published_at": "2026-06-10",
                "province": "广东",
                "topic": "志愿填报",
                "content": "官方志愿填报安排",
                "confidence": "official",
            }
        ]

        state = {"messages": [HumanMessage(content="帮我做志愿规划")]}
        result = await search_policy(state)

        assert result["policy_source"] == "official_mcp"
        assert result["search_results"][0]["source"] == "广东省教育考试院"
        mock_official.assert_called_once()
        mock_web.assert_not_called()

    @patch("src.graph.planner.web_search_fn")
    @patch("src.graph.planner.search_official_policy")
    async def test_falls_back_to_web_when_official_empty(self, mock_official, mock_web):
        mock_official.return_value = []
        mock_web.return_value = [
            {"content": "fallback policy", "title": "web", "url": "https://example.com"},
        ]

        state = {"messages": [HumanMessage(content="帮我做志愿规划")]}
        result = await search_policy(state)

        assert result["policy_source"] == "web_fallback"
        assert result["search_results"][0]["content"] == "fallback policy"
        mock_official.assert_called_once()
        mock_web.assert_called_once()


