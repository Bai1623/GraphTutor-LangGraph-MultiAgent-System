"""Tests for official policy/admission MCP search normalization."""

from __future__ import annotations

import json
from unittest.mock import patch

from src.tools import policy_search


def test_returns_empty_without_mcp_configuration(monkeypatch):
    monkeypatch.delenv("POLICY_MCP_URL", raising=False)
    monkeypatch.delenv("POLICY_MCP_COMMAND", raising=False)

    assert policy_search.search_official_policy("2026 高考政策") == []


def test_normalizes_structured_results():
    payload = {
        "structuredContent": [
            {
                "source": "广东省教育考试院",
                "url": "https://eea.gd.gov.cn/a",
                "published_at": "2026-06-10",
                "province": "广东",
                "topic": "志愿填报",
                "content": "官方政策内容",
                "confidence": "official",
            }
        ]
    }

    results = policy_search._normalize_policy_results(
        payload,
        default_province=None,
        default_topic=None,
    )

    assert results == [
        {
            "source": "广东省教育考试院",
            "url": "https://eea.gd.gov.cn/a",
            "published_at": "2026-06-10",
            "province": "广东",
            "topic": "志愿填报",
            "content": "官方政策内容",
            "confidence": "official",
        }
    ]


def test_normalizes_mcp_text_content_json():
    payload = {
        "content": [
            {
                "type": "text",
                "text": json.dumps({
                    "results": [
                        {
                            "title": "教育部通知",
                            "link": "https://www.moe.gov.cn/a",
                            "date": "2026-01-01",
                            "text": "通知正文",
                        }
                    ]
                }, ensure_ascii=False),
            }
        ]
    }

    results = policy_search._normalize_policy_results(
        payload,
        default_province="全国",
        default_topic="高考政策",
    )

    assert results[0]["source"] == "教育部通知"
    assert results[0]["url"] == "https://www.moe.gov.cn/a"
    assert results[0]["published_at"] == "2026-01-01"
    assert results[0]["province"] == "全国"
    assert results[0]["topic"] == "高考政策"
    assert results[0]["content"] == "通知正文"
    assert results[0]["confidence"] == "official"


def test_search_official_policy_uses_http_mcp(monkeypatch):
    monkeypatch.setenv("POLICY_MCP_URL", "https://mcp.example.com")
    monkeypatch.delenv("POLICY_MCP_COMMAND", raising=False)
    mcp_payload = {
        "structuredContent": [
            {
                "source": "省考试院",
                "url": "https://example.com",
                "content": "一分一段表发布",
            }
        ]
    }

    with patch("src.tools.policy_search._call_http_mcp", return_value=mcp_payload) as mock_call:
        results = policy_search.search_official_policy(
            "一分一段表",
            province="广东",
            topic="一分一段表",
            limit=3,
        )

    assert results[0]["source"] == "省考试院"
    assert results[0]["province"] == "广东"
    assert results[0]["topic"] == "一分一段表"
    mock_call.assert_called_once()


def test_format_policy_results():
    formatted = policy_search.format_policy_results([
        {
            "source": "广东省教育考试院",
            "url": "https://eea.gd.gov.cn/a",
            "published_at": "2026-06-10",
            "province": "广东",
            "topic": "志愿填报",
            "content": "官方政策内容",
            "confidence": "official",
        }
    ])

    assert "广东省教育考试院" in formatted
    assert "official" in formatted
    assert "官方政策内容" in formatted


def test_stdio_response_parser_finds_tools_call_result():
    output = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "ok"}]}}),
    ])

    response = policy_search._find_json_rpc_response(output, 2)

    assert response["result"]["content"][0]["text"] == "ok"
