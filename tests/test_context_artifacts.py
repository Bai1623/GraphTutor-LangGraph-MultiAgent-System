"""Tests for recoverable context artifact compaction."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import UploadFile
from langchain_core.messages import HumanMessage

from src.graph import academic, planner
from src.memory.artifacts import ContextArtifactStore, compact_with_artifact
from src.tools import document_question_parser as parser


def test_context_artifact_store_writes_payload_and_returns_preview(tmp_path):
    store = ContextArtifactStore(tmp_path)
    payload = {"content": "导数题分析" * 200, "source": "exam.pdf"}

    ref = store.put(
        kind="rag_retrieval_doc",
        payload=payload,
        preview_source=payload["content"],
        max_preview_chars=80,
    )

    assert ref.artifact_id.startswith("ctx_")
    assert len(ref.preview) <= 80
    saved = json.loads(next(tmp_path.glob(f"**/{ref.artifact_id}.json")).read_text(encoding="utf-8"))
    assert saved["payload"] == payload


def test_compact_with_artifact_replaces_content_with_preview(tmp_path):
    store = ContextArtifactStore(tmp_path)
    item = {"type": "rag", "content": "完整内容" * 200, "score": 0.9}

    with patch("src.memory.artifacts.get_context_artifact_store", return_value=store):
        compact = compact_with_artifact(
            item,
            kind="rag_retrieval_doc",
            preview_chars=60,
        )

    assert compact["recoverable"] is True
    assert compact["artifact_id"].startswith("ctx_")
    assert len(compact["content"]) <= 60
    assert compact["full_content_chars"] == len(item["content"])


@pytest.mark.asyncio
async def test_rag_retrieve_stores_full_doc_as_artifact(tmp_path):
    store = ContextArtifactStore(tmp_path)
    full_content = "判别式用法" * 200
    state = {
        "messages": [HumanMessage(content="判别式怎么用")],
        "keypoints": ["判别式"],
        "subject": "math",
    }

    with (
        patch.object(
            academic,
            "retrieve",
            return_value={"docs": [{"content": full_content, "source": "math.pdf", "score": 0.9}]},
        ),
        patch("src.memory.artifacts.get_context_artifact_store", return_value=store),
    ):
        result = await academic.rag_retrieve(state)

    item = result["context"][0]
    assert item["type"] == "rag"
    assert item["content"] != full_content
    assert item["artifact_id"].startswith("ctx_")
    saved = json.loads(next(tmp_path.glob(f"**/{item['artifact_id']}.json")).read_text(encoding="utf-8"))
    assert saved["payload"]["content"] == full_content


@pytest.mark.asyncio
async def test_search_policy_compacts_official_results(tmp_path):
    store = ContextArtifactStore(tmp_path)
    full_content = "官方志愿填报安排" * 200

    with (
        patch.object(
            planner,
            "search_official_policy",
            return_value=[{
                "source": "广东省教育考试院",
                "url": "https://example.edu.cn",
                "published_at": "2026-06-10",
                "province": "广东",
                "topic": "志愿填报",
                "content": full_content,
                "confidence": "official",
            }],
        ),
        patch("src.memory.artifacts.get_context_artifact_store", return_value=store),
    ):
        result = await planner.search_policy({"messages": [HumanMessage(content="规划志愿")]})

    item = result["search_results"][0]
    assert item["content"] != full_content
    assert item["artifact_id"].startswith("ctx_")
    assert result["policy_source"] == "official_mcp"


@pytest.mark.asyncio
async def test_document_parse_returns_preview_and_artifact_ref(tmp_path):
    store = ContextArtifactStore(tmp_path)
    upload = UploadFile(
        filename="exam.pdf",
        file=_BytesFile(b"%PDF-1.4\n"),
        headers={"content-type": "application/pdf"},
    )
    full_text = "第1题 求函数零点。" * 200

    with (
        patch("src.tools.document_question_parser.get_context_artifact_store", return_value=store),
        patch.object(parser, "_extract_pdf_text", return_value=full_text),
        patch.object(parser, "_segment_with_mcp", return_value=None),
    ):
        result = await parser.parse_exam_uploads([upload], "这个文件的知识点是什么")

    assert result.artifact_id.startswith("ctx_")
    assert len(result.recognized_text) < len(full_text)
    assert "文档 artifact_id" in result.query
    assert "识别内容预览" in result.query
    saved = json.loads(next(tmp_path.glob(f"**/{result.artifact_id}.json")).read_text(encoding="utf-8"))
    assert saved["payload"]["recognized_text"] == full_text


class _BytesFile:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            data, self._data = self._data, b""
            return data
        data, self._data = self._data[:size], self._data[size:]
        return data

    def close(self) -> None:
        pass
