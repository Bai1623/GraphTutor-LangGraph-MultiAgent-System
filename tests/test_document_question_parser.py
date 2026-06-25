"""Tests for the document-to-question MCP pipeline."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile

from src.tools import document_question_parser as parser


def test_normalizes_structured_questions():
    payload = {
        "structuredContent": {
            "questions": [
                {
                    "number": 17,
                    "subject": "math",
                    "stem": "已知函数 f(x)，求其零点。",
                    "options": [],
                    "figures": [{"page": 3, "bbox": [1, 2, 3, 4]}],
                    "detected_knowledge_points": ["导数", "函数零点"],
                    "source_pages": [3],
                }
            ]
        }
    }

    questions = parser._normalize_questions(payload)

    assert questions[0].number == "17"
    assert questions[0].subject == "math"
    assert questions[0].detected_knowledge_points == ["导数", "函数零点"]
    assert questions[0].figures[0]["page"] == 3
    assert questions[0].source_pages == [3]


def test_normalizes_text_content_json():
    payload = {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "questions": [{"question_number": "2", "question": "选择正确的一项", "options": ["A", "B"]}]
            }, ensure_ascii=False),
        }]
    }

    questions = parser._normalize_questions(payload)

    assert questions[0].number == "2"
    assert questions[0].stem == "选择正确的一项"
    assert questions[0].options == ["A", "B"]


def test_pdf_routes_to_configured_mcp_tool(monkeypatch):
    monkeypatch.setenv("DOCUMENT_MCP_URL", "https://mcp.example.com")
    monkeypatch.setenv("PDF_PARSE_MCP_TOOL", "custom_pdf_parse")
    files = [{
        "filename": "exam.pdf",
        "content_type": "application/pdf",
        "kind": "pdf",
        "data": b"%PDF",
    }]

    with patch.object(parser, "_call_http_mcp", return_value={"questions": []}) as call:
        payload, backend = parser._parse_with_mcp("pdf", files)

    assert payload == {"questions": []}
    assert backend == "mcp:custom_pdf_parse"
    assert call.call_args.args[1] == "custom_pdf_parse"
    assert call.call_args.args[2]["files"][0]["data_base64"] == "JVBERg=="


@pytest.mark.asyncio
async def test_multiple_images_use_ocr_fallback_when_mcp_fails():
    uploads = [
        UploadFile(filename="page1.png", file=_BytesFile(b"one"), headers={"content-type": "image/png"}),
        UploadFile(filename="page2.png", file=_BytesFile(b"two"), headers={"content-type": "image/png"}),
    ]
    fallback_result = type("Ocr", (), {"text": "第1题 内容"})()

    with (
        patch.object(parser, "_parse_with_mcp", side_effect=parser.DocumentParseError("offline")),
        patch.object(parser, "perform_exam_ocr_bytes", new=AsyncMock(return_value=fallback_result)) as ocr,
        patch.object(parser, "_segment_with_mcp", return_value=None),
    ):
        result = await parser.parse_exam_uploads(uploads, "讲解这些题")

    assert result.parser == "vision_ocr_fallback"
    assert "page1.png" in result.recognized_text
    assert "讲解这些题" in result.query
    assert ocr.await_count == 2


def test_build_query_prefers_question_structure():
    question = parser.ParsedQuestion(
        number="17",
        subject="math",
        stem="求函数零点",
        detected_knowledge_points=["函数零点"],
    )

    query = parser.build_document_question_query([question], "raw", "给出步骤")

    assert '"number": "17"' in query
    assert "函数零点" in query
    assert "我的补充问题：给出步骤" in query
    assert "识别内容" not in query


def test_docx_local_fallback_extracts_paragraphs():
    buffer = BytesIO()
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>第1题</w:t></w:r><w:r><w:t> 求函数零点</w:t></w:r></w:p>
        <w:p><w:r><w:t>第2题 选择正确答案</w:t></w:r></w:p>
      </w:body>
    </w:document>"""
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    text = parser._extract_docx_text(buffer.getvalue())

    assert text == "第1题 求函数零点\n第2题 选择正确答案"


def test_document_fallback_returns_local_parser():
    files = [{
        "filename": "notes.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "kind": "docx",
        "data": b"docx",
    }]

    with patch.object(parser, "_extract_docx_text", return_value="第1题 内容"):
        payload, backend = parser._fallback_document_parse("docx", files)

    assert payload["recognized_text"] == "第1题 内容"
    assert backend == "local:docx"


@pytest.mark.asyncio
async def test_pdf_uses_local_fallback_without_mcp(monkeypatch):
    monkeypatch.delenv("DOCUMENT_MCP_URL", raising=False)
    monkeypatch.delenv("DOCUMENT_MCP_COMMAND", raising=False)
    upload = UploadFile(
        filename="exam.pdf",
        file=_BytesFile(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with (
        patch.object(parser, "_extract_pdf_text", return_value="--- Page 1 ---\n第1题 内容"),
        patch.object(parser, "_segment_with_mcp", return_value=None),
    ):
        result = await parser.parse_exam_uploads([upload], "讲解")

    assert result.parser == "local:pymupdf"
    assert "第1题" in result.recognized_text
    assert "讲解" in result.query


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
