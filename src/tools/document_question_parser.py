"""Parse exam documents into question-level structures through MCP tools."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import zipfile
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from fastapi import HTTPException, UploadFile

from src.tools.ocr_tool import perform_exam_ocr_bytes
from src.tools.policy_search import (
    PolicySearchError,
    _call_http_mcp,
    _call_stdio_mcp,
)


ALLOWED_DOCUMENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
}
EXTENSION_KINDS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
}
DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_FILES = 12


@dataclass(frozen=True)
class ParsedQuestion:
    number: str
    subject: str | None
    stem: str
    options: list[str] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    detected_knowledge_points: list[str] = field(default_factory=list)
    answer: str | None = None
    analysis: str | None = None
    source_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentParseResult:
    questions: list[ParsedQuestion]
    recognized_text: str
    query: str
    filenames: list[str]
    parser: str
    segmenter_used: bool


class DocumentParseError(RuntimeError):
    """Raised when a configured document MCP cannot parse an upload."""


async def parse_exam_uploads(
    uploads: list[UploadFile],
    question: str | None = None,
) -> DocumentParseResult:
    """Parse one document or multiple exam images into normalized questions."""
    prepared = await _read_uploads(uploads)
    kinds = {item["kind"] for item in prepared}
    if len(kinds) != 1:
        raise HTTPException(
            status_code=400,
            detail="Upload one PDF/DOCX or a batch containing images only.",
        )
    kind = next(iter(kinds))
    if kind in {"pdf", "docx"} and len(prepared) != 1:
        raise HTTPException(status_code=400, detail="Only one PDF or DOCX can be parsed at a time.")

    try:
        payload, parser = await asyncio.to_thread(_parse_with_mcp, kind, prepared)
    except DocumentParseError as exc:
        if kind == "image":
            payload, parser = await _fallback_image_ocr(prepared)
        else:
            payload, parser = await asyncio.to_thread(_fallback_document_parse, kind, prepared)

    questions = _normalize_questions(payload)
    recognized_text = _extract_recognized_text(payload, questions)
    segmenter_used = False
    if not questions and recognized_text:
        segmented = await asyncio.to_thread(_segment_with_mcp, recognized_text, prepared)
        questions = _normalize_questions(segmented)
        segmenter_used = bool(questions)

    query = build_document_question_query(questions, recognized_text, question)
    return DocumentParseResult(
        questions=questions,
        recognized_text=recognized_text,
        query=query,
        filenames=[item["filename"] for item in prepared],
        parser=parser,
        segmenter_used=segmenter_used,
    )


def build_document_question_query(
    questions: list[ParsedQuestion],
    recognized_text: str,
    user_question: str | None = None,
) -> str:
    """Build a downstream tutoring query from structured exam content."""
    parts = [
        "我上传了试卷或学习材料，系统已将内容解析为题目级结构。",
        "请结合题目内容检索相关知识点、题型方法和相似材料，再按我的要求作答。",
    ]
    if (user_question or "").strip():
        parts.append(f"我的补充问题：{user_question.strip()}")
    if questions:
        compact = [
            {
                "number": item.number,
                "subject": item.subject,
                "stem": item.stem,
                "options": item.options,
                "figures": item.figures,
                "detected_knowledge_points": item.detected_knowledge_points,
            }
            for item in questions
        ]
        parts.append("题目结构：\n" + json.dumps(compact, ensure_ascii=False, indent=2))
    elif recognized_text.strip():
        parts.append("识别内容：\n" + recognized_text.strip())
    return "\n\n".join(parts)


async def _read_uploads(uploads: list[UploadFile]) -> list[dict[str, Any]]:
    if not uploads:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    max_files = _env_int("DOCUMENT_PARSE_MAX_FILES", DEFAULT_MAX_FILES, 1, 30)
    if len(uploads) > max_files:
        raise HTTPException(status_code=413, detail=f"Too many files. Max count is {max_files}.")

    max_bytes = _env_int(
        "DOCUMENT_PARSE_MAX_FILE_MB",
        DEFAULT_MAX_FILE_BYTES // 1024 // 1024,
        1,
        100,
    ) * 1024 * 1024
    prepared = []
    for upload in uploads:
        filename = upload.filename or "upload"
        content_type = upload.content_type or ""
        kind = ALLOWED_DOCUMENT_TYPES.get(content_type) or EXTENSION_KINDS.get(
            Path(filename).suffix.lower()
        )
        if not kind:
            raise HTTPException(
                status_code=415,
                detail="Only PDF, DOCX, JPEG, PNG, and WebP files are supported.",
            )
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {filename}")
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large: {filename}. Max size is {max_bytes // 1024 // 1024}MB.",
            )
        prepared.append({
            "filename": filename,
            "content_type": content_type,
            "kind": kind,
            "data": data,
        })
    return prepared


def _parse_with_mcp(kind: str, files: list[dict[str, Any]]) -> tuple[Any, str]:
    tool_env = {
        "pdf": ("PDF_PARSE_MCP_TOOL", "pdf_parse"),
        "docx": ("DOCX_PARSE_MCP_TOOL", "docx_parse"),
        "image": ("IMAGE_OCR_MCP_TOOL", "image_ocr_plus"),
    }
    env_name, default_tool = tool_env[kind]
    tool_name = os.getenv(env_name, default_tool)
    arguments = {
        "files": [
            {
                "filename": item["filename"],
                "content_type": item["content_type"],
                "data_base64": base64.b64encode(item["data"]).decode("ascii"),
            }
            for item in files
        ],
        "preserve_layout": True,
        "recognize_formulas": True,
        "include_figures": True,
        "output_format": "questions",
    }
    return _call_document_mcp(tool_name, arguments), f"mcp:{tool_name}"


def _segment_with_mcp(recognized_text: str, files: list[dict[str, Any]]) -> Any:
    tool_name = os.getenv("QUESTION_SEGMENTER_MCP_TOOL", "question_segmenter")
    arguments = {
        "text": recognized_text,
        "filenames": [item["filename"] for item in files],
        "output_format": "questions",
    }
    try:
        return _call_document_mcp(tool_name, arguments)
    except DocumentParseError:
        return None


def _fallback_document_parse(kind: str, files: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Extract text locally when no document MCP is available."""
    item = files[0]
    try:
        if kind == "pdf":
            text = _extract_pdf_text(item["data"])
            parser = "local:pymupdf"
        elif kind == "docx":
            text = _extract_docx_text(item["data"])
            parser = "local:docx"
        else:
            raise DocumentParseError(f"Local fallback does not support {kind}.")
    except DocumentParseError:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Local {kind.upper()} parsing failed: {exc}",
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                f"No text was extracted from {item['filename']}. "
                "For scanned documents, configure DOCUMENT_MCP_URL or "
                "DOCUMENT_MCP_COMMAND with OCR-capable tools."
            ),
        )
    return {"recognized_text": text}, parser


def _extract_pdf_text(data: bytes) -> str:
    import fitz

    pages = []
    with fitz.open(stream=data, filetype="pdf") as document:
        for page_number, page in enumerate(document, 1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"--- Page {page_number} ---\n{text}")
    return "\n\n".join(pages)


def _extract_docx_text(data: bytes) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(BytesIO(data)) as archive:
        document_xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(document_xml)
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(
            node.text or ""
            for node in paragraph.findall(".//w:t", namespace)
        ).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _call_document_mcp(tool_name: str, arguments: dict[str, Any]) -> Any:
    endpoint = os.getenv("DOCUMENT_MCP_URL")
    command = os.getenv("DOCUMENT_MCP_COMMAND")
    timeout = float(_env_int("DOCUMENT_MCP_TIMEOUT_SECONDS", 60, 1, 300))
    if not endpoint and not command:
        raise DocumentParseError(
            "Document MCP is not configured. Set DOCUMENT_MCP_URL or DOCUMENT_MCP_COMMAND."
        )
    try:
        if endpoint:
            return _call_http_mcp(endpoint, tool_name, arguments, timeout)
        return _call_stdio_mcp(command or "", tool_name, arguments, timeout)
    except PolicySearchError as exc:
        raise DocumentParseError(f"Document MCP tool {tool_name} failed: {exc}") from exc


async def _fallback_image_ocr(
    files: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    texts = []
    for item in files:
        result = await perform_exam_ocr_bytes(
            item["data"],
            item["content_type"],
            filename=item["filename"],
        )
        texts.append(f"## {item['filename']}\n{result.text}")
    return {"recognized_text": "\n\n".join(texts)}, "vision_ocr_fallback"


def _normalize_questions(payload: Any) -> list[ParsedQuestion]:
    container = _extract_payload(payload)
    raw_questions: Any = container.get("questions", []) if isinstance(container, dict) else []
    if not isinstance(raw_questions, list):
        return []

    questions = []
    for index, item in enumerate(raw_questions, 1):
        if not isinstance(item, dict):
            continue
        stem = str(item.get("stem") or item.get("question") or item.get("content") or "").strip()
        if not stem:
            continue
        questions.append(ParsedQuestion(
            number=str(item.get("number") or item.get("question_number") or index),
            subject=_optional_text(item.get("subject")),
            stem=stem,
            options=_string_list(item.get("options")),
            figures=_figure_list(item.get("figures") or item.get("images")),
            detected_knowledge_points=_string_list(
                item.get("detected_knowledge_points") or item.get("knowledge_points")
            ),
            answer=_optional_text(item.get("answer")),
            analysis=_optional_text(item.get("analysis") or item.get("explanation")),
            source_pages=_int_list(item.get("source_pages") or item.get("pages")),
        ))
    return questions


def _extract_recognized_text(payload: Any, questions: list[ParsedQuestion]) -> str:
    container = _extract_payload(payload)
    if isinstance(container, dict):
        text = container.get("recognized_text") or container.get("text") or container.get("content")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return "\n\n".join(f"{item.number}. {item.stem}" for item in questions)


def _extract_payload(payload: Any) -> Any:
    if payload is None:
        return {}
    if isinstance(payload, str):
        try:
            return _extract_payload(json.loads(payload))
        except json.JSONDecodeError:
            return {"recognized_text": payload}
    if isinstance(payload, list):
        return {"questions": payload}
    if not isinstance(payload, dict):
        return {}
    structured = payload.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    if isinstance(structured, list):
        return {"questions": structured}
    content = payload.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                extracted = _extract_payload(str(part.get("text") or ""))
                if extracted:
                    return extracted
    return payload


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _figure_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    figures = []
    for item in value:
        if isinstance(item, dict):
            figures.append(item)
        elif str(item).strip():
            figures.append({"reference": str(item).strip()})
    return figures


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))
