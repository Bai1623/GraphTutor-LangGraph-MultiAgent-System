"""Multimodal OCR helpers for exam-paper image uploads."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, UploadFile

from src.security.upload_security import validate_upload_security

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class OcrResult:
    text: str
    query: str
    filename: str | None
    content_type: str


def _get_max_image_bytes() -> int:
    raw = os.getenv("OCR_MAX_IMAGE_MB", "8")
    try:
        mb = int(raw)
    except ValueError:
        mb = 8
    return max(1, min(mb, 25)) * 1024 * 1024


def _build_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def build_exam_ocr_query(ocr_text: str, question: str | None = None) -> str:
    """Build the downstream RAG query from OCR text and optional user intent."""
    cleaned_text = ocr_text.strip()
    cleaned_question = (question or "").strip()

    parts = [
        "我上传了一张高考试卷/题目图片，以下是 OCR 识别出的内容。",
        "请先结合题目内容检索相关知识点、题型方法和相似材料，再给出清晰解答。",
    ]
    if cleaned_question:
        parts.append(f"我的补充问题：{cleaned_question}")
    parts.append(f"OCR 识别内容：\n{cleaned_text}")
    return "\n\n".join(parts)


async def read_image_upload(upload: UploadFile) -> tuple[bytes, str]:
    max_bytes = _get_max_image_bytes()
    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Max size is {max_bytes // 1024 // 1024}MB.",
        )
    metadata = await validate_upload_security(
        data=image_bytes,
        filename=upload.filename,
        content_type=upload.content_type,
        expected_kind="image",
    )
    return image_bytes, metadata.content_type


def get_ocr_llm() -> Any:
    api_key = os.getenv("OCR_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "OCR_API_KEY is not configured. Set OCR_API_KEY/OCR_BASE_URL/OCR_MODEL "
                "to an OpenAI-compatible vision model before uploading exam images."
            ),
        )

    base_url = os.getenv("OCR_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("OCR_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OCR_BASE_URL is required when OCR_API_KEY is set.",
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OCR_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        streaming=False,
        timeout=float(os.getenv("OCR_TIMEOUT_SECONDS", "60")),
    )


async def perform_exam_ocr(upload: UploadFile, question: str | None = None) -> OcrResult:
    image_bytes, content_type = await read_image_upload(upload)
    return await perform_exam_ocr_bytes(
        image_bytes,
        content_type,
        filename=upload.filename,
        question=question,
    )


async def perform_exam_ocr_bytes(
    image_bytes: bytes,
    content_type: str,
    *,
    filename: str | None = None,
    question: str | None = None,
) -> OcrResult:
    """Run the existing vision OCR fallback on already-read image bytes."""
    data_url = _build_data_url(image_bytes, content_type)
    from langchain_core.messages import HumanMessage

    prompt = (
        "你是高考试卷 OCR 助手。请从图片中尽可能完整、准确地转写文字、题号、选项、公式、"
        "表格和图形中的关键标注。只输出识别结果，不要解题，不要添加图片中不存在的内容。"
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    try:
        response = await get_ocr_llm().ainvoke([message])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OCR model request failed: {exc}") from exc

    text = str(response.content or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="OCR did not return any text.")

    return OcrResult(
        text=text,
        query=build_exam_ocr_query(text, question),
        filename=filename,
        content_type=content_type,
    )
