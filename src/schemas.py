"""Data structures that can be reused across modules."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(max_length=4096)
    thread_id: str | None = None
    user_id: str | None = Field(default=None, max_length=128)


class ResumeRequest(BaseModel):
    """Resume a graph interrupted by Human-in-the-loop."""

    thread_id: str
    edited_plan: str = Field(default="", max_length=16384)
    feedback: str | None = Field(default=None, max_length=4096)


class FeedbackRequest(BaseModel):
    """用户对AI回复的质量反馈（👍/👎）"""

    message_id: str
    rating: str  # "up" 或 "down"
    query_preview: str = Field(default="", max_length=500)


class OcrResponse(BaseModel):
    """OCR result returned before the recognized text enters the chat/RAG flow."""

    recognized_text: str
    query: str
    filename: str | None = None
    content_type: str


class ParsedQuestionResponse(BaseModel):
    """A normalized question extracted from an exam document."""

    number: str
    subject: str | None = None
    stem: str
    options: list[str] = Field(default_factory=list)
    figures: list[dict] = Field(default_factory=list)
    detected_knowledge_points: list[str] = Field(default_factory=list)
    answer: str | None = None
    analysis: str | None = None
    source_pages: list[int] = Field(default_factory=list)


class DocumentParseResponse(BaseModel):
    """Structured result returned by the document-question MCP pipeline."""

    questions: list[ParsedQuestionResponse]
    recognized_text: str
    query: str
    filenames: list[str]
    parser: str
    segmenter_used: bool
