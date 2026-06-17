from __future__ import annotations

from src.tools.ocr_tool import build_exam_ocr_query


def test_build_exam_ocr_query_includes_ocr_text_and_question():
    query = build_exam_ocr_query("1. 已知函数 f(x)=x^2，求导数。", "请讲解解题步骤")

    assert "OCR 识别内容" in query
    assert "已知函数" in query
    assert "我的补充问题：请讲解解题步骤" in query
    assert "检索相关知识点" in query


def test_build_exam_ocr_query_omits_blank_question():
    query = build_exam_ocr_query("阅读下面的材料，完成题目。", "  ")

    assert "我的补充问题" not in query
    assert "阅读下面的材料" in query
