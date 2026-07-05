"""试卷章节感知分块器（SectionAwareSplitter）—— 中文试卷专用

标准的 RecursiveCharacterTextSplitter 按固定字符数切分文本，
但对于高考试卷这种有明确标题结构的文档，存在两个问题：
1. 会在段落中间切断（"一、现代文阅读" 和正文被分割到不同块）
2. 无法保留章节标题信息（不知道这段来自哪个大题）

SectionAwareSplitter 的解决方案：
1. 用正则识别中文章节标题（"一、"、"二．"、"三." 等）
2. 按标题将试卷分为独立的大题
3. 在每个大题内部再用 RecursiveCharacterTextSplitter 子切分
4. 每个 chunk 携带 section_title 字段，保留章节归属

正则匹配规则：
  ^([一二三四五六七八九十]+[、.．]\s*.+)
  匹配行首的"一、标题"、"二．内容"、"三.文本"等格式

面试追问点：
- 为什么不用 PDF 的章节结构？
  答：大多数高考 PDF 是扫描版或纯文本嵌入，没有原生章节结构。
  正则匹配标题是最通用的方案。
- 如果试卷没有章节标题（如英语阅读理解）？
  答：正则找不到匹配，回退到整个文本作为一个 section，
  内部仍使用 RecursiveCharacterTextSplitter 切分。
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 匹配中文一级标题，如：
#   一、现代文阅读   二．填空题   三.选择题
# 不匹配二级标题如 （一）（1）
SECTION_PATTERN = re.compile(
    r"^([一二三四五六七八九十]+[、.．]\s*.+)",
    re.MULTILINE,
)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200


class SectionAwareSplitter:
    """按章节标题切分，并在每个 chunk 中保留标题元数据。

    与 RecursiveCharacterTextSplitter 的接口兼容，
    可以作为 loader.py 中 load_documents() 的参数直接替换默认切分器。

    使用方式：
        splitter = SectionAwareSplitter()
        docs = load_documents("data/chinese", subject="chinese",
                              doc_type="exam_paper", splitter=splitter)
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    # ------------------------------------------------------------------
    # 公共 API（与 RecursiveCharacterTextSplitter 接口兼容）
    # ------------------------------------------------------------------

    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[Document]:
        """按章节标题切分文本列表。

        参数与 RecursiveCharacterTextSplitter.create_documents 一致，
        因此可以作为 loader.py 中的 drop-in 替换。

        处理逻辑：
        1. 对每篇文本，按标题正则拆分为 (title, body) 列表
        2. 对每个 (title, body)，在 body 内部按字符数子切分
        3. 每个子 chunk 携带 base_meta + {section_title: title}
        """
        all_chunks: list[Document] = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            base_meta = metadatas[i] if metadatas else {}
            sections = self._split_into_sections(text)
            for title, body in sections:
                if not body.strip():
                    continue
                chunk_meta = {**base_meta, "section_title": title}
                chunks = self._splitter.create_documents(
                    texts=[body],
                    metadatas=[chunk_meta],
                )
                all_chunks.extend(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """将文本按标题拆分为 (title, body) 对列表。

        算法：
        1. 用 SECTION_PATTERN 找到所有标题位置
        2. 第一个标题之前的内容为"前导文本"，合并到第一个 section
        3. 每个标题到下一个标题之前的内容为其 body
        4. 没有标题时，返回 [("", full_text)]

        示例：
            输入："这是一段前言。一、现代文阅读\\n文章内容...\\n二、古诗文阅读\\n古诗内容..."
            输出：[("一、现代文阅读", "文章内容..."), ("二、古诗文阅读", "古诗内容...")]
            注意："这是一段前言"会被合并到第一个 section 的 body 中
        """
        matches = list(SECTION_PATTERN.finditer(text))

        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()

        for idx, match in enumerate(matches):
            title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            # 前导文本合并到第一个 section
            if idx == 0 and preamble:
                body = preamble + "\n" + body

            sections.append((title, body))

        return sections
