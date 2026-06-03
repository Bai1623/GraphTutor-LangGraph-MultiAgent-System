"""文档加载器——PDF/Markdown/TXT → 分块后的 LangChain Documents

负责将原始的高考试卷文件加载到内存，并切分成适合检索的文本块（chunks）。
支持三种文件格式：PDF（PyMuPDF 解析）、Markdown、纯文本。

切分策略：
- 普通文档：RecursiveCharacterTextSplitter（1000字符/块，200字符重叠）
- 高考试卷：SectionAwareSplitter（按"一、现代文阅读"等标题切分）

每个 chunk 携带元数据：
  {subject, source_file, year, doc_type, section_title}

面试追问点：
- chunk_size 为什么选 1000？
  答：BGE-M3 嵌入模型的最大输入长度是 8192 tokens，1000 个中文字符
  约 1500 tokens，留足空间。太小的 chunk（如 200）会导致上下文碎片化。
- chunk_overlap 为什么是 200？
  答：防止在切分边界处丢失信息。比如"函数"这个词出现在 chunk A 末尾，
  如果没有 overlap，chunk B 中就找不到这个词。
- 为什么不直接全文检索？
  答：单篇试卷可能 5000+ 字，直接检索意味着"整篇相关度"而不是
  "相关段落"，精确度大幅下降。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 1000        # 每个文本块的最大字符数
CHUNK_OVERLAP = 200       # 相邻文本块之间的重叠字符数

# 默认切分器（通用场景）
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _guess_year(filename: str) -> Optional[str]:
    """从文件名中尝试提取 4 位年份。

    例如 "2024年高考语文试卷（新课标Ⅰ卷）.txt" → "2024"
    如果文件名不包含年份返回 None，由调用方处理。
    """
    m = re.search(r"(20\d{2})", filename)
    return m.group(1) if m else None


def _read_pdf(path: Path) -> str:
    """使用 PyMuPDF（fitz）读取 PDF 文件内容。"""
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _read_text(path: Path) -> str:
    """读取纯文本文件（UTF-8 编码）。"""
    return path.read_text(encoding="utf-8")


# 文件扩展名到读取函数的映射
_READERS = {
    ".pdf": _read_pdf,
    ".md": _read_text,
    ".txt": _read_text,
}


def load_documents(
    data_dir: str | Path,
    subject: str,
    doc_type: str = "exam",
    splitter=None,
) -> list[Document]:
    """加载指定目录下的所有受支持文件，切分为 chunk 并返回。

    参数：
        data_dir: 数据目录路径（如 data/chinese/）
        subject: 学科名称（如 "chinese"、"math"），存入 metadata
        doc_type: 文档类型（默认 "exam"），存入 metadata
        splitter: 自定义切分器。None 时使用默认 RecursiveCharacterTextSplitter。
                  对于高考试卷，传入 SectionAwareSplitter 实例。

    返回：
        LangChain Document 列表，每个元素包含 page_content 和 metadata。
        metadata 包含：subject, source_file, year, doc_type
    """
    active_splitter = splitter if splitter is not None else _splitter

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []
    for filepath in sorted(data_dir.iterdir()):
        ext = filepath.suffix.lower()
        reader = _READERS.get(ext)
        if reader is None:
            continue  # 跳过不支持的文件类型

        raw_text = reader(filepath)
        if not raw_text.strip():
            continue  # 跳过空文件

        # 构建元数据（后续检索和过滤使用）
        metadata = {
            "subject": subject,
            "source_file": filepath.name,
            "year": _guess_year(filepath.name) or "unknown",
            "doc_type": doc_type,
        }

        # 切分文档
        chunks = active_splitter.create_documents(
            texts=[raw_text],
            metadatas=[metadata],
        )
        documents.extend(chunks)

    return documents
