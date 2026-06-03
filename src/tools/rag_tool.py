"""RAG 检索工具——封装为 LangChain @tool 格式

将混合检索管道（向量+BM25+Reranker）包装为 LangChain 标准的 Tool。
虽然在当前架构中，它是由图节点直接调用的（而非 LLM 自主选择），
但 @tool 装饰器让函数的参数和返回值 schema 显式化，便于维护。

函数签名与 src.rag.retriever.retrieve() 一致，只是多了一层 @tool 封装。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from src.rag.retriever import retrieve


@tool
def rag_retrieve(query: str, subject: Optional[str] = None) -> dict:
    """从本地高考知识库检索相关内容（历年真题、课程大纲、笔记等）。

    参数：
        query: 检索查询文本
        subject: 可选的学科过滤（math/chinese），缩小检索范围

    返回：
        dict with keys:
            docs: 检索到的文档列表 [{content, source, score, metadata}, ...]
            is_hit: 是否找到相关结果（用于判断是否依赖 LLM 自身知识）
    """
    return retrieve(query=query, subject=subject)
