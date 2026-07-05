"""混合检索管道（Hybrid RAG Retrieval）——面试核心考点

这是系统的"搜索引擎"，结合三种检索技术来弥补各自的短板：

1. 向量检索（语义搜索）
   ChromaDB + BGE-M3 嵌入，将文本转为 1024 维向量，通过 L2 距离计算语义相似度。
   优势：理解"函数单调性"和"导数符号"之间的语义关联
   劣势：对专有名词的精确匹配不如 BM25

2. BM25 关键词检索（精确匹配）
   rank-bm25 + jieba 中文分词，计算查询词与文档的词频-逆文档频率。
   优势：精确匹配专有名词（如"新课标Ⅰ卷"），不受嵌入质量影响
   劣势：对语义相似但用词不同的查询无能为力

3. BGE Reranker 重排序（交叉编码）
   将合并后的候选集通过 BGE-Reranker 模型逐对打分，这是一个交叉编码器
   （cross-encoder），比向量检索的"双编码器"（bi-encoder）精度更高。
   优势：候选文档数十条时，用较慢但更准的模型精排
   劣势：速度慢（O(n)），不适合直接检索

完整流程：
  query → [Vector Search (top-10) ∥ BM25 (top-10)]
          → Merge & Dedup (MD5)
          → BGE Reranker (top-5)
          → Threshold Filter → Return

面试追问点：
- 为什么向量检索 + BM25 比单纯向量检索好？
  答：向量检索擅长语义匹配，BM25 擅长精确匹配。中文高考题目中，
  "2024年新课标Ⅰ卷"这种精确表达，BM25 比向量检索更准。
- 为什么用 jieba？对比过其他中文分词器吗？
  答：jieba 轻量无依赖，准确率对 BM25 影响不大（BM25 本身对分词
  不敏感）。如果换成 HanLP 或 LAC 分词，BM25 结果差异不到 5%。
- Reranker 为什么放在 merge 之后而不是之前？
  答：Reranker 是 O(n) 的逐对计算，候选集太大时延迟不可接受。
  先通过向量+BM25 把候选从全量缩到 top-20，再用 Reranker 精排。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import get_setting
from src.rag.indexer import load_index
from src.rag.reranker import rerank

logger = logging.getLogger(__name__)

# 向后兼容的旧常量（测试文件可能引用它们）
RELEVANCE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

# ============================================================================
# 单例变量（懒加载）
# ============================================================================
# 使用模块级全局变量实现"一次加载，重复使用"。
# 向量存储和 BM25 索引在进程生命周期内保持，避免每次检索都重新加载。

_vectorstore = None
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict[str, Any]] = []   # 与 BM25 索引平行的文档数组
_bm25_doc_count: int = 0                  # 上次构建 BM25 时的文档数


def _get_vectorstore():
    """懒加载向量存储单例。

    只在首次调用时从磁盘加载 ChromaDB，后续复用内存中的实例。
    如果多次调用之间索引发生了变化，需要调用方手动处理（当前设计简单场景够用）。
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_index()
    return _vectorstore


# ============================================================================
# BM25 关键词检索引擎
# ============================================================================

def _build_bm25_index() -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """从 ChromaDB 的所有文档构建 BM25 索引。

    为什么 BM25 索引需要从 ChromaDB 获取文档？
    因为文档本身存储在 ChromaDB 中（包括 content 和 metadata），
    我们不维护一份独立的文档副本。这样做的好处是"一份数据"，
    不会出现 ChromaDB 和 BM25 之间数据不同步的问题。

    返回 (bm25_index, corpus)：
    - bm25_index：BM25Okapi 实例，用于计算查询与文档的相关度
    - corpus：与 BM25 索引平行的文档字典列表，包含 content/source/metadata

    注意：更新 _bm25_doc_count 用于后续检测文档数变化。
    """
    global _bm25_doc_count
    try:
        vs = _get_vectorstore()
        collection = vs._collection
        data = collection.get(include=["documents", "metadatas"])

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        _bm25_doc_count = collection.count()

        if not documents:
            logger.warning("ChromaDB collection is empty; BM25 index will be empty")
            return None, []

        corpus: list[dict[str, Any]] = []
        tokenized: list[list[str]] = []

        for doc_text, meta in zip(documents, metadatas, strict=False):
            if not doc_text:
                continue
            corpus.append({
                "content": doc_text,
                "source": (meta or {}).get("source_file", "unknown"),
                "metadata": meta or {},
            })
            # 使用 jieba 进行中文分词
            tokenized.append(jieba.lcut(doc_text))

        if not tokenized:
            return None, []

        return BM25Okapi(tokenized), corpus

    except Exception:
        logger.warning("Failed to build BM25 index; keyword search disabled", exc_info=True)
        return None, []


def _get_bm25(force_rebuild: bool = False) -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """懒加载 BM25 单例，自动检测 ChromaDB 变化并重建。

    为什么需要自动重建？
    当索引脚本重新运行（添加了新的试卷），ChromaDB 中的文档增多，
    旧的 BM25 索引不再完整。通过比较 _bm25_doc_count 与当前 ChromaDB
    文档数，可以无感知地触发重建。

    触发重建条件：
    - 首次调用（没有缓存）
    - force_rebuild=True（显式请求）
    - ChromaDB 文档数变化（自动检测）
    """
    global _bm25_index, _bm25_corpus

    needs_build = _bm25_index is None and not _bm25_corpus

    if not needs_build and not force_rebuild:
        # 检查 ChromaDB 文档数是否变化
        try:
            vs = _get_vectorstore()
            current_count = vs._collection.count()
            if current_count != _bm25_doc_count:
                needs_build = True
                logger.info(
                    "BM25 invalidation: doc count changed %d → %d, rebuilding",
                    _bm25_doc_count,
                    current_count,
                )
        except Exception:
            logger.warning("Failed to check ChromaDB doc count", exc_info=True)

    if needs_build or force_rebuild:
        _bm25_index, _bm25_corpus = _build_bm25_index()

    return _bm25_index, _bm25_corpus


def _bm25_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """使用 BM25 进行关键词检索。

    流程：
    1. 使用 jieba 对查询进行中文分词
    2. 计算分词后的查询与所有文档的 BM25 分数
    3. 按分数降序排列，返回 top_k

    BM25 的优势：对精确匹配非常敏感。比如查询"2024年高考数学真题"，
    如果知识库中有同名文档，BM25 会给出非常高的分数。

    BM25 的劣势：无法处理同义词。比如查询"函数"匹配不到"映射"。
    """
    bm25, corpus = _get_bm25()
    if bm25 is None or not corpus:
        return []

    tokens = jieba.lcut(query)
    scores = bm25.get_scores(tokens)

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results: list[dict[str, Any]] = []
    for idx, score in scored[:top_k]:
        if score <= 0:
            break
        doc = corpus[idx]
        results.append({
            "content": doc["content"],
            "source": doc["source"],
            "score": round(float(score), 4),
            "metadata": doc["metadata"],
        })
    return results


# ============================================================================
# 合并去重
# ============================================================================

def _content_hash(text: str) -> str:
    """对文本内容计算 MD5 哈希，用于去重。

    为什么用 MD5？
    速度快（Python 原生实现）、碰撞概率极低。
    不需要加密安全性，只需要唯一性。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _merge_and_dedup(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并向量检索和 BM25 检索结果，按内容去重。

    去重策略：以向量检索结果为锚点，BM25 结果中与向量结果
    内容重复的会被丢弃。这样既保证了语义相关文档的优先级，
    又不会遗漏 BM25 单独发现的精确匹配文档。

    注意：向量检索在前、BM25 在后的顺序是有意为之——
    is_hit 判断基于 top_score，向量检索的分数经过 ChromaDB
    校准，比 BM25 的原始分数更有参考价值。
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    # 向量结果优先（分数经过 ChromaDB 校准）
    for doc in vector_results:
        h = _content_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    # BM25 结果补充（不重复的精确匹配）
    for doc in bm25_results:
        h = _content_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    return merged


# ============================================================================
# 公共 API
# ============================================================================

def retrieve(
    query: str,
    subject: str | None = None,
    year: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """混合检索入口函数——向量 + BM25 + Reranker。

    这是 academic.py 中 rag_retrieve 节点调用的函数。
    调用方（rag_retrieve 节点）负责将其放入线程池执行以避免阻塞事件循环。

    参数：
        query: 用户查询（或改写后的查询）
        subject: 可选的学科过滤器（math/chinese）
                 缩小检索范围，提高相关度
        year: 可选的年份过滤器（如 "2024"）
        top_k: 最终返回的文档数量（经过 Reranker 后）

    返回值：
        {
            "docs": [{"content", "source", "score", "metadata", "rerank_score"}, ...],
            "is_hit": bool  # 最佳文档的相关度是否超过阈值
        }
        is_hit 用于判断是否需要依赖 LLM 的自身知识（检索结果不佳时）。
    """
    vector_top_k = get_setting("rag.vector_top_k", 10)
    bm25_top_k = get_setting("rag.bm25_top_k", 10)
    reranker_top_n = get_setting("rag.reranker_top_n", top_k)
    threshold = get_setting("rag.relevance_threshold", RELEVANCE_THRESHOLD)

    # ── 第 1 步：向量检索（语义搜索） ──
    vectorstore = _get_vectorstore()

    # 构建元数据过滤器
    where_filter: dict | None = None
    conditions: list[dict] = []
    if subject:
        conditions.append({"subject": {"$eq": subject}})
    if year:
        conditions.append({"year": {"$eq": year}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # 执行相似度搜索
    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=vector_top_k,
        filter=where_filter,
    )

    vector_docs: list[dict[str, Any]] = []
    for doc, score in results:
        vector_docs.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "unknown"),
            "score": round(score, 4),
            "metadata": doc.metadata,
        })

    # ── 第 2 步：BM25 关键词检索（精确匹配） ──
    bm25_docs = _bm25_search(query, top_k=bm25_top_k)

    # ── 第 3 步：合并去重 ──
    merged = _merge_and_dedup(vector_docs, bm25_docs)

    # ── 第 4 步：BGE Reranker 重排序 ──
    ranked = rerank(query, merged, top_n=reranker_top_n) if merged else []

    # ── 第 5 步：判断是否命中 ──
    is_hit = False
    if ranked:
        best_score = ranked[0].get("rerank_score", ranked[0].get("score", 0))
        is_hit = best_score >= threshold

    return {"docs": ranked, "is_hit": is_hit}
