"""BGE Reranker API 封装——通过 SiliconFlow 调用交叉编码器重排序

Reranker（重排序器）是混合检索管道的最后一道关卡：

  Vector Search (Top-10) + BM25 (Top-10)
         → Merge & Dedup
         → BGE Reranker (Top-5)  ← 本模块
         → Return

为什么需要 Reranker？
向量检索（BGE-M3）是"双编码器"（bi-encoder），将查询和文档分别编码为向量，
再用余弦/L2 距离计算相似度。这种方式速度快但精度有限——两个独立编码器
无法捕捉查询和文档之间的细粒度交互。

Reranker 是"交叉编码器"（cross-encoder），将查询和文档拼接后一起输入，
能够建模它们之间的所有交互，精度更高但速度更慢。
所以在管道中——先用向量+BM25 从全量数据库缩到 top-20，
再让 Reranker 精排 top-5。

面试追问点：
- Reranker 和 Embedding 模型的区别？
  答：Embedding 模型（BGE-M3）是 bi-encoder，查询和文档独立编码，
  适合大规模检索。Reranker（BGE-Reranker）是 cross-encoder，
  查询和文档联合编码，适合小规模重排序。
- Reranker 的延迟是多少？
  答：10 篇文档的重排序约 500ms。如果候选集太大（如 50 篇），
  延迟可能超过 2 秒，所以需要先用向量+BM25 粗筛。
- 失败后为什么不阻塞？
  答：Reranker 只是"锦上添花"，没有它，向量+BM25 的结果
  已经可用。直接返回降级结果比让用户等待错误处理更好。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.config import get_setting

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
_TIMEOUT = 15  # 秒


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """通过 SiliconFlow BGE Reranker 对候选文档重排序。

    参数：
        query: 用户查询字符串
        documents: 候选文档列表，每项必须有 "content" 字段
        top_n: 返回结果数（默认来自 settings.yaml 的 rag.reranker_top_n）

    返回：
        按 reranker 相关度排序的文档列表（最多 top_n 条），
        每项增加 "rerank_score" 字段表示重排序分数。
        如果 API 调用失败，返回截断到 top_n 的原始文档列表（降级）。

    请求示例（HTTPS）：
        POST https://api.siliconflow.cn/v1/rerank
        {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "用户的问题",
            "documents": ["候选1", "候选2", ...],
            "top_n": 5
        }
        → {"results": [{"index": 2, "relevance_score": 0.95}, ...]}
    """
    if not documents:
        return []

    if top_n is None:
        top_n = get_setting("rag.reranker_top_n", 5)

    api_key = os.getenv("SILICONFLOW_API_KEY")
    model = os.getenv(
        "RERANKER_MODEL",
        get_setting("rag.reranker_model", _DEFAULT_RERANKER_MODEL),
    )

    doc_texts = [d["content"] for d in documents]

    try:
        resp = httpx.post(
            _RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "query": query,
                "documents": doc_texts,
                "top_n": top_n,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        # —— 降级：API 失败时返回原始顺序 ——
        logger.warning("Reranker API call failed; returning original order", exc_info=True)
        return documents[:top_n]

    results: list[dict[str, Any]] = data.get("results", [])
    ranked: list[dict[str, Any]] = []
    for item in results:
        idx = item["index"]
        if 0 <= idx < len(documents):
            doc = {**documents[idx], "rerank_score": item["relevance_score"]}
            ranked.append(doc)

    return ranked[:top_n]
