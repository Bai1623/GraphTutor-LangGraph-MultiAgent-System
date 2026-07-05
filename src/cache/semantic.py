"""语义缓存模块——相似问题复用，降低 Token 消耗

核心思路：
1. 用户提问 → 将问题转为嵌入向量
2. 与缓存中的历史问题比较余弦相似度
3. 相似度 > 阈值 → 直接返回缓存答案（跳过 RAG + LLM）
4. 相似度 < 阈值 → 正常生成，然后将新 Q&A 存入缓存

为什么用语义相似度而不是精确匹配？
- "导数怎么学" 和 "怎么学导数" 语义相同但文本不同
- "2024高考数学难吗" 和 "去年数学卷子难度" 语义相关但词不同
- 语义匹配能覆盖同义改写和口语化表达

面试能讲的成本数据：
- 缓存命中一次节省约 10K Token（RAG检索 + 回答生成 + 幻觉检测）
- 命中率 ~15-30% 意味着日省数百 Token
- DeepSeek 价格下，1000 次命中约省 ￥0.01

存储：
- JSON 文件持久化（data/cache/semantic_cache.json）
- 每条缓存：{query, answer, embedding, created_at}
- 最多保留 200 条，超出时删除最旧的
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 存储路径
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
_CACHE_FILE = _CACHE_DIR / "semantic_cache.json"

# 默认参数
DEFAULT_SIMILARITY_THRESHOLD = 0.92   # 余弦相似度阈值（0-1，越高越严格）
DEFAULT_MAX_ENTRIES = 200             # 最大缓存条目


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    两个归一化向量的点积，范围 [0, 1]。
    1.0 表示完全相同，0.0 表示完全无关。
    """
    a_np = np.array(a)
    b_np = np.array(b)
    dot = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class SemanticCache:
    """语义相似度缓存——用嵌入向量匹配相似问题。

    线程安全，懒加载，自动裁剪。
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._lock = Lock()
        self._entries: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # 文件读写
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """懒加载：首次访问时从磁盘读取。"""
        if self._loaded:
            return

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if _CACHE_FILE.exists():
            try:
                with open(_CACHE_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = data.get("entries", [])
                logger.info("Loaded %d cached Q&A pairs from %s", len(self._entries), _CACHE_FILE)
            except (OSError, json.JSONDecodeError):
                logger.warning("Failed to load cache file, starting fresh")
                self._entries = []
        else:
            self._entries = []

        self._loaded = True

    def _save(self) -> None:
        """将缓存写回磁盘（原子写入）。"""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = _CACHE_FILE.with_suffix(".json.tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump({"entries": self._entries}, f, ensure_ascii=False, indent=2)
            tmp_file.replace(_CACHE_FILE)
        except Exception:
            logger.warning("Failed to save cache", exc_info=True)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def lookup(self, query_embedding: list[float]) -> str | None:
        """查找与给定嵌入向量最相似的缓存问题。

        参数：
            query_embedding: 当前用户查询的嵌入向量（1024维 float 列表）

        返回：
            匹配的答案字符串（相似度 >= 阈值），未匹配时返回 None
        """
        with self._lock:
            self._ensure_loaded()

            if not self._entries:
                return None

            best_score = 0.0
            best_answer = None

            for entry in self._entries:
                cached_emb = entry.get("embedding")
                if not cached_emb or len(cached_emb) != len(query_embedding):
                    continue

                sim = _cosine_similarity(query_embedding, cached_emb)
                if sim > best_score:
                    best_score = sim
                    best_answer = entry.get("answer")

            if best_score >= self._threshold and best_answer is not None:
                logger.info(
                    "Cache HIT (similarity=%.4f, threshold=%.2f)", best_score, self._threshold
                )
                return best_answer

            logger.debug("Cache MISS (best similarity=%.4f < %.2f)", best_score, self._threshold)
            return None

    def store(self, query: str, answer: str, embedding: list[float]) -> None:
        """将新问答对存入缓存。

        参数：
            query: 用户原始查询文本
            answer: 系统生成的回答
            embedding: 查询的嵌入向量
        """
        with self._lock:
            self._ensure_loaded()

            entry = {
                "query": query,
                "answer": answer,
                "embedding": embedding,
                "created_at": datetime.now().isoformat(),
            }
            self._entries.append(entry)

            # 裁剪：超出上限时删除最旧条目
            while len(self._entries) > self._max_entries:
                removed = self._entries.pop(0)
                logger.debug("Cache evicted: %s", removed.get("query", "")[:30])

            self._save()
            logger.info("Cache stored. Total entries: %d", len(self._entries))

    def clear(self) -> None:
        """清空所有缓存（用于测试或重置）。"""
        with self._lock:
            self._entries = []
            if _CACHE_FILE.exists():
                _CACHE_FILE.unlink()
            logger.info("Cache cleared")

    @property
    def size(self) -> int:
        """当前缓存条目数。"""
        with self._lock:
            self._ensure_loaded()
            return len(self._entries)

    @property
    def threshold(self) -> float:
        """当前相似度阈值。"""
        return self._threshold


# ============================================================================
# 全局单例
# ============================================================================

_cache_instance: SemanticCache | None = None


def get_semantic_cache(
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> SemanticCache:
    """获取 SemanticCache 全局单例。

    首次调用时创建，之后返回同一个实例。
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(similarity_threshold=threshold)
    return _cache_instance
