"""语义缓存模块——用嵌入向量匹配相似问题，命中时跳过RAG+LLM

使用方式：
    from src.cache.semantic import get_semantic_cache
    cache = get_semantic_cache()

    # 查缓存
    emb = embedding_fn.embed_query(query)
    cached = cache.lookup(emb)

    # 存缓存
    if not cached:
        answer = generate_answer(...)
        cache.store(query, answer, emb)

面试追问点：
- 相似度阈值为什么设 0.92？答：权衡命中率和准确性。
  太低（0.85）会返回不相关答案，太高（0.97）命中率趋近零。
  0.92 约对应"同义改写"的相似度。
- 为什么用余弦相似度而不是 L2 距离？
  答：语义相似度看方向不看距离，"你好"和"您好"的嵌入方向一致
  但 L2 距离可能较大，余弦相似度更能体现语义接近程度。
"""

from src.cache.semantic import SemanticCache, get_semantic_cache

__all__ = ["SemanticCache", "get_semantic_cache"]
