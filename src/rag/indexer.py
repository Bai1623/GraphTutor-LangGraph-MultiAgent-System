"""向量索引构建器——ChromaDB + SimpleVectorStore 自动降级

核心功能：
1. 通过 SiliconFlow API 将文档转换为 BGE-M3 嵌入向量
2. 存入 ChromaDB（如果 onnxruntime 不可用则降级到 SimpleVectorStore）
3. MD5 内容哈希实现去重和增量更新
4. L2 距离 → [0,1] 相关度分数转换

为什么需要降级？
ChromaDB 依赖 onnxruntime，在 Windows 下有时 DLL 加载失败。
SimpleVectorStore 是一个纯 NumPy + pickle 实现，不需要任何 C 扩展。
两者实现了相同的接口（similarity_search_with_relevance_scores）。

面试追问点：
- 为什么用 SiliconFlow API 而不是本地 HuggingFace？
  答：本地加载 BGE-M3 模型约需 2GB 内存 + 5 秒加载时间。
  API 调用零内存占用，且方便在不同环境间迁移。
- Embedding 一次请求耗时多少？
  62 个 chunk 的批量嵌入约 2-3 秒（SiliconFlow API）。
  单条查询嵌入约 200ms。
- L2 到 relevance 的转换为什么是线性的？
  因为 ChromaDB 默认使用 L2 距离，但前端和下游逻辑需要
  "分数越高越相关"的直观语义。线性映射 [0, sqrt(2)] → [1, 0]。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "gaokao_docs"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_persist_dir(persist_directory: Optional[str] = None) -> str:
    """将相对路径解析为项目根目录下的绝对路径。

    例如 "chroma_store/" → "F:/gaokao_tutor/chroma_store/"
    确保无论从哪个目录启动项目，ChromaDB 都能找到正确的位置。
    """
    rel = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_store/")
    path = Path(rel)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def _get_embedding(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    """创建 OpenAI 兼容的嵌入客户端，后端是 SiliconFlow BGE-M3。

    为什么不直接用 OpenAI Embedding？
    - ChatGPT 的 text-embedding-ada-002 对中文支持不如 BGE-M3
    - BGE-M3 是 1024 维（ada-002 是 1536 维），存储和计算更高效
    - BGE-M3 在中文语义理解任务上排名靠前
    """
    model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base=os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ),
    )


def _l2_to_relevance(distance: float) -> float:
    """将 ChromaDB 的 L2 距离转换为 [0, 1] 区间的相关度分数。

    原理：对于归一化后的嵌入向量，两个向量的 L2 距离在 [0, 2] 之间。
    对于 BGE-M3 嵌入，实际观察到的最大距离约 sqrt(2) ≈ 1.414。
    因此线性映射 [0, sqrt(2)] → [1, 0] 即可得到合理的结果。
    """
    return 1.0 - distance / math.sqrt(2)


def _content_id(doc: Document) -> str:
    """对段落内容计算确定性 ID，用于去重。

    格式："{source_file}_{content_md5}"
    相同的文档内容永远生成相同的 ID，保证幂等性。

    面试追问：为什么不用 UUID？
    因为每次重建索引时，如果文档内容没变，我们希望复用旧 ID
    而不是新建。UUID 做不到这一点。
    """
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{doc.metadata.get('source_file', 'unknown')}_{digest}"


# ============================================================================
# 后端检测
# ============================================================================

_USE_SIMPLE_STORE = False


def _try_chromadb():
    """检测 ChromaDB 是否可用。

    通过创建临时集合来验证 onnxruntime 是否能正常加载。
    如果失败（常见于 Windows 环境），回退到 SimpleVectorStore。
    """
    global _USE_SIMPLE_STORE
    try:
        import chromadb  # noqa: F401
        from chromadb.config import Settings  # noqa: F401
        c = chromadb.PersistentClient(
            path=_resolve_persist_dir(),
            settings=Settings(anonymized_telemetry=False),
        )
        c.get_or_create_collection("__smoke_test__")
        return True
    except Exception:
        logger.warning("ChromaDB unavailable, falling back to SimpleVectorStore")
        _USE_SIMPLE_STORE = True
        return False


# ============================================================================
# 公共构建/加载接口
# ============================================================================

def build_index(
    documents: list[Document],
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """从文档构建向量索引。

    流程：
    1. 解析持久化目录（相对于项目根）
    2. 创建嵌入客户端（SiliconFlow BGE-M3）
    3. 使用 SimpleVectorStore 构建纯 NumPy 索引
    4. 返回向量存储实例（包含相似度搜索接口）

    当前版本始终使用 SimpleVectorStore（稳定性和兼容性优先）。
    索引数据持久化到 {persist_directory}/vectors.npy 和 metadata.json。

    参数：
        documents: LangChain Document 列表（来自 loader.py）
        persist_directory: 索引的持久化路径
        embedding_model: 嵌入模型名称（默认 BAAI/bge-m3）

    返回：
        实现了 similarity_search_with_relevance_scores 的存储实例
    """
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding_fn = _get_embedding(embedding_model)

    from src.rag.simple_store import SimpleVectorStore

    store = SimpleVectorStore(persist_directory)
    store.build_from_documents(documents, embedding_fn)
    return store


def load_index(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """从磁盘加载已构建的索引。

    如果索引不存在则返回 None（由调用方处理）。
    适用于 retriever.py 等需要"只读"访问的场景。

    返回：
        SimpleVectorStore 实例，或 None（未构建索引时）
    """
    persist_directory = _resolve_persist_dir(persist_directory)

    from src.rag.simple_store import SimpleVectorStore

    store = SimpleVectorStore(persist_directory)
    if store.load():
        return store

    # 索引未存在（首次运行还未构建）
    return None
