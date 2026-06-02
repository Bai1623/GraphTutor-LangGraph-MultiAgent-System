"""Vector store index builder — ChromaDB with SimpleVectorStore fallback.

Uses SiliconFlow's OpenAI-compatible embedding API (BAAI/bge-m3).
Automatically falls back to a pure-numpy store when ChromaDB's onnxruntime
dependency is unavailable (common on Windows).
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
    """Always resolve to an absolute path anchored at project root."""
    rel = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_store/")
    path = Path(rel)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def _get_embedding(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    """Create an OpenAI-compatible embedding client backed by SiliconFlow."""
    model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base=os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ),
    )


def _l2_to_relevance(distance: float) -> float:
    """Convert L2 distance to a [0, 1] relevance score."""
    return 1.0 - distance / math.sqrt(2)


def _content_id(doc: Document) -> str:
    """Deterministic ID from chunk content — true dedup across repeated runs."""
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{doc.metadata.get('source_file', 'unknown')}_{digest}"


# ---------------------------------------------------------------------------
# Detect backend
# ---------------------------------------------------------------------------

_USE_SIMPLE_STORE = False


def _try_chromadb():
    """Return True if chromadb is usable."""
    global _USE_SIMPLE_STORE
    try:
        import chromadb  # noqa: F401
        from chromadb.config import Settings  # noqa: F401
        # Quick smoke test
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index(
    documents: list[Document],
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """Build vector index from *documents*. Returns ChromaWrapper or SimpleVectorStore."""
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
    """Load existing index. Returns ChromaWrapper or SimpleVectorStore."""
    persist_directory = _resolve_persist_dir(persist_directory)

    from src.rag.simple_store import SimpleVectorStore

    store = SimpleVectorStore(persist_directory)
    if store.load():
        return store

    # No index exists yet
    return None
