"""Simple numpy-based vector store — zero native dependencies.

Replaces ChromaDB when onnxruntime is unavailable (common on Windows).
Provides the same interface: similarity_search_with_relevance_scores, _collection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def _l2_to_relevance(distance: float) -> float:
    """Convert L2 distance to [0, 1] relevance score."""
    return 1.0 - distance / math.sqrt(2)


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "gaokao_docs"


class SimpleVectorStore:
    """Drop-in replacement for ChromaDB-based index.

    Stores embeddings in memory + on disk via pickle for persistence.
    """

    def __init__(self, persist_dir: str):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._docs: list[Document] = []
        self._embeddings: np.ndarray | None = None

    # ---- Persistence ---------------------------------------------------------

    def save(self) -> None:
        """Write state to disk."""
        data = {
            "docs": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in self._docs
            ],
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else None,
        }
        path = self._persist_dir / f"{COLLECTION_NAME}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """Load state from disk. Returns False if no saved state exists."""
        path = self._persist_dir / f"{COLLECTION_NAME}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._docs = [
            Document(page_content=d["content"], metadata=d["metadata"])
            for d in data["docs"]
        ]
        if data["embeddings"]:
            self._embeddings = np.array(data["embeddings"], dtype=np.float32)
        return True

    # ---- Build ---------------------------------------------------------------

    def build_from_documents(
        self,
        documents: list[Document],
        embedding_fn: OpenAIEmbeddings,
    ) -> None:
        """Index *documents* with embeddings from the given function."""
        texts = [d.page_content for d in documents]
        print(f"  Embedding {len(texts)} chunks via SiliconFlow API (batch_size=5)...")
        all_embeddings = []
        import time
        for i in range(0, len(texts), 5):
            batch = texts[i : i + 5]
            all_embeddings.extend(embedding_fn.embed_documents(batch))
            print(f"    batch {i // 5 + 1}/{(len(texts) + 4) // 5} OK")
            if i + 5 < len(texts):
                time.sleep(0.3)
        self._docs = list(documents)
        self._embeddings = np.array(all_embeddings, dtype=np.float32)
        self.save()
        print(f"  Saved {len(self._docs)} vectors to {self._persist_dir}")

    # ---- Search --------------------------------------------------------------

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Return top-k (Document, relevance_score) pairs."""
        if self._embeddings is None or len(self._docs) == 0:
            return []

        # Embed the query
        embedding_fn = _make_embedding_fn()
        query_vec = np.array(embedding_fn.embed_query(query), dtype=np.float32)

        # Compute L2 distances to all documents
        distances = np.linalg.norm(self._embeddings - query_vec, axis=1)

        # Apply metadata filter if provided
        valid_indices = list(range(len(self._docs)))
        if filter:
            valid_indices = self._apply_filter(filter)

        # Get top-k among valid indices
        scored = [(i, distances[i]) for i in valid_indices]
        scored.sort(key=lambda x: x[1])

        results: list[tuple[Document, float]] = []
        for idx, dist in scored[:k]:
            score = _l2_to_relevance(dist)
            results.append((self._docs[idx], score))

        return results

    def _apply_filter(self, filter_dict: dict) -> list[int]:
        """Filter documents by metadata. Supports {'subject': {'$eq': val}} and {'$and': [...]}."""
        indices = []
        for i, doc in enumerate(self._docs):
            meta = doc.metadata
            if _matches_filter(meta, filter_dict):
                indices.append(i)
        return indices

    @property
    def _collection(self):
        """Compatibility shim: expose a .count() interface."""
        return _CollectionProxy(self)


class _CollectionProxy:
    def __init__(self, store: SimpleVectorStore):
        self._store = store

    def count(self) -> int:
        return len(self._store._docs)

    def get(self, include=None):
        """Compatibility shim for BM25 builder."""
        documents = [d.page_content for d in self._store._docs]
        metadatas = [d.metadata for d in self._store._docs]
        return {"documents": documents, "metadatas": metadatas}


def _matches_filter(meta: dict, filter_dict: dict) -> bool:
    """Check if *meta* satisfies a Chroma-style filter."""
    if "$and" in filter_dict:
        return all(_matches_filter(meta, sub) for sub in filter_dict["$and"])
    for key, cond in filter_dict.items():
        if isinstance(cond, dict) and "$eq" in cond:
            if meta.get(key) != cond["$eq"]:
                return False
    return True


def _make_embedding_fn() -> OpenAIEmbeddings:
    """Create embedding client from env vars."""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    )
