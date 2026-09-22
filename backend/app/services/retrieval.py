"""
Retrieval service.

Loads the persisted FAISS vector store built by the notebook
(notebooks/rag_pipeline.ipynb) and exposes a simple retrieve() function.
The backend never rebuilds embeddings at request time -- it only loads
what the notebook already persisted to disk.
"""
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[dict] | None = None
        self._embedding_model: SentenceTransformer | None = None

    def load(self) -> None:
        """Load the persisted vector store once, at app startup."""
        store_dir = Path(settings.VECTOR_STORE_DIR)
        index_path = store_dir / f"{settings.COLLECTION_NAME}.faiss"
        meta_path = store_dir / f"{settings.COLLECTION_NAME}.pkl"

        logger.info("Loading vector store from %s", store_dir)

        self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info(
            "Vector store loaded. Index '%s' has %d chunks.",
            settings.COLLECTION_NAME,
            self._index.ntotal,
        )

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def count(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        """Return the top_k most relevant chunks for a question."""
        if self._index is None or self._chunks is None:
            raise RuntimeError("Vector store not loaded. Call load() at startup first.")

        k = top_k or settings.TOP_K

        # Normalize the query the same way the notebook normalized chunk
        # embeddings at index time, so inner product == cosine similarity.
        query_vec = self._embedding_model.encode([question], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")

        scores, indices = self._index.search(query_vec, k)

        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._chunks[idx]
            chunks.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "text": meta["text"],
                    "document": meta.get("source", "unknown"),
                    "score": float(score),  # cosine similarity (inner product on normalized vectors)
                }
            )
        return chunks


# Singleton instance used across the app (loaded once in the FastAPI lifespan)
retrieval_service = RetrievalService()
