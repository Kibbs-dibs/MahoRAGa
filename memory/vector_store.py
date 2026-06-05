"""
FAISS dense-vector store for Project Aletheia.

Responsibilities
────────────────
• Manage a 384-dimensional FAISS IndexFlatIP (cosine similarity via
  normalised inner-product) as specified in the PRD constraints.
• Encode text chunks using ``sentence-transformers/all-MiniLM-L6-v2``.
• Persist and reload the FAISS binary index + a sidecar JSON metadata
  file that maps integer FAISS ids → original chunk text and metadata.
• Provide top-K dense retrieval consumed by the Memory-Guided Retriever
  (``graph/nodes/retriever.py``) and the Reciprocal Rank Fusion (RRF)
  dual-index merger.

All public methods raise ``VectorStoreError`` on failure, keeping the
exception boundary clean for the orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from core.config import settings
from core.exceptions import EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)


# ── Lazy model singleton ────────────────────────────────────────────────
# We defer the heavyweight import so that modules that only need the
# FAISS I/O helpers don't pay the transformer-load cost on import.

_ENCODER = None


def _get_encoder():
    """Return the shared SentenceTransformer encoder, loading once."""
    global _ENCODER
    if _ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer

            _ENCODER = SentenceTransformer(settings.embedding_model_name)
            logger.info(
                "Loaded embedding model: %s", settings.embedding_model_name
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load SentenceTransformer "
                f"'{settings.embedding_model_name}': {exc}"
            ) from exc
    return _ENCODER


# ═══════════════════════════════════════════════════════════════════════════
#  FAISS Vector Store
# ═══════════════════════════════════════════════════════════════════════════


class VectorStore:
    """FAISS-backed dense vector index with disk serialisation.

    Usage
    -----
    >>> store = VectorStore()
    >>> store.add_chunks(["Transformers use self-attention …"], [{"chunk_id": "c1"}])
    >>> results = store.search("attention mechanism", top_k=5)
    >>> store.save()

    The index uses **IndexFlatIP** (inner-product) over L2-normalised
    vectors, which is mathematically equivalent to cosine similarity.
    """

    def __init__(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self._index_path = index_path or settings.faiss_index_path
        self._metadata_path = metadata_path or settings.faiss_metadata_path
        self._dim = embedding_dim or settings.embedding_dim

        # Initialise an empty flat inner-product index
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)

        # Metadata sidecar: list aligned 1-to-1 with FAISS row ids
        # Each entry stores the raw chunk text + arbitrary metadata.
        self._metadata: List[Dict[str, Any]] = []

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self) -> None:
        """Serialise the FAISS index and metadata to disk."""
        try:
            Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, self._index_path)
            logger.info(
                "FAISS index saved to %s (%d vectors)",
                self._index_path,
                self._index.ntotal,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to save FAISS index to {self._index_path}: {exc}"
            ) from exc

        try:
            Path(self._metadata_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._metadata_path, "w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, ensure_ascii=False)
            logger.info(
                "FAISS metadata saved to %s (%d entries)",
                self._metadata_path,
                len(self._metadata),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to save FAISS metadata to {self._metadata_path}: {exc}"
            ) from exc

    def load(self) -> None:
        """Load a previously serialised FAISS index + metadata from disk.

        Raises ``VectorStoreError`` if the files are absent or corrupt.
        """
        if not os.path.isfile(self._index_path):
            raise VectorStoreError(
                f"FAISS index file not found: {self._index_path}"
            )
        if not os.path.isfile(self._metadata_path):
            raise VectorStoreError(
                f"FAISS metadata file not found: {self._metadata_path}"
            )

        try:
            self._index = faiss.read_index(self._index_path)
            logger.info(
                "FAISS index loaded from %s (%d vectors)",
                self._index_path,
                self._index.ntotal,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to read FAISS index: {exc}"
            ) from exc

        try:
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            logger.info(
                "FAISS metadata loaded (%d entries)", len(self._metadata)
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to read FAISS metadata: {exc}"
            ) from exc

        # Sanity: metadata length must match index size
        if len(self._metadata) != self._index.ntotal:
            raise VectorStoreError(
                f"Metadata/index mismatch: {len(self._metadata)} metadata "
                f"entries vs {self._index.ntotal} index vectors."
            )

    def load_if_exists(self) -> bool:
        """Attempt to load from disk; return False (no raise) if absent."""
        if os.path.isfile(self._index_path) and os.path.isfile(
            self._metadata_path
        ):
            self.load()
            return True
        logger.info("No existing FAISS index found — starting fresh.")
        return False

    # ── Encoding ────────────────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* into L2-normalised 384-dim float32 vectors.

        The normalisation ensures that IndexFlatIP computes cosine
        similarity rather than raw dot-product.
        """
        encoder = _get_encoder()
        try:
            embeddings: np.ndarray = encoder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalise → cosine sim
            )
            return embeddings.astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    # ── Indexing ────────────────────────────────────────────────────────

    def add_chunks(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Embed *texts* and insert into the FAISS index.

        Parameters
        ----------
        texts:
            Raw chunk strings to embed and index.
        metadatas:
            Optional per-chunk metadata dicts.  If ``None``, each entry
            defaults to ``{"text": <chunk>}``.

        Returns
        -------
        The number of vectors added.
        """
        if not texts:
            return 0

        embeddings = self.encode(texts)

        if metadatas is None:
            metadatas = [{"text": t} for t in texts]
        else:
            # Ensure every metadata entry carries the raw text
            for meta, txt in zip(metadatas, texts):
                meta.setdefault("text", txt)

        if len(metadatas) != len(texts):
            raise VectorStoreError(
                f"Metadata length ({len(metadatas)}) != texts length "
                f"({len(texts)})."
            )

        self._index.add(embeddings)
        self._metadata.extend(metadatas)

        logger.debug("Added %d vectors (total now %d)", len(texts), self._index.ntotal)
        return len(texts)

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Insert pre-computed vectors directly (skip encoding).

        Useful when embeddings have already been generated externally,
        e.g. during offline ingestion.
        """
        if vectors.shape[0] != len(metadatas):
            raise VectorStoreError(
                f"Vector count ({vectors.shape[0]}) != metadata count "
                f"({len(metadatas)})."
            )
        # Normalise just in case caller forgot
        faiss.normalize_L2(vectors)
        self._index.add(vectors.astype(np.float32))
        self._metadata.extend(metadatas)
        return vectors.shape[0]

    # ── Retrieval ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search.

        Returns up to *top_k* results, each dict containing:
        ``score`` (float), ``rank`` (int 1-based), and all metadata
        keys (including ``text``).
        """
        top_k = top_k or settings.top_k_retrieval
        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return []

        query_vec = self.encode([query])
        # Clamp top_k to index size
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], ids[0]), start=1
        ):
            if idx == -1:
                continue  # padding from FAISS
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            entry["rank"] = rank
            results.append(entry)

        return results

    def search_by_vector(
        self,
        vector: np.ndarray,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Search with a pre-computed query vector (1, dim)."""
        top_k = top_k or settings.top_k_retrieval
        if self._index.ntotal == 0:
            return []

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(vector.astype(np.float32), k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], ids[0]), start=1
        ):
            if idx == -1:
                continue
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            entry["rank"] = rank
            results.append(entry)
        return results

    # ── Utilities ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    def reset(self) -> None:
        """Clear the index and metadata (in-memory only)."""
        self._index.reset()
        self._metadata.clear()
        logger.info("FAISS index reset (in-memory).")

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return the metadata dict for a given internal FAISS row id."""
        if 0 <= idx < len(self._metadata):
            return self._metadata[idx]
        raise VectorStoreError(f"Metadata index {idx} out of range.")
