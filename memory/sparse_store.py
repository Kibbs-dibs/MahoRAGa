"""
Disk-backed BM25 sparse-lexical store for Project Aletheia.

Responsibilities
────────────────
• Maintain a BM25Okapi index over tokenised chunk texts for sparse
  lexical retrieval (the "third memory layer").
• Persist the index + corpus metadata to disk via ``pickle`` so that the
  index survives process restarts without full re-ingestion.
• Provide top-K lexical retrieval consumed by the Memory-Guided
  Retriever and the Reciprocal Rank Fusion (RRF) dual-index merger.
• Expose a re-ranking helper using ``cross-encoder/ms-marco-MiniLM-L-6-v2``
  for final precision pass.

Errors are surfaced as ``SparseStoreError`` from ``core.exceptions``.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from core.config import settings
from core.exceptions import SparseStoreError

logger = logging.getLogger(__name__)


# ── Lazy cross-encoder singleton ────────────────────────────────────────

_CROSS_ENCODER = None


def _get_cross_encoder():
    """Return the shared CrossEncoder model, loading once."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder

            _CROSS_ENCODER = CrossEncoder(settings.cross_encoder_model_name)
            logger.info(
                "Loaded cross-encoder model: %s",
                settings.cross_encoder_model_name,
            )
        except Exception as exc:
            raise SparseStoreError(
                f"Failed to load CrossEncoder "
                f"'{settings.cross_encoder_model_name}': {exc}"
            ) from exc
    return _CROSS_ENCODER


# ── Simple whitespace tokeniser ─────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenise(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation suitable for BM25Okapi."""
    return _TOKEN_RE.findall(text.lower())


# ═══════════════════════════════════════════════════════════════════════════
#  Disk-Backed BM25 Store
# ═══════════════════════════════════════════════════════════════════════════


class SparseStore:
    """BM25Okapi index with disk persistence and cross-encoder re-ranking.

    Usage
    -----
    >>> store = SparseStore()
    >>> store.add_documents(["Transformers use self-attention …"], [{"chunk_id": "c1"}])
    >>> results = store.search("attention mechanism", top_k=10)
    >>> store.save()

    Reload on next start-up:

    >>> store = SparseStore()
    >>> store.load()
    >>> results = store.search("multi-head attention")
    """

    def __init__(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        self._index_path = index_path or settings.bm25_index_path
        self._metadata_path = metadata_path or settings.bm25_metadata_path

        # Corpus of tokenised documents (list[list[str]])
        self._tokenised_corpus: List[List[str]] = []
        # Metadata aligned 1-to-1 with the corpus
        self._metadata: List[Dict[str, Any]] = []
        # The BM25Okapi instance — rebuilt whenever the corpus changes
        self._bm25: Optional[BM25Okapi] = None

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self) -> None:
        """Serialise the BM25 index, tokenised corpus, and metadata to disk."""
        try:
            Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tokenised_corpus": self._tokenised_corpus,
                "bm25": self._bm25,
            }
            with open(self._index_path, "wb") as fh:
                pickle.dump(payload, fh)
            logger.info(
                "BM25 index saved to %s (%d documents)",
                self._index_path,
                len(self._tokenised_corpus),
            )
        except Exception as exc:
            raise SparseStoreError(
                f"Failed to save BM25 index to {self._index_path}: {exc}"
            ) from exc

        try:
            Path(self._metadata_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._metadata_path, "w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, ensure_ascii=False)
            logger.info(
                "BM25 metadata saved to %s (%d entries)",
                self._metadata_path,
                len(self._metadata),
            )
        except Exception as exc:
            raise SparseStoreError(
                f"Failed to save BM25 metadata to {self._metadata_path}: {exc}"
            ) from exc

    def load(self) -> None:
        """Load a previously serialised BM25 index + metadata from disk."""
        if not os.path.isfile(self._index_path):
            raise SparseStoreError(
                f"BM25 index file not found: {self._index_path}"
            )
        if not os.path.isfile(self._metadata_path):
            raise SparseStoreError(
                f"BM25 metadata file not found: {self._metadata_path}"
            )

        try:
            with open(self._index_path, "rb") as fh:
                payload = pickle.load(fh)
            self._tokenised_corpus = payload["tokenised_corpus"]
            self._bm25 = payload["bm25"]
            logger.info(
                "BM25 index loaded from %s (%d documents)",
                self._index_path,
                len(self._tokenised_corpus),
            )
        except Exception as exc:
            raise SparseStoreError(
                f"Failed to read BM25 index: {exc}"
            ) from exc

        try:
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            logger.info(
                "BM25 metadata loaded (%d entries)", len(self._metadata)
            )
        except Exception as exc:
            raise SparseStoreError(
                f"Failed to read BM25 metadata: {exc}"
            ) from exc

        if len(self._metadata) != len(self._tokenised_corpus):
            raise SparseStoreError(
                f"Metadata/corpus mismatch: {len(self._metadata)} metadata "
                f"entries vs {len(self._tokenised_corpus)} corpus documents."
            )

    def load_if_exists(self) -> bool:
        """Attempt to load from disk; return False (no raise) if absent."""
        if os.path.isfile(self._index_path) and os.path.isfile(
            self._metadata_path
        ):
            self.load()
            return True
        logger.info("No existing BM25 index found — starting fresh.")
        return False

    # ── Indexing ────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Tokenise *texts*, append to the corpus, and rebuild the BM25
        index.

        Parameters
        ----------
        texts:
            Raw chunk strings to add.
        metadatas:
            Optional per-document metadata dicts.

        Returns
        -------
        Number of documents added.
        """
        if not texts:
            return 0

        if metadatas is None:
            metadatas = [{"text": t} for t in texts]
        else:
            for meta, txt in zip(metadatas, texts):
                meta.setdefault("text", txt)

        if len(metadatas) != len(texts):
            raise SparseStoreError(
                f"Metadata length ({len(metadatas)}) != texts length "
                f"({len(texts)})."
            )

        new_tokens = [_tokenise(t) for t in texts]
        self._tokenised_corpus.extend(new_tokens)
        self._metadata.extend(metadatas)

        # Rebuild the entire BM25 index (BM25Okapi is not incremental)
        self._rebuild_index()

        logger.debug(
            "Added %d documents (total now %d)",
            len(texts),
            len(self._tokenised_corpus),
        )
        return len(texts)

    def _rebuild_index(self) -> None:
        """Reconstruct the BM25Okapi instance from the current corpus."""
        if not self._tokenised_corpus:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenised_corpus)

    # ── Retrieval ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """BM25 lexical search.

        Returns up to *top_k* results, each dict containing:
        ``score`` (float), ``rank`` (int 1-based), and all metadata
        keys (including ``text``).
        """
        top_k = top_k or settings.top_k_retrieval
        if self._bm25 is None or not self._tokenised_corpus:
            logger.warning("BM25 index is empty — returning no results.")
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            logger.warning("Query produced no tokens after tokenisation.")
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Argsort descending, then take top_k
        k = min(top_k, len(scores))
        top_indices = scores.argsort()[::-1][:k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                break  # BM25 scores ≤ 0 indicate no relevance
            entry = dict(self._metadata[idx])
            entry["score"] = float(scores[idx])
            entry["rank"] = rank
            results.append(entry)

        return results

    # ── Cross-Encoder Re-ranking ────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Re-rank *candidates* using the ``ms-marco-MiniLM-L-6-v2``
        cross-encoder for final precision.

        Each candidate dict must contain a ``text`` key.

        Returns the re-ranked list (highest cross-encoder score first),
        each dict augmented with a ``rerank_score`` key.
        """
        top_k = top_k or settings.top_k_retrieval
        if not candidates:
            return []

        cross_encoder = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]

        try:
            ce_scores = cross_encoder.predict(pairs)
        except Exception as exc:
            raise SparseStoreError(
                f"Cross-encoder re-ranking failed: {exc}"
            ) from exc

        # Attach scores and sort descending
        for candidate, score in zip(candidates, ce_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(
            candidates, key=lambda c: c["rerank_score"], reverse=True
        )
        return reranked[:top_k]

    # ── Reciprocal Rank Fusion (RRF) utility ────────────────────────────

    @staticmethod
    def reciprocal_rank_fusion(
        *result_lists: List[Dict[str, Any]],
        k: int = 60,
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Merge multiple ranked result lists using RRF.

        Each result dict must contain a ``text`` key used as the
        identity for deduplication.

        Parameters
        ----------
        result_lists:
            Variable number of ranked lists (e.g. FAISS results, BM25
            results).
        k:
            RRF smoothing constant (default 60 per the original paper).
        top_n:
            Maximum number of fused results to return.

        Returns
        -------
        Fused list sorted by descending RRF score.
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                text_key = doc.get("text", "")
                rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + 1.0 / (
                    k + rank
                )
                # Keep the richest metadata (first encounter wins)
                if text_key not in doc_map:
                    doc_map[text_key] = dict(doc)

        # Build final list sorted by RRF score
        fused: List[Dict[str, Any]] = []
        for text_key, score in sorted(
            rrf_scores.items(), key=lambda t: t[1], reverse=True
        ):
            entry = doc_map[text_key]
            entry["rrf_score"] = score
            fused.append(entry)

        if top_n:
            fused = fused[:top_n]
        return fused

    # ── Utilities ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of documents currently in the corpus."""
        return len(self._tokenised_corpus)

    def reset(self) -> None:
        """Clear the index and metadata (in-memory only)."""
        self._tokenised_corpus.clear()
        self._metadata.clear()
        self._bm25 = None
        logger.info("BM25 index reset (in-memory).")
