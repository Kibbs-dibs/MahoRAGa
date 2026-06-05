"""
Custom exception classes for Project Aletheia.
"""


class AletheiaError(Exception):
    """Base exception for the entire Project Aletheia pipeline."""


# ── Memory / Storage Layer ──────────────────────────────────────────────


class Neo4jConnectionError(AletheiaError):
    """Raised when the Neo4j driver cannot connect or a session fails."""


class Neo4jQueryError(AletheiaError):
    """Raised when a Cypher query execution fails."""


class VectorStoreError(AletheiaError):
    """Raised for FAISS index load/save/query failures."""


class SparseStoreError(AletheiaError):
    """Raised for BM25 index load/save/query failures."""


class EmbeddingError(AletheiaError):
    """Raised when the SentenceTransformers embedding model fails."""


# ── Retrieval / Evaluation ──────────────────────────────────────────────


class RetrievalError(AletheiaError):
    """Raised when the retrieval pipeline encounters a non-recoverable error."""


class EvidenceInsufficient(AletheiaError):
    """Raised when evidence falls below the rubric threshold and external search also fails."""


# ── Ingestion ───────────────────────────────────────────────────────────


class IngestionError(AletheiaError):
    """Raised during PDF parsing or chunk extraction failures."""


class ConflictResolutionError(AletheiaError):
    """Raised when the conflict resolution agent cannot reconcile triples."""


# ── Generation ──────────────────────────────────────────────────────────


class GenerationError(AletheiaError):
    """Raised when the LLM generator or critic agent fails."""
