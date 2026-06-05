"""
Pydantic BaseSettings for Project Aletheia.

Reads from environment variables and .env file at project root.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Centralised configuration consumed by every layer of the pipeline.
    Values are loaded from environment variables / .env file.
    """

    # ── Neo4j ───────────────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt connection URI.",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j authentication username.",
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j authentication password.",
    )

    # ── FAISS ───────────────────────────────────────────────────────────
    faiss_index_path: str = Field(
        default="data/faiss_index.bin",
        description="Filesystem path for the serialized FAISS index.",
    )
    faiss_metadata_path: str = Field(
        default="data/faiss_metadata.json",
        description="Filesystem path for the FAISS chunk-text metadata store.",
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers model for 384-dim dense embeddings.",
    )
    embedding_dim: int = Field(
        default=384,
        description="Dimensionality of the dense embedding vectors.",
    )

    # ── BM25 ────────────────────────────────────────────────────────────
    bm25_index_path: str = Field(
        default="data/bm25_index.pkl",
        description="Filesystem path for the serialized BM25 index.",
    )
    bm25_metadata_path: str = Field(
        default="data/bm25_metadata.json",
        description="Filesystem path for the BM25 document-text metadata store.",
    )

    # ── Cross-Encoder Re-ranking ────────────────────────────────────────
    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model for re-ranking.",
    )

    # ── LLM / Ollama ────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama local inference server base URL.",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Default local LLM model identifier.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for cloud LLM fallback.",
    )

    # ── External Academic APIs ──────────────────────────────────────────
    semantic_scholar_api_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional, improves rate limits).",
    )

    # ── Retrieval Thresholds ────────────────────────────────────────────
    evidence_threshold: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Minimum rubric score before external search is triggered.",
    )
    drift_guard_min_overlap: float = Field(
        default=0.30,
        description="Minimum term-overlap ratio for drift guard during agentic retry.",
    )

    # ── PPR (Personalized PageRank) ─────────────────────────────────────
    ppr_alpha: float = Field(
        default=0.85,
        description="Damping factor for Personalized PageRank.",
    )
    ppr_top_k: int = Field(
        default=10,
        description="Number of top nodes to return from PPR.",
    )

    # ── General ─────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=3500,
        description="Target character length for structure-aware chunking.",
    )
    top_k_retrieval: int = Field(
        default=10,
        description="Default number of results for vector / lexical retrieval.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()
