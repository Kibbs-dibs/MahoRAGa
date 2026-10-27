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
        default="",
        description=(
            "Neo4j authentication password. No usable default — set NEO4J_PASSWORD "
            "in the environment/.env. An empty value means Neo4j auth is unconfigured "
            "(the client will fail loudly on connect rather than using a weak default)."
        ),
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
    cross_encoder_rerank: bool = Field(
        default=True,
        description=(
            "Re-rank the RRF-fused retrieval candidates with the cross-encoder for "
            "final precision (Feature 1.2). Adds latency; degrades gracefully to RRF-only "
            "if the model cannot load. Set false for the fastest/local path."
        ),
    )
    rerank_candidate_pool: int = Field(
        default=20,
        description="How many top fused candidates to cross-encoder re-rank before taking top_k.",
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
    openai_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI API base URL (allows local proxies or compatible endpoints).",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model name.",
    )
    anthropic_base_url: Optional[str] = Field(
        default=None,
        description="Anthropic / NIM proxy base URL (e.g. from cc-nim).",
    )
    anthropic_auth_token: Optional[str] = Field(
        default=None,
        description="Anthropic / NIM proxy authorization token.",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key.",
    )
    anthropic_model: Optional[str] = Field(
        default=None,
        description="Anthropic / NIM proxy model identifier.",
    )
    max_compression_ratio: float = Field(
        default=6.3,
        description="Target maximum compression ratio for AttentionRAG context pruning.",
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
    persist_external_abstracts: bool = Field(
        default=False,
        description=(
            "When true, vetted external-search abstracts are written into FAISS + BM25 "
            "so the corpus self-enriches from use (Feature 5.1). Off by default because "
            "it mutates the persistent corpus; enable for a learning deployment."
        ),
    )
    llm_fact_extraction: bool = Field(
        default=False,
        description=(
            "When true, knowledge-graph fact extraction uses the LLM (structured triples) "
            "instead of the regex heuristic (Feature 6.1), falling back to regex on failure "
            "or offline. Off by default since it adds LLM cost at ingest time."
        ),
    )
    semantic_pruning: bool = Field(
        default=True,
        description=(
            "When true, the AttentionRAG pruner blends embedding cosine similarity "
            "(query vs sentence) into sentence relevance for genuinely semantic "
            "compression (Feature 3.1), falling back to lexical scoring if the encoder "
            "is unavailable. Set false for the fastest lexical-only path."
        ),
    )

    # ── Resilience / Failure Handling ───────────────────────────────────
    request_timeout_seconds: float = Field(
        default=90.0,
        description=(
            "Hard wall-clock ceiling for a single /query pipeline run. On expiry the "
            "endpoint returns HTTP 503 (service unavailable / overloaded) with a clear "
            "message instead of hanging or emitting an opaque 500. Guards against a "
            "wedged backend or runaway retrieval holding the request open indefinitely."
        ),
    )
    max_concurrent_queries: int = Field(
        default=4,
        description=(
            "Maximum pipeline runs executing at once. A memory-heavy pipeline (encoders, "
            "cross-encoder, FAISS) can OOM the host under a burst, so excess requests "
            "wait briefly for a slot and are rejected with HTTP 503 if none frees within "
            "query_admission_wait_seconds — back-pressure instead of a crash."
        ),
    )
    query_admission_wait_seconds: float = Field(
        default=10.0,
        description="How long a query waits for a concurrency slot before returning 503.",
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
        default=1000,
        description=(
            "Target character length for structure-aware chunking. Kept close to "
            "the dense embedder's input window (all-MiniLM-L6-v2 truncates at ~256 "
            "word-pieces, roughly 1000 chars) so FAISS embeds the whole chunk "
            "rather than only its opening. Raise only if you also switch to a "
            "longer-context embedding model."
        ),
    )
    top_k_retrieval: int = Field(
        default=10,
        description="Default number of results for vector / lexical retrieval.",
    )

    # ── API / CORS ──────────────────────────────────────────────────────
    cors_allow_origins: str = Field(
        default="*",
        description=(
            "Comma-separated allowed CORS origins for the FastAPI backend. "
            "'*' allows any origin but DISABLES credentialed requests (safe default). "
            "Set explicit origins (e.g. 'http://localhost:7860') to enable credentials."
        ),
    )

    # ── API authentication (scoped) ─────────────────────────────────────
    aletheia_api_key: Optional[str] = Field(
        default=None,
        description=(
            "Shared secret (bearer token) required for write/destructive REST "
            "routes (ingest, corpus wipe/retry, watcher sync, evaluate, session "
            "create/delete). Read endpoints stay open. When UNSET, those write "
            "routes FAIL CLOSED (HTTP 503) on a network-exposed bind so a "
            "misconfigured deployment cannot be wiped by anonymous callers; on a "
            "loopback bind they are allowed for frictionless local dev."
        ),
    )
    expose_api_docs: bool = Field(
        default=True,
        description=(
            "Expose interactive OpenAPI docs at /docs and /redoc. Set false for a "
            "network-exposed deployment to avoid advertising the full API surface "
            "(defense-in-depth; does not replace authentication)."
        ),
    )
    aletheia_ui_auth_enabled: bool = Field(
        default=False,
        description=(
            "Require a username/password login for the Gradio UI mounted at '/'. "
            "The UI's ingest/wipe/arxiv buttons call the service layer directly "
            "(not the gated REST routes), so enable this when exposing the UI."
        ),
    )
    aletheia_ui_username: str = Field(
        default="admin",
        description="Username for the Gradio UI login (when ui auth is enabled).",
    )
    aletheia_ui_password: Optional[str] = Field(
        default=None,
        description="Password for the Gradio UI login (when ui auth is enabled).",
    )
    web_ui_host: str = Field(
        default="127.0.0.1",
        description=(
            "Interface the unified server binds to. Defaults to loopback "
            "(127.0.0.1) so a fresh run is not network-exposed by accident. Set "
            "'0.0.0.0' explicitly to expose on all interfaces."
        ),
    )

    # ── Upload limits ───────────────────────────────────────────────────
    max_upload_bytes: int = Field(
        default=25 * 1024 * 1024,
        description=(
            "Maximum accepted size (bytes) for a single uploaded PDF. Guards "
            "against memory exhaustion / disk-fill from a network client. "
            "Default 25 MB."
        ),
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()
