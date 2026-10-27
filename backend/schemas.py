"""
Pydantic Request and Response Schemas for Project Aletheia API (backend/schemas.py)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for submitting a literature research question to the agent pipeline."""
    query: str = Field(..., description="Academic question or literature review prompt")
    session_id: Optional[str] = Field(None, description="Optional persistent conversation session ID")
    llm_provider: str = Field("Local Ollama", description="LLM provider: Local Ollama, OpenAI API, Anthropic Proxy, Offline Fallback")
    local_model: str = Field("qwen2.5:7b", description="Model identifier")
    evidence_threshold: float = Field(80.0, ge=0.0, le=100.0, description="Minimum evidence score before fallback triggering")
    max_compression_ratio: float = Field(6.3, ge=1.0, le=15.0, description="Maximum AttentionRAG pruning factor")
    enable_external_search: bool = Field(True, description="Enable fallback search APIs if memory evidence is weak")


class SourceItem(BaseModel):
    """Retrieved evidence item."""
    source_type: str
    preview: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Synthesized research review with evidence provenance and metrics."""
    session_id: str
    answer: str
    evidence_score: int
    compression_ratio: float
    critic_verdict: str
    latency_seconds: float
    triples: List[str] = Field(default_factory=list)
    vector_chunks: List[str] = Field(default_factory=list)
    lexical_chunks: List[str] = Field(default_factory=list)
    external_abstracts: List[str] = Field(default_factory=list)
    # Evidence transparency — per-component rubric breakdown (Feature 2.3)
    evidence_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="100-pt rubric sub-scores: retrieval_confidence, answer_specificity, "
                    "source_diversity, metadata_completeness, recency.",
    )
    # Structured click-through citations (Feature 4.2): tag -> source text
    source_map: Dict[str, str] = Field(default_factory=dict)
    # Degraded-state transparency: which memory layers were unavailable, so the
    # UI can warn that the answer came from a reduced evidence base.
    degraded: bool = Field(
        default=False,
        description="True if one or more retrieval layers were unavailable for this query.",
    )
    degraded_layers: List[str] = Field(
        default_factory=list,
        description="Memory layers that failed/were unavailable (e.g. neo4j, faiss, bm25).",
    )


class MessageItem(BaseModel):
    """Single chat message in conversation history."""
    id: str
    session_id: str
    role: str
    content: str
    rubric_score: Optional[int] = None
    compression_ratio: Optional[float] = None
    critic_verdict: Optional[str] = None
    latency: Optional[float] = None
    created_at: str


class SessionCreate(BaseModel):
    """Payload to initiate a research session."""
    title: Optional[str] = None


class SessionDetail(BaseModel):
    """Full session details with chat history."""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageItem] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Result of uploading and indexing academic literature."""
    status: str
    ingested_count: int
    skipped_duplicates: int
    new_chunks: int
    total_documents: int
    total_chunks: int
    neo4j_status: str
    message: str


class HealthResponse(BaseModel):
    """System health check and component status."""
    status: str
    version: str = "1.0.0"
    faiss_ready: bool
    bm25_ready: bool
    neo4j_online: bool
    total_documents: int
    total_chunks: int


class MetricsResponse(BaseModel):
    """Aggregated pipeline observability telemetry and latency percentiles."""
    total_queries: int
    latency_p50: float
    latency_p95: float
    avg_rubric_score: float
    avg_compression_ratio: float
    success_rate_pct: float
    node_durations_avg: Dict[str, float] = Field(default_factory=dict)


class BenchmarkResponse(BaseModel):
    """Ragas-style evaluation benchmark output and metric aggregates."""
    timestamp: str
    total_samples: int
    avg_composite_score: float
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_recall: float
    avg_compression_fidelity: float
    avg_latency_seconds: float
    samples: List[Dict[str, Any]] = Field(default_factory=list)

