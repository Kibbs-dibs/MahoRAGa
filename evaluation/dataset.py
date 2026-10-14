"""
Benchmark Evaluation Dataset for Project Aletheia (evaluation/dataset.py)

Curated academic research benchmark questions, expected facts, and reference answers
designed to evaluate RAG Faithfulness, Answer Relevance, and Attention Pruning Fidelity.
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """Single benchmark evaluation sample."""
    sample_id: str
    query: str
    reference_answer: str
    ground_truth_entities: List[str] = Field(default_factory=list)
    ground_truth_facts: List[str] = Field(default_factory=list)
    target_document: str


BENCHMARK_DATASET: List[EvalSample] = [
    EvalSample(
        sample_id="eval_001",
        query="How does AttentionRAG compress context while preserving citation fidelity?",
        reference_answer=(
            "AttentionRAG reformulates queries into a next-token prediction paradigm using an answer hint prefix. "
            "It isolates the semantic focus to a single anchor token, computes attention features across context chunks, "
            "and selects sentences with top-k attended tokens, achieving up to 6.3x compression."
        ),
        ground_truth_entities=[
            "AttentionRAG", "focal token", "next-token prediction", "6.3x", "LLMLingua"
        ],
        ground_truth_facts=[
            "reformulates queries into next-token prediction",
            "isolates attention to single token",
            "achieves up to 6.3x context compression",
        ],
        target_document="AttentionRAG",
    ),
    EvalSample(
        sample_id="eval_002",
        query="What are the core components of the Three-Layer Global Memory architecture?",
        reference_answer=(
            "The Three-Layer Global Memory consists of: 1) Neo4j Hierarchical Indexing Graph "
            "for ontology schemas, fact triples, and passage provenance; 2) FAISS 384-dimensional IndexFlatIP "
            "dense vector store for semantic similarity; and 3) Disk-Backed BM25Okapi sparse lexical store "
            "with Cross-Encoder reranking."
        ),
        ground_truth_entities=[
            "Neo4j", "FAISS", "BM25", "IndexFlatIP", "Cross-Encoder", "Personalized PageRank"
        ],
        ground_truth_facts=[
            "Neo4j stores ontology schemas and fact triples",
            "FAISS manages 384-dimensional dense vectors",
            "BM25 provides sparse lexical retrieval with cross-encoder reranking",
        ],
        target_document="MemGraphRAG",
    ),
    EvalSample(
        sample_id="eval_003",
        query="How does TechGraphRAG utilize multi-agent retrieval for technical domains?",
        reference_answer=(
            "TechGraphRAG employs agentic graph traversal where specialized retrieval agents "
            "navigate domain ontology relationships, expand entity neighborhoods via Personalized PageRank, "
            "and fuse dual-index lexical and dense search results using Reciprocal Rank Fusion."
        ),
        ground_truth_entities=[
            "TechGraphRAG", "multi-agent", "ontology", "PageRank", "Reciprocal Rank Fusion"
        ],
        ground_truth_facts=[
            "agents navigate domain ontology relationships",
            "uses Personalized PageRank for diffusion",
            "fuses dual-index results with Reciprocal Rank Fusion",
        ],
        target_document="TechGraphRAG",
    ),
]
