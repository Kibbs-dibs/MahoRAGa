"""
Tests for the Project Aletheia Gradio Web UI (ui/app.py).

Validates:
1. Gradio Blocks app instantiation and component initialization.
2. Stepper HTML, metrics panel, and markdown formatters.
3. Stream generator handling empty queries and populated states.
"""

from __future__ import annotations

import pytest
import gradio as gr

from core.state import (
    AletheiaState,
    QueryState,
    MemoryState,
    EvaluationState,
    CompressionState,
    GenerationState,
)
from ui.app import (
    create_ui,
    render_stepper_html,
    render_metrics_html,
    format_neo4j_markdown,
    format_faiss_markdown,
    format_bm25_markdown,
    format_external_markdown,
    stream_query_pipeline,
)


def test_ui_instantiation():
    """Verify create_ui returns a valid Gradio Blocks instance."""
    app = create_ui()
    assert isinstance(app, gr.Blocks)
    assert app.title == "Project Aletheia — Autonomous AI Research"


def test_stepper_html_rendering():
    """Verify pipeline stepper renders clean HTML for different states."""
    idle_html = render_stepper_html()
    assert "stepper-container" in idle_html
    assert "1. Router" in idle_html
    assert "6. Generator & Critic" in idle_html

    active_html = render_stepper_html(
        active_node="retriever",
        completed_nodes=["router"],
        elapsed_time=1.5,
        status_label="Searching Memory",
    )
    assert "step-active" in active_html
    assert "step-done" in active_html
    assert "1.5s" in active_html


def test_metrics_html_rendering():
    """Verify metrics panel handles both None and populated states."""
    empty_metrics = render_metrics_html(None)
    assert "Evidence Rubric" in empty_metrics
    assert "Compression Ratio" in empty_metrics

    state = AletheiaState(
        query=QueryState(original_query="What is AttentionRAG?"),
        compression=CompressionState(compression_ratio=5.2),
        evaluation=EvaluationState(rubric_score=88),
        generation=GenerationState(
            draft_answer="AttentionRAG utilizes...",
            critic_feedback="Approved by Critic",
        ),
        pipeline_cost=0.0015,
    )
    populated_metrics = render_metrics_html(state, elapsed=2.1)
    assert "88/100" in populated_metrics
    assert "5.2x" in populated_metrics
    assert "2.10s" in populated_metrics
    assert "Approved" in populated_metrics


def test_sources_formatters():
    """Verify sources formatters produce markdown tables/cards."""
    state = AletheiaState(
        query=QueryState(original_query="test"),
        memory=MemoryState(
            vector_chunks=[
                "[AttentionRAG | Section: Methods | p. 3] AttentionRAG extracts micro-windows."
            ],
            lexical_chunks=[
                "[MemGraphRAG | Section: Architecture | p. 2] MemGraphRAG maintains a hierarchical graph."
            ],
            serialized_graph_triples=[
                "AttentionRAG COMPRESSES Context with focal token isolation."
            ],
            external_api_abstracts=[
                "TechGraphRAG demonstrates multi-agent graph reasoning."
            ],
        ),
    )

    neo4j_md = format_neo4j_markdown(state)
    assert "AttentionRAG" in neo4j_md
    assert "COMPRESSES" in neo4j_md

    faiss_md = format_faiss_markdown(state)
    assert "AttentionRAG" in faiss_md
    assert "Chunk #1" in faiss_md

    bm25_md = format_bm25_markdown(state)
    assert "MemGraphRAG" in bm25_md
    assert "Lexical Match #1" in bm25_md

    ext_md = format_external_markdown(state)
    assert "TechGraphRAG" in ext_md

    # Test empty state fallbacks
    assert "offline" in format_neo4j_markdown(None).lower()
    assert "No FAISS" in format_faiss_markdown(None)
    assert "No BM25" in format_bm25_markdown(None)


def test_stream_empty_query():
    """Verify stream generator cleanly ignores empty query strings."""
    gen = stream_query_pipeline(
        message="",
        history=[],
        llm_provider="Local Ollama",
        local_model="qwen2.5:7b",
        evidence_threshold=80.0,
        enable_external_search=True,
        max_compression_ratio=6.3,
    )
    results = list(gen)
    assert len(results) == 1
    hist, clean_inp, stepper, neo, faiss, bm25, ext, metrics = results[0]
    assert hist == []
    assert clean_inp == ""
