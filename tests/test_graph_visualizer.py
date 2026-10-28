"""
Tests for the Interactive Knowledge Graph Visualizer (tests/test_graph_visualizer.py).

Validates:
1. Triple string parsing into Vis.js nodes and edges.
2. Entity categorization and styling palettes.
3. HTML iframe generation with physics configuration.
4. UI integration with format_neo4j_markdown.
"""

from __future__ import annotations

import pytest

from core.state import AletheiaState, MemoryState, QueryState
from ui.graph_visualizer import (
    parse_triples_to_graph,
    generate_interactive_graph_html,
    _infer_entity_type,
)
from ui.app import format_neo4j_markdown


def test_infer_entity_type():
    """Verify semantic classification heuristics."""
    assert _infer_entity_type("AttentionRAG") == "Model"
    assert _infer_entity_type("HotpotQA Benchmark") == "Benchmark"
    assert _infer_entity_type("Compression Ratio") == "Metric"
    assert _infer_entity_type("Context Pruning") == "Mechanism"
    assert _infer_entity_type("Knowledge Base") == "Concept"


def test_parse_triples_to_graph():
    """Verify triple strings convert into deduplicated nodes and edges."""
    triples = [
        "AttentionRAG COMPRESSES Long Context with single-token focal attention",
        "AttentionRAG UTILIZES Next-Token Prediction paradigm",
        "AttentionRAG OUTPERFORMS LLMLingua",
        "LLMLingua UTILIZES Budget Controller",
    ]

    nodes, edges = parse_triples_to_graph(triples)

    assert len(nodes) >= 4
    assert len(edges) >= 4

    node_ids = {n["id"] for n in nodes}
    assert "AttentionRAG" in node_ids
    assert "LLMLingua" in node_ids

    # Verify edge connections
    edge_pairs = {(e["from"], e["to"]) for e in edges}
    assert ("AttentionRAG", "LLMLingua") in edge_pairs

    # Verify edge formatting
    for edge in edges:
        assert "arrows" in edge
        assert "color" in edge
        assert "label" in edge


def test_generate_interactive_graph_html():
    """Verify HTML canvas generation with iframe wrapper and Vis.js script."""
    nodes = [{"id": "Node1", "label": "Node 1"}]
    edges = [{"from": "Node1", "to": "Node2", "label": "links"}]

    html_out = generate_interactive_graph_html(nodes, edges, height_px=400)
    assert "<iframe" in html_out
    assert "vis-network" in html_out
    assert "Node1" in html_out

    # Empty graph
    empty_html = generate_interactive_graph_html([], [])
    assert "The graph is empty" in empty_html


def test_format_neo4j_markdown_integration():
    """Verify format_neo4j_markdown produces both interactive canvas and markdown bullets."""
    state = AletheiaState(
        query=QueryState(original_query="What is AttentionRAG?"),
        memory=MemoryState(
            serialized_graph_triples=[
                "AttentionRAG COMPRESSES Context with focal token isolation.",
                "AttentionRAG OUTPERFORMS Baseline models.",
            ]
        ),
    )

    result = format_neo4j_markdown(state)
    assert "<iframe" in result
    assert "AttentionRAG" in result
    assert "COMPRESSES" in result
    assert "Active Knowledge Graph Triples" in result

    # Offline / empty state test
    offline_result = format_neo4j_markdown(None)
    assert "offline" in offline_result.lower()
