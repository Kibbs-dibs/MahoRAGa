"""
Unit tests for the Attention-Guided Context Compressor (AttentionRAG) node.
"""

from core.state import AletheiaState, QueryState, MemoryState, EvaluationState, CompressionState, GenerationState
from graph.nodes.pruner import pruner_node, _extract_focal_token


def test_pruner_node_initialization():
    """Test that the pruner node initializes compression state correctly."""
    state = AletheiaState(
        query=QueryState(original_query="test query"),
        memory=None,
        evaluation=EvaluationState(),
        compression=None,
        generation=GenerationState()
    )

    result_state = pruner_node(state)

    assert result_state.compression is not None
    assert isinstance(result_state.compression, CompressionState)
    assert result_state.compression.final_pruned_prompt != ""
    assert result_state.compression.focal_token is not None


def test_pruner_node_passthrough():
    """Test that the pruner node handles state correctly."""
    state = AletheiaState(
        query=QueryState(original_query="test query", reformulated_query="test query"),
        memory=MemoryState(
            serialized_graph_triples=["Triple 1", "Triple 2"],
            vector_chunks=["Chunk 1", "Chunk 2"],
            lexical_chunks=["Lexical 1", "Lexical 2"]
        ),
        evaluation=EvaluationState(
            rubric_score=85,
            is_evidence_weak=False
        ),
        compression=None,
        generation=GenerationState()
    )

    result_state = pruner_node(state)

    assert result_state is state
    assert result_state.compression is not None
    assert "Triple 1" in result_state.compression.final_pruned_prompt
    assert "Chunk 1" in result_state.compression.final_pruned_prompt
    assert "Lexical 1" in result_state.compression.final_pruned_prompt


def test_pruner_focal_token_extraction():
    """Test that focal token extraction isolates the key technical concept."""
    token1 = _extract_focal_token("What is a Transformer?")
    assert "transformer" in token1

    token2 = _extract_focal_token("Explain how backpropagation works in deep neural networks")
    assert any(term in token2 for term in ["backpropagation", "neural", "deep"])


def test_pruner_compression_and_micro_windows():
    """Test that large context is segmented into micro-windows and compressed."""
    # Build a larger context > 1000 characters
    long_chunks = [
        f"This is passage sentence {i} discussing the Transformer self-attention architecture in detail."
        for i in range(25)
    ]
    state = AletheiaState(
        query=QueryState(original_query="How does the transformer self-attention mechanism work?"),
        memory=MemoryState(vector_chunks=long_chunks),
        evaluation=EvaluationState(rubric_score=90, is_evidence_weak=False),
        compression=CompressionState(),
        generation=GenerationState()
    )

    result = pruner_node(state)

    assert result.compression.focal_token == "transformer"
    assert len(result.compression.micro_windows) > 1
    assert result.compression.compression_ratio >= 1.0
    assert len(result.compression.final_pruned_prompt) < sum(len(c) for c in long_chunks)