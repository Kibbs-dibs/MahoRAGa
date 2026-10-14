"""
Unit tests for the generator node.
"""

from core.state import AletheiaState, QueryState, MemoryState, EvaluationState, CompressionState, GenerationState
from graph.nodes.generator import generator_node, _run_critic_audit
from core.exceptions import AletheiaError


def test_generator_node_initialization():
    """Test that generator node initializes properly and produces a cited draft answer."""
    state = AletheiaState(
        query=QueryState(original_query="What is a transformer?"),
        memory=MemoryState(),
        evaluation=EvaluationState(),
        compression=CompressionState(
            final_pruned_prompt="The Transformer model relies on self-attention mechanisms [Vaswani et al., 2017]."
        ),
        generation=GenerationState()
    )

    result_state = generator_node(state)

    assert result_state.generation is not None
    assert result_state.generation.draft_answer != ""
    # Should contain citation brackets
    assert "[" in result_state.generation.draft_answer and "]" in result_state.generation.draft_answer


def test_generator_node_with_empty_context():
    """Test generator node handles empty context gracefully by raising AletheiaError."""
    state = AletheiaState(
        query=QueryState(original_query="test query"),
        memory=MemoryState(),
        evaluation=EvaluationState(),
        compression=CompressionState(final_pruned_prompt=""),
        generation=GenerationState()
    )

    try:
        generator_node(state)
        assert False, "Expected AletheiaError for empty context"
    except AletheiaError as e:
        assert "No pruned context available" in str(e)


def test_generator_node_retry_logic():
    """Test that generator node respects the single regeneration limit."""
    state = AletheiaState(
        query=QueryState(original_query="test query"),
        memory=MemoryState(),
        evaluation=EvaluationState(),
        compression=CompressionState(final_pruned_prompt="Some relevant factual context [Ref, 2024]."),
        generation=GenerationState(retry_count=1)  # Already at max retries
    )

    result_state = generator_node(state)
    assert result_state.generation.retry_count <= 1


def test_generator_node_preserves_other_state():
    """Test that generator node preserves other state components."""
    original_query = "What is the latest trend in AI?"
    state = AletheiaState(
        query=QueryState(original_query=original_query),
        memory=MemoryState(serialized_graph_triples=["triple1", "triple2"]),
        evaluation=EvaluationState(rubric_score=85),
        compression=CompressionState(
            final_pruned_prompt="Compressed context about AI trends [Author, 2024]",
            compression_ratio=0.5
        ),
        generation=GenerationState()
    )

    result_state = generator_node(state)

    assert result_state.query.original_query == original_query
    assert len(result_state.memory.serialized_graph_triples) == 2
    assert result_state.evaluation.rubric_score == 85
    assert result_state.compression.compression_ratio == 0.5


def test_critic_audit_validation():
    """Test that critic audit detects missing citations."""
    context = "BERT uses masked language modeling [Devlin et al., 2018]."
    answer_with_cites = "BERT utilizes masked language modeling [Devlin et al., 2018]."
    answer_without_cites = "BERT is a great model without any source citation."

    passed_cited, _, _ = _run_critic_audit(context, answer_with_cites)
    assert passed_cited is True

    passed_uncited, feedback, _ = _run_critic_audit(context, answer_without_cites)
    assert passed_uncited is False
    assert feedback is not None