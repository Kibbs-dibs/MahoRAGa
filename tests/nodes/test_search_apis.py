"""
Tests for the search_apis node.
"""

from core.state import AletheiaState, QueryState, MemoryState, EvaluationState, CompressionState, GenerationState
from graph.nodes.search_apis import search_apis_node


def test_search_apis_node_exists():
    """Test that the search_apis_node function exists and is callable."""
    assert callable(search_apis_node)


def test_search_apis_node_returns_state():
    """Test that the search_apis_node returns an AletheiaState object."""
    # Create a minimal state for testing with all required components
    state = AletheiaState(
        query=QueryState(original_query="test query"),
        memory=MemoryState(),
        evaluation=EvaluationState(),
        compression=CompressionState(),
        generation=GenerationState(),
    )

    # Call the node
    result_state = search_apis_node(state)

    # Verify we got back an AletheiaState
    assert isinstance(result_state, AletheiaState)
    assert result_state is state  # Should be the same object (direct manipulation)


def test_search_apis_node_skips_when_evidence_strong():
    """Test that search_apis_node skips search when evidence is strong."""
    # Create state with strong evidence (score >= 80)
    state = AletheiaState(
        query=QueryState(original_query="test query"),
        memory=MemoryState(),
        evaluation=EvaluationState(
            rubric_score=85,  # Strong evidence
            is_evidence_weak=False
        ),
        compression=CompressionState(),
        generation=GenerationState(),
    )

    # Call the node
    result_state = search_apis_node(state)

    # Should return the same state unchanged
    assert result_state is state
    assert result_state.evaluation.is_evidence_weak is False


def test_search_apis_node_runs_when_evidence_weak():
    """Test that search_apis_node runs when evidence is weak."""
    # Create state with weak evidence (score < 80)
    state = AletheiaState(
        query=QueryState(original_query="test query"),
        memory=MemoryState(),
        evaluation=EvaluationState(
            rubric_score=75,  # Weak evidence
            is_evidence_weak=True
        ),
        compression=CompressionState(),
        generation=GenerationState(),
    )

    # Call the node
    result_state = search_apis_node(state)

    # Should return the same state (placeholder implementation doesn't modify it)
    assert result_state is state
    assert result_state.evaluation.is_evidence_weak is True


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_search_apis_node_exists,
        test_search_apis_node_returns_state,
        test_search_apis_node_skips_when_evidence_strong,
        test_search_apis_node_runs_when_evidence_weak
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)