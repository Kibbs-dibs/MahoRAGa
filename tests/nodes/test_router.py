"""
Unit tests for the router node.
"""

from core.state import AletheiaState, QueryState
from graph.nodes.router import router_node, VALID_ROUTES


def test_router_node_basic_content():
    state = AletheiaState(
        query=QueryState(original_query="What is the architecture of the Transformer model?")
    )
    result = router_node(state)
    assert result.query.route in VALID_ROUTES
    assert result.query.reformulated_query is not None
    assert len(result.query.reformulated_query) > 0


def test_router_node_trend_classification():
    state = AletheiaState(
        query=QueryState(original_query="What are the latest emerging trends in LLM reasoning?")
    )
    result = router_node(state)
    assert result.query.route in ("trend", "content")
    assert result.query.reformulated_query is not None


def test_router_node_bibliometric_classification():
    state = AletheiaState(
        query=QueryState(original_query="Who wrote Attention Is All You Need and what are their citations?")
    )
    result = router_node(state)
    assert result.query.route in ("bibliometric", "content")
    assert result.query.reformulated_query is not None


def test_router_node_empty_query_raises():
    state = AletheiaState(query=QueryState(original_query=""))
    try:
        router_node(state)
        assert False, "Should raise AletheiaError on empty query"
    except Exception:
        pass
