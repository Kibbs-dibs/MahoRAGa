"""
End-to-End Pipeline Integration Tests for Project Aletheia.

Tests both synchronous and asynchronous pipeline execution paths across:
Router -> Retriever -> Evaluator -> Search APIs -> Pruner -> Generator.
"""

import pytest
from core.state import (
    AletheiaState,
    QueryState,
    MemoryState,
    EvaluationState,
    CompressionState,
    GenerationState,
)
from graph.orchestrator import run_aletheia_pipeline, run_aletheia_pipeline_async


def test_pipeline_sync_execution():
    """Verify synchronous pipeline completes end-to-end and populates all layers."""
    state = AletheiaState(
        query=QueryState(original_query="Explain the self-attention mechanism in the Transformer architecture."),
        memory=MemoryState(),
        evaluation=EvaluationState(rubric_score=90, is_evidence_weak=False),
        compression=CompressionState(),
        generation=GenerationState(),
    )

    result = run_aletheia_pipeline(state)

    # 1. Router verification
    assert result["query"].route in ("content", "trend", "bibliometric", "current_world")
    assert result["query"].reformulated_query is not None

    # 2. AttentionRAG Pruner verification
    assert result["compression"].focal_token is not None
    assert result["compression"].compression_ratio >= 1.0
    assert result["compression"].final_pruned_prompt != ""

    # 3. Generator & Critic verification
    assert result["generation"].draft_answer != ""
    assert result["generation"].retry_count <= 1

    # 4. Pipeline Cost
    assert result["pipeline_cost"] >= 0.0


@pytest.mark.asyncio
async def test_pipeline_async_execution():
    """Verify asynchronous pipeline completes end-to-end via ainvoke."""
    state = AletheiaState(
        query=QueryState(original_query="Who wrote Attention Is All You Need?"),
        memory=MemoryState(),
        evaluation=EvaluationState(rubric_score=85, is_evidence_weak=False),
        compression=CompressionState(),
        generation=GenerationState(),
    )

    result = await run_aletheia_pipeline_async(state)

    assert result["query"].route in ("bibliometric", "content", "trend")
    assert result["compression"].focal_token is not None
    assert result["generation"].draft_answer != ""
    assert result["pipeline_cost"] >= 0.0
