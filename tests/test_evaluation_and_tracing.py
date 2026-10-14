"""
Automated Test Suite for Milestone 5: Observability Tracing & Ragas Benchmark (tests/test_evaluation_and_tracing.py)

Validates:
1. PipelineTracer telemetry recording, span tracking, and latency percentiles (p50, p95).
2. Ragas-style metric calculators: Faithfulness, Answer Relevance, Context Recall, Compression Fidelity.
3. Automated benchmark runner and Markdown scorecard generation.
4. FastAPI Observability & Evaluation endpoints (/api/v1/metrics, /api/v1/metrics/traces, /api/v1/evaluate).
"""

from __future__ import annotations

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from core.tracing import PipelineTracer, Span, TraceRecord, tracer
from evaluation.dataset import EvalSample
from evaluation.metrics import (
    calculate_answer_relevance,
    calculate_compression_fidelity,
    calculate_context_recall,
    calculate_faithfulness,
    calculate_ragas_scorecard,
)
from evaluation.benchmark import format_markdown_scorecard, run_benchmark


@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def test_pipeline_tracer_metrics():
    """Verify tracer records spans, computes percentiles, and tracks error rates."""
    custom_tracer = PipelineTracer(max_history=100)

    # Simulate 5 traces
    for i in range(1, 6):
        trace = custom_tracer.start_trace(f"Test query {i}")
        custom_tracer.record_span(trace, name="router", duration=0.05 * i)
        custom_tracer.record_span(trace, name="retriever", duration=0.10 * i)
        trace.finish(
            rubric_score=80 + i * 2,
            compression_ratio=2.0 + (0.5 * i),
            critic_verdict="Approved",
        )

    metrics = custom_tracer.get_aggregated_metrics()
    assert metrics["total_queries"] == 5
    assert metrics["latency_p50"] > 0.0
    assert metrics["avg_rubric_score"] > 80.0
    assert metrics["avg_compression_ratio"] > 2.0
    assert metrics["success_rate_pct"] == 100.0
    assert "router" in metrics["node_durations_avg"]
    assert "retriever" in metrics["node_durations_avg"]


def test_ragas_metrics_functions():
    """Verify math and boundary behavior of Ragas metric calculators."""
    # 1. Faithfulness
    facts = [
        "AttentionRAG uses an attention weight score to prune tokens.",
        "Compression ratio can reach up to 6.3x.",
    ]
    grounded_answer = "AttentionRAG applies attention weight scores to prune tokens, achieving up to 6.3x compression."
    score_grounded = calculate_faithfulness(grounded_answer, facts)
    assert score_grounded >= 0.70

    unrelated_answer = "Quantum computing uses superconducting qubits cooled to near absolute zero."
    score_unrelated = calculate_faithfulness(unrelated_answer, facts)
    assert score_unrelated < score_grounded

    # 2. Answer Relevance
    query = "What is the key mechanism of AttentionRAG?"
    ans = "AttentionRAG prunes context tokens using self-attention weights."
    relevance_score = calculate_answer_relevance(query, ans)
    assert relevance_score >= 0.50

    # 3. Context Recall
    entities = ["AttentionRAG", "compression", "tokens", "pruning"]
    contexts = [
        "The AttentionRAG framework uses attention weight scores to reduce and prune tokens while retaining factual fidelity.",
        "High compression ratio can reach up to 6.3x, significantly improving inference throughput.",
    ]
    recall_score = calculate_context_recall(facts, contexts)
    assert recall_score == 1.0

    # 4. Compression Fidelity
    pruned_prompt = "AttentionRAG framework retains tokens via attention pruning."
    fidelity_score = calculate_compression_fidelity(entities, pruned_prompt)
    assert fidelity_score >= 0.75

    # 5. Composite Scorecard
    scorecard = calculate_ragas_scorecard(
        query=query,
        answer=ans,
        retrieved_contexts=contexts,
        pruned_context=pruned_prompt,
        ground_truth_facts=facts,
        ground_truth_entities=entities,
    )
    assert 0.0 <= scorecard["composite_score"] <= 100.0
    assert scorecard["composite_score"] >= 60.0


def test_benchmark_runner_and_scorecard():
    """Verify automated benchmark runner generates summaries and markdown scorecards."""
    sample = EvalSample(
        sample_id="test_sample_01",
        query="What is the primary contribution of AttentionRAG?",
        target_document="AttentionRAG",
        reference_answer="AttentionRAG introduces attention-guided token pruning for efficient context compression.",
        ground_truth_facts=["Attention-guided token pruning", "Context compression"],
        ground_truth_entities=["AttentionRAG", "pruning", "compression"],
    )

    summary = run_benchmark([sample])
    assert summary["total_samples"] == 1
    assert "avg_composite_score" in summary
    assert "avg_faithfulness" in summary
    assert "avg_answer_relevance" in summary
    assert "avg_context_recall" in summary
    assert "avg_compression_fidelity" in summary
    assert len(summary["samples"]) == 1

    markdown = format_markdown_scorecard(summary)
    assert "# 🛡️ Project Aletheia — Ragas Evaluation Scorecard" in markdown
    assert "test_sample_01" in markdown
    assert "AttentionRAG" in markdown


def test_fastapi_observability_endpoints(client: TestClient):
    """Verify /api/v1/metrics and /api/v1/metrics/traces return observability data."""
    # Seed a trace
    trace = tracer.start_trace("Observability test query")
    tracer.record_span(trace, name="evaluator", duration=0.08)
    trace.finish(rubric_score=92, compression_ratio=4.5, critic_verdict="Approved")

    # 1. Test metrics
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    metrics = res.json()
    assert metrics["total_queries"] >= 1
    assert "latency_p50" in metrics
    assert "avg_rubric_score" in metrics
    assert "success_rate_pct" in metrics

    # 2. Test traces
    res_traces = client.get("/api/v1/metrics/traces?limit=10")
    assert res_traces.status_code == 200
    traces_list = res_traces.json()
    assert isinstance(traces_list, list)
    assert len(traces_list) >= 1
    assert traces_list[0]["query"] == "Observability test query"


def test_fastapi_evaluate_endpoint(client: TestClient):
    """Verify /api/v1/evaluate executes benchmark and returns structured response."""
    mock_summary = {
        "timestamp": "2026-10-14T12:00:00Z",
        "total_samples": 3,
        "avg_composite_score": 87.5,
        "avg_faithfulness": 0.88,
        "avg_answer_relevance": 0.85,
        "avg_context_recall": 0.90,
        "avg_compression_fidelity": 0.87,
        "avg_latency_seconds": 1.25,
        "samples": [
            {
                "sample_id": "test_1",
                "query": "Sample query",
                "target_document": "AttentionRAG",
                "faithfulness": 0.88,
                "answer_relevance": 0.85,
                "context_recall": 0.90,
                "compression_fidelity": 0.87,
                "composite_score": 87.5,
                "latency": 1.25,
            }
        ],
    }

    with patch("evaluation.benchmark.run_benchmark", return_value=mock_summary):
        res = client.post("/api/v1/evaluate")
        assert res.status_code == 200
        data = res.json()
        assert data["total_samples"] == 3
        assert data["avg_composite_score"] == 87.5
        assert data["avg_faithfulness"] == 0.88
        assert len(data["samples"]) == 1
