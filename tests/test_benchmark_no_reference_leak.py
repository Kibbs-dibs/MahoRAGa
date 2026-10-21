"""
Regression test for evaluation/benchmark.py (FEAT-002, finding A4).

Guards against the ground-truth reference answer leaking into the scored
answer. The pipeline is stubbed so these tests run without the heavy ML
stack (langgraph / neo4j / faiss / sentence-transformers / torch).
"""

from __future__ import annotations

import sys
import types

# ── Stub heavy modules BEFORE importing evaluation.benchmark ────────────
# benchmark -> graph.orchestrator (langgraph) -> graph.nodes -> memory.*
# (neo4j, faiss, numpy, sentence_transformers, torch).

def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)
    return sys.modules[name]


def _ensure_heavy_stubs():
    _stub("neo4j", GraphDatabase=type("GraphDatabase", (), {}), Session=type("Session", (), {}))
    _stub(
        "neo4j.exceptions",
        AuthError=type("AuthError", (Exception,), {}),
        ServiceUnavailable=type("ServiceUnavailable", (Exception,), {}),
        SessionExpired=type("SessionExpired", (Exception,), {}),
    )
    _stub("faiss")
    _stub("torch")
    _stub(
        "sentence_transformers",
        SentenceTransformer=type("SentenceTransformer", (), {}),
        CrossEncoder=type("CrossEncoder", (), {}),
    )
    # langgraph: the orchestrator builds + compiles a StateGraph at import time,
    # so the stub must accept the builder calls. We never invoke the compiled
    # graph in these tests (run_aletheia_pipeline is monkeypatched).
    class _FakeCompiled:
        def invoke(self, state):  # pragma: no cover - never called here
            return state

    class _FakeStateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *args, **kwargs):
            return None

        def set_entry_point(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

        def add_conditional_edges(self, *args, **kwargs):
            return None

        def compile(self, *args, **kwargs):
            return _FakeCompiled()

    _stub("langgraph")
    _stub("langgraph.graph", StateGraph=_FakeStateGraph, END="__END__")
    try:
        import numpy  # noqa: F401
    except Exception:  # pragma: no cover - only when numpy absent
        _stub("numpy")


_ensure_heavy_stubs()

import evaluation.benchmark as benchmark  # noqa: E402
from core.state import (  # noqa: E402
    AletheiaState,
    QueryState,
    MemoryState,
    EvaluationState,
    CompressionState,
    GenerationState,
)
from evaluation.dataset import EvalSample  # noqa: E402


_SAMPLE = EvalSample(
    sample_id="eval_leak",
    query="What does AttentionRAG do?",
    reference_answer="SENTINEL_REFERENCE_ANSWER_SHOULD_NOT_BE_SCORED",
    ground_truth_entities=["AttentionRAG"],
    ground_truth_facts=["compresses context"],
    target_document="AttentionRAG",
)


def _empty_final_state() -> AletheiaState:
    return AletheiaState(
        query=QueryState(original_query=_SAMPLE.query),
        memory=MemoryState(),
        evaluation=EvaluationState(rubric_score=0),
        compression=CompressionState(),
        generation=GenerationState(draft_answer=""),
    )


def test_empty_generation_not_replaced_by_reference(monkeypatch):
    """An empty generated answer must stay empty and be flagged as failed."""
    monkeypatch.setattr(
        benchmark, "run_aletheia_pipeline", lambda state: _empty_final_state()
    )

    summary = benchmark.run_benchmark([_SAMPLE])
    result = summary["samples"][0]

    # The reference answer must never be substituted for a missing generation.
    assert result["answer"] == ""
    assert _SAMPLE.reference_answer not in result["answer"]
    assert result["generation_failed"] is True


def test_pipeline_exception_marked_failed(monkeypatch):
    """A raising pipeline is recorded as failed, not silently scored empty."""
    def _boom(state):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(benchmark, "run_aletheia_pipeline", _boom)

    summary = benchmark.run_benchmark([_SAMPLE])
    result = summary["samples"][0]

    assert result["generation_failed"] is True
    assert "pipeline exploded" in (result["pipeline_error"] or "")


def test_real_generation_is_scored(monkeypatch):
    """A genuine non-empty generation is scored and not flagged as failed."""
    def _good(state):
        final = _empty_final_state()
        final.generation.draft_answer = "AttentionRAG compresses context using attention."
        return final

    monkeypatch.setattr(benchmark, "run_aletheia_pipeline", _good)

    summary = benchmark.run_benchmark([_SAMPLE])
    result = summary["samples"][0]

    assert result["generation_failed"] is False
    assert result["pipeline_error"] is None
