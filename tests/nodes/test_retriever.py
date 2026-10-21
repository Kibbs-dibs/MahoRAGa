"""
Unit tests for the Memory-Guided Retriever node.

These tests monkeypatch VectorStore, SparseStore, and Neo4jClient so no real
FAISS/BM25 indices or Neo4j database are required. Heavy third-party modules
are stubbed via tests/nodes/conftest.py before the imports below resolve.
"""

import graph.nodes.retriever as retriever_mod
from graph.nodes.retriever import retriever_node
from core.exceptions import SparseStoreError
from core.state import AletheiaState, QueryState, MemoryState


# ── Test doubles ────────────────────────────────────────────────────────


class _FakeVectorStore:
    """Returns a fixed list of dense-retrieval result dicts."""

    def __init__(self, *args, **kwargs):
        pass

    def load_if_exists(self):
        return True

    def search(self, query, top_k=None):
        return [
            {"text": "shared chunk about transformers", "score": 0.9, "rank": 1},
            {"text": "dense only chunk", "score": 0.5, "rank": 2},
        ]


class _FakeSparseStore:
    """Returns a fixed list of lexical-retrieval result dicts.

    reciprocal_rank_fusion is inherited behaviour we exercise via the real
    SparseStore static method, so we delegate to it. rerank raises
    SparseStoreError to simulate an offline cross-encoder by default.
    """

    rerank_should_fail = True

    def __init__(self, *args, **kwargs):
        pass

    def load_if_exists(self):
        return True

    def search(self, query, top_k=None):
        return [
            {"text": "shared chunk about transformers", "score": 3.1, "rank": 1},
            {"text": "lexical only chunk", "score": 1.2, "rank": 2},
        ]

    def rerank(self, query, candidates, top_k=None):
        if type(self).rerank_should_fail:
            raise SparseStoreError("cross-encoder unavailable (offline)")
        # Deterministic rerank: reverse the fused order.
        return list(reversed(candidates))[: (top_k or len(candidates))]

    # RRF is a static method on the real class; mirror it so retriever's
    # SparseStore.reciprocal_rank_fusion(...) call works against the double.
    reciprocal_rank_fusion = staticmethod(
        __import__(
            "memory.sparse_store", fromlist=["SparseStore"]
        ).SparseStore.reciprocal_rank_fusion
    )


class _FakeNeo4jClient:
    """Context-manager client that serves PPR + serialisation results."""

    ppr_results = [
        {
            "node_id": 1,
            "labels": ["FACT_TRIPLE"],
            "score": 0.9,
            "properties": {
                "head_entity": "Transformer",
                "relation": "uses",
                "tail_entity": "Self-Attention",
                "similarity_weight": 0.95,
            },
        }
    ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ensure_indexes(self):
        pass

    def run_personalised_pagerank(self, seed_entity, alpha=None, top_k=None):
        return self.ppr_results

    def serialise_subgraph_to_sentences(self, entity, limit=30):
        return [f"{entity} relates to concept (weight=0.50)."]


def _patch_stores(monkeypatch, sparse_fail=True):
    _FakeSparseStore.rerank_should_fail = sparse_fail
    monkeypatch.setattr(retriever_mod, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(retriever_mod, "SparseStore", _FakeSparseStore)
    monkeypatch.setattr(retriever_mod, "Neo4jClient", _FakeNeo4jClient)


def _make_state():
    return AletheiaState(
        query=QueryState(
            original_query="What does the Transformer use?",
            reformulated_query="What does the Transformer use?",
        ),
        memory=MemoryState(),
    )


# ── Tests ─────────────────────────────────────────────────────────────


def test_fused_chunks_populated_and_deduplicated(monkeypatch):
    """RRF fusion populates fused_chunks and dedupes the overlapping text."""
    _patch_stores(monkeypatch, sparse_fail=True)
    result = retriever_node(_make_state())

    fused = result.memory.fused_chunks
    assert fused, "fused_chunks should be populated"
    # The shared chunk appears in both FAISS and BM25 but must be deduped.
    assert fused.count("shared chunk about transformers") == 1
    # Distinct chunks from each layer are present.
    assert "dense only chunk" in fused
    assert "lexical only chunk" in fused
    # The overlapping doc wins RRF and should rank first.
    assert fused[0] == "shared chunk about transformers"


def test_per_layer_fields_still_populated(monkeypatch):
    """The additive fused field must not cannibalise the per-layer fields."""
    _patch_stores(monkeypatch, sparse_fail=True)
    result = retriever_node(_make_state())

    assert result.memory.vector_chunks == [
        "shared chunk about transformers",
        "dense only chunk",
    ]
    assert result.memory.lexical_chunks == [
        "shared chunk about transformers",
        "lexical only chunk",
    ]
    # PPR-seeded graph triple serialised from a FACT_TRIPLE node.
    assert "Transformer uses Self-Attention (weight=0.95)." in (
        result.memory.serialized_graph_triples
    )


def test_cross_encoder_unavailable_falls_back_to_rrf(monkeypatch):
    """When rerank raises SparseStoreError the node degrades, not crashes."""
    _patch_stores(monkeypatch, sparse_fail=True)
    # Should not raise.
    result = retriever_node(_make_state())
    assert result.memory.fused_chunks  # RRF order preserved


def test_cross_encoder_available_applies_rerank(monkeypatch):
    """When the cross-encoder is available its ordering is used."""
    _patch_stores(monkeypatch, sparse_fail=False)
    result = retriever_node(_make_state())
    fused = result.memory.fused_chunks
    assert fused
    # Fake rerank reverses RRF order, so the top RRF doc is no longer first.
    assert fused[0] != "shared chunk about transformers"
    assert "shared chunk about transformers" in fused
