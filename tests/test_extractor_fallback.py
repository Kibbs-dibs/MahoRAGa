"""
Focused unit tests for ingestion.extractor fact extraction.

Under the offline deterministic LLM stub, ``llm_client.call`` does not return
extraction JSON for the extractor prompt shape, so ``extract_from_chunk`` must
fall back to the regex extractor and still produce facts. These tests run on the
lightweight interpreter without PyMuPDF by building a synthetic Chunk directly.

``ingestion.chunker`` imports ``ingestion.parser``, which does ``import fitz``
at module top. A minimal ``fitz`` stub is installed before those imports so
this test is self-contained and passes in isolation regardless of collection
order.
"""

import sys
import types

# Stub fitz so importing ingestion.parser (via ingestion.chunker) does not
# require PyMuPDF. Must run before the ingestion.* imports below.
if "fitz" not in sys.modules:
    sys.modules["fitz"] = types.ModuleType("fitz")

from ingestion.chunker import Chunk
from ingestion.extractor import extract_from_chunk, CANONICAL_RELATIONS


def _make_chunk(content: str) -> Chunk:
    return Chunk(
        chunk_id="chunk_test_1",
        doc_id="doc_test",
        doc_title="Test Paper: A Study",
        section_name="Method",
        page_number=1,
        content=content,
    )


def test_regex_fallback_produces_facts_offline():
    """Offline stub yields no extraction JSON, so regex fallback must extract facts."""
    chunk = _make_chunk(
        "Transformer proposes a self-attention mechanism, and the model "
        "outperforms recurrent baselines on translation tasks."
    )

    facts = extract_from_chunk(chunk)

    assert len(facts) > 0
    for fact in facts:
        assert fact.head_entity != ""
        assert fact.relation != ""
        assert fact.tail_entity != ""
        assert fact.chunk_id == "chunk_test_1"
        # Relations are drawn from the canonical vocabulary (stored upper-cased).
        assert fact.relation.lower() in CANONICAL_RELATIONS


def test_deterministic_fact_ids():
    """The offline extraction path is deterministic across repeated calls."""
    chunk = _make_chunk("BERT utilizes bidirectional encoders for representation.")

    first = extract_from_chunk(chunk)
    second = extract_from_chunk(chunk)

    assert [f.fact_id for f in first] == [f.fact_id for f in second]
