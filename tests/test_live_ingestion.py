"""
Automated Test Suite for Milestone 1: Live In-Browser PDF Ingestion (tests/test_live_ingestion.py)

Validates:
1. Catalog initialization and bootstrapping from existing stores.
2. MD5 deduplication when uploading previously ingested papers.
3. End-to-end PDF layout parsing, chunking, dense FAISS embedding, and BM25 tokenization.
4. Neo4j offline graceful degradation (amber warning flag without unhandled crash).
5. Complete memory wipe routine (clears FAISS, BM25, uploads, and catalog to 0 documents).
6. UI formatters (header meta bar, status alert pills, document catalog cards).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import fitz
import pytest

from ingestion.service import LiveIngestionService
from ui.app import (
    render_header_meta_html,
    render_ingest_status_html,
    render_catalog_html,
)


def _create_sample_pdf(file_path: Path, title: str, content: str) -> Path:
    """Helper to generate a lightweight academic PDF for testing."""
    doc = fitz.open()
    page = doc.new_page()
    full_text = f"{title}\n\nAbstract\n{content}\n\n1 Introduction\nIn this work, {title} proposes a novel methodology for scalable AI retrieval."
    page.insert_text((50, 72), full_text)
    doc.save(str(file_path))
    doc.close()
    return file_path


def test_catalog_initialization(tmp_path: Path):
    """Verify that an empty or missing catalog initializes cleanly."""
    data_dir = tmp_path / "data"
    uploads_dir = tmp_path / "uploads"
    service = LiveIngestionService(data_dir=data_dir, uploads_dir=uploads_dir)

    cat = service.get_catalog()
    assert cat["total_documents"] == 0
    assert cat["total_chunks"] == 0
    assert cat["documents"] == []
    assert (data_dir / "corpus_catalog.json").is_file()


def test_live_ingestion_end_to_end(tmp_path: Path):
    """Verify end-to-end PDF parsing, embedding into FAISS, indexing in BM25, and catalog updating."""
    data_dir = tmp_path / "data"
    uploads_dir = tmp_path / "uploads"
    service = LiveIngestionService(data_dir=data_dir, uploads_dir=uploads_dir)

    # Create three synthetic research papers
    pdf1 = _create_sample_pdf(
        tmp_path / "paper_alpha.pdf",
        "AlphaRAG: Fast Dense Retrieval",
        "AlphaRAG utilizes dense semantic indexing and achieves high precision on multi-hop question answering benchmarks.",
    )
    pdf2 = _create_sample_pdf(
        tmp_path / "paper_beta.pdf",
        "BetaCompressor: Adaptive Token Pruning",
        "BetaCompressor compresses long academic contexts by 4.5x while preserving citation boundaries.",
    )
    pdf3 = _create_sample_pdf(
        tmp_path / "paper_gamma.pdf",
        "GammaAttention: Neural Pruning",
        "GammaAttention isolates anchor tokens for efficient attention calculation.",
    )

    # Ingest all three files
    progress_records = []
    def progress_tracker(ratio: float, desc: str):
        progress_records.append((ratio, desc))

    res = service.ingest_files([pdf1, pdf2, pdf3], progress_cb=progress_tracker)

    # Dense + lexical always succeed; graph layer may be offline in CI, so the
    # honest contract is success (graph synced) OR partial_success (graph offline).
    assert res["status"] in ("success", "partial_success")
    assert (res["status"] == "success") == (res["neo4j_status"] == "synced")
    assert res["ingested_count"] == 3
    assert res["new_chunks"] >= 3
    assert res["total_documents"] == 3
    assert res["total_chunks"] >= 3
    assert len(progress_records) > 0

    # Verify persistent storage
    assert (uploads_dir / "paper_alpha.pdf").is_file()
    assert (uploads_dir / "paper_beta.pdf").is_file()
    assert (data_dir / "faiss_index.bin").is_file()
    assert (data_dir / "faiss_metadata.json").is_file()
    assert (data_dir / "bm25_index.pkl").is_file()
    assert (data_dir / "bm25_metadata.json").is_file()

    # Verify searchability in FAISS
    from memory.vector_store import VectorStore
    v_store = VectorStore(
        index_path=str(data_dir / "faiss_index.bin"),
        metadata_path=str(data_dir / "faiss_metadata.json"),
    )
    assert v_store.load_if_exists() is True
    results = v_store.search("dense semantic indexing", top_k=2)
    assert len(results) > 0
    assert any("AlphaRAG" in r.get("title", "") for r in results)

    # Verify searchability in BM25
    from memory.sparse_store import SparseStore
    s_store = SparseStore(
        index_path=str(data_dir / "bm25_index.pkl"),
        metadata_path=str(data_dir / "bm25_metadata.json"),
    )
    assert s_store.load_if_exists() is True
    bm25_results = s_store.search("BetaCompressor token pruning", top_k=2)
    assert len(bm25_results) > 0


def test_deduplication(tmp_path: Path):
    """Verify that uploading the exact same PDF twice skips re-embedding."""
    data_dir = tmp_path / "data"
    uploads_dir = tmp_path / "uploads"
    service = LiveIngestionService(data_dir=data_dir, uploads_dir=uploads_dir)

    pdf = _create_sample_pdf(
        tmp_path / "paper_gamma.pdf",
        "GammaAttention: Neural Pruning",
        "GammaAttention isolates anchor tokens for efficient attention calculation.",
    )

    # First upload
    res1 = service.ingest_files([pdf])
    assert res1["status"] in ("success", "partial_success")
    assert res1["ingested_count"] == 1
    orig_chunks = res1["total_chunks"]

    # Second upload of the identical file
    res2 = service.ingest_files([pdf])
    assert res2["status"] == "skipped"
    assert res2["ingested_count"] == 0
    assert res2["skipped_duplicates"] == 1
    assert res2["total_chunks"] == orig_chunks


def test_complete_memory_wipe(tmp_path: Path):
    """Verify that wipe_all() completely empties all indexes and catalog to 0 documents."""
    data_dir = tmp_path / "data"
    uploads_dir = tmp_path / "uploads"
    service = LiveIngestionService(data_dir=data_dir, uploads_dir=uploads_dir)

    pdf = _create_sample_pdf(
        tmp_path / "paper_delta.pdf",
        "DeltaGraph: Graph Memory",
        "DeltaGraph constructs hierarchical knowledge graphs for reasoning.",
    )
    service.ingest_files([pdf])
    assert service.get_catalog()["total_documents"] == 1

    # Execute complete wipe
    wipe_res = service.wipe_all()
    assert wipe_res["status"] == "wiped"
    assert wipe_res["total_documents"] == 0
    assert wipe_res["total_chunks"] == 0

    # Verify stores are empty
    cat = service.get_catalog()
    assert cat["total_documents"] == 0
    assert len(cat["documents"]) == 0
    assert len(list(uploads_dir.glob("*.pdf"))) == 0

    # Reload the on-disk indexes and confirm they truly hold zero entries
    # (prove clearance, not just that the return value said 0) — Issue 40.
    from memory.vector_store import VectorStore
    from memory.sparse_store import SparseStore
    v_store = VectorStore(
        index_path=str(data_dir / "faiss_index.bin"),
        metadata_path=str(data_dir / "faiss_metadata.json"),
    )
    v_store.load_if_exists()
    assert v_store.size == 0
    s_store = SparseStore(
        index_path=str(data_dir / "bm25_index.pkl"),
        metadata_path=str(data_dir / "bm25_metadata.json"),
    )
    s_store.load_if_exists()
    assert s_store.size == 0


def test_ui_rendering_helpers():
    """Verify CodeRabbit HTML renderers produce valid markup for diverse catalog states."""
    mock_catalog = {
        "total_documents": 2,
        "total_chunks": 42,
        "documents": [
            {
                "title": "AttentionRAG: Fast Context Pruning",
                "page_count": 9,
                "chunk_count": 24,
                "ingested_at": "2026-10-14T20:00:00Z",
            },
            {
                "title": "MemGraphRAG: Multi-Agent System",
                "page_count": 14,
                "chunk_count": 18,
                "ingested_at": "2026-10-14T20:10:00Z",
            },
        ],
    }

    # Header meta
    header_html = render_header_meta_html(mock_catalog)
    assert "2 Papers Ingested" in header_html
    assert "42 Chunks" in header_html
    assert "cr-meta-bar" in header_html

    # Catalog list
    cat_html = render_catalog_html(mock_catalog)
    assert "AttentionRAG" in cat_html
    assert "24 Chunks" in cat_html
    assert "MemGraphRAG" in cat_html

    # Empty catalog list
    empty_cat_html = render_catalog_html({"documents": []})
    assert "No documents loaded" in empty_cat_html

    # Status pills
    success_status = render_ingest_status_html({
        "status": "success",
        "message": "Ingested 1 document",
        "neo4j_status": "offline",
    })
    assert "✅" in success_status
    assert "Neo4j Offline" in success_status

    wiped_status = render_ingest_status_html({
        "status": "wiped",
        "message": "Memory wiped to clean slate.",
    })
    assert "🗑️" in wiped_status



def test_ingestion_populates_neo4j_graph_layer(tmp_path, monkeypatch):
    """Guard against Issue 37: ingestion must call the Neo4j client with the
    SIGNATURE IT ACTUALLY HAS, and report neo4j_status='synced' when the graph
    layer accepts the writes. A MagicMock stands in for a live Neo4j so this runs
    in CI; crucially it asserts upsert_document is called with the kwargs the
    service passes (url=, authors=, publication_year=) — a mismatch would raise
    TypeError and previously was swallowed as 'offline'.
    """
    import inspect
    from unittest.mock import MagicMock
    import memory.neo4j_client as neo4j_mod
    from memory.neo4j_client import Neo4jClient

    data_dir = tmp_path / "data"
    uploads_dir = tmp_path / "uploads"

    # Validate the real signature accepts the service's kwargs BEFORE mocking it out,
    # so the mock test can never pass against a broken signature.
    inspect.signature(Neo4jClient.upsert_document).bind(
        None, doc_id="d", title="t", url="u", authors=["a"], publication_year=2021
    )

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    monkeypatch.setattr(neo4j_mod, "Neo4jClient", lambda *a, **k: mock_client)

    service = LiveIngestionService(data_dir=data_dir, uploads_dir=uploads_dir)
    pdf = _create_sample_pdf(
        tmp_path / "paper_graph.pdf",
        "GraphTest: Structured Memory",
        "GraphTest proposes a hierarchical knowledge graph for retrieval augmentation.",
    )
    res = service.ingest_files([pdf])

    # Graph layer accepted the writes -> full success, not partial.
    assert res["neo4j_status"] == "synced", res
    assert res["status"] == "success", res
    # upsert_document was actually called (graph layer populated), with valid kwargs.
    assert mock_client.upsert_document.called
    _, kwargs = mock_client.upsert_document.call_args
    assert "doc_id" in kwargs
