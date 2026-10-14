"""
Tests for the Offline Ingestion Pipeline (MemGraphRAG).

Validates:
1. PDF Parser (PyMuPDF layout extraction)
2. Structure-Aware Chunker (section and paragraph preservation)
3. Knowledge Extractor (Ontology Schema & Fact Triples)
4. FAISS and BM25 index creation and queryability
"""

import os
from pathlib import Path
import pytest

from ingestion.parser import parse_pdf, ParsedDocument
from ingestion.chunker import chunk_document, Chunk
from ingestion.extractor import extract_knowledge
from ingestion.cli import run_ingestion_pipeline
from memory.vector_store import VectorStore
from memory.sparse_store import SparseStore

SAMPLE_PDF = Path("RAGxamples/AttentionRAG.pdf")


def test_pdf_parser():
    """Verify PyMuPDF parser extracts title, authors, sections, and pages."""
    assert SAMPLE_PDF.exists(), "Sample PDF not found"
    doc = parse_pdf(SAMPLE_PDF)

    assert isinstance(doc, ParsedDocument)
    assert doc.doc_id == "AttentionRAG"
    assert "AttentionRAG" in doc.title
    assert len(doc.authors) > 0
    assert len(doc.sections) > 1
    assert doc.total_pages > 0


def test_chunker_structure_preservation():
    """Verify chunker respects section boundaries and size limits."""
    doc = parse_pdf(SAMPLE_PDF)
    chunks = chunk_document(doc, target_chunk_size=3500)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.doc_id == "AttentionRAG"
        assert chunk.section_name != ""
        assert chunk.page_number >= 1
        assert len(chunk.content) > 30
        assert f"[{doc.title}" in chunk.content


def test_knowledge_extractor():
    """Verify fact triples and ontology schemas are extracted."""
    doc = parse_pdf(SAMPLE_PDF)
    chunks = chunk_document(doc)
    extraction = extract_knowledge(doc.doc_id, chunks)

    assert extraction.doc_id == "AttentionRAG"
    assert len(extraction.schemas) > 0
    assert len(extraction.facts) > 0
    for fact in extraction.facts:
        assert fact.head_entity != ""
        assert fact.relation != ""
        assert fact.tail_entity != ""
        assert fact.chunk_id != ""


def test_ingestion_pipeline_end_to_end(tmp_path):
    """Verify full ingestion pipeline creates valid FAISS and BM25 index stores."""
    summary = run_ingestion_pipeline(
        input_dir="RAGxamples",
        output_dir=str(tmp_path),
        index_neo4j=False,  # Isolated from external Neo4j during testing
    )

    assert summary["status"] == "success"
    assert summary["documents_ingested"] >= 3
    assert summary["total_chunks"] > 10
    assert summary["total_facts"] > 0

    faiss_file = tmp_path / "faiss_index.bin"
    faiss_meta = tmp_path / "faiss_metadata.json"
    bm25_file = tmp_path / "bm25_index.pkl"
    bm25_meta = tmp_path / "bm25_metadata.json"

    assert faiss_file.exists()
    assert faiss_meta.exists()
    assert bm25_file.exists()
    assert bm25_meta.exists()

    # Verify stores can be reloaded and queried
    v_store = VectorStore(
        index_path=str(faiss_file),
        metadata_path=str(faiss_meta),
    )
    v_store.load()
    dense_results = v_store.search("attention mechanism", top_k=3)
    assert len(dense_results) > 0

    s_store = SparseStore(
        index_path=str(bm25_file),
        metadata_path=str(bm25_meta),
    )
    s_store.load()
    sparse_results = s_store.search("attention mechanism", top_k=3)
    assert len(sparse_results) > 0
