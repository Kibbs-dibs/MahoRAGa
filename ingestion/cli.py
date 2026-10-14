"""
Ingestion CLI & Pipeline Runner for Project Aletheia.

Indexes academic literature from PDF files into the Three-Layer Global Memory:
1. FAISS Dense Vector Store (384-dim IndexFlatIP)
2. BM25 Sparse Lexical Store (BM25Okapi + Cross-Encoder sidecar)
3. Neo4j Hierarchical Indexing Graph (Ontology, Fact, Passage, Document)

Usage:
    python -m ingestion.cli --input RAGxamples/ --output data/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from core.config import settings
from ingestion.parser import parse_pdf, ParsedDocument
from ingestion.chunker import chunk_document, Chunk
from ingestion.extractor import extract_knowledge, ExtractionResult
from memory.vector_store import VectorStore
from memory.sparse_store import SparseStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion.cli")


def run_ingestion_pipeline(
    input_dir: str | Path = "RAGxamples",
    output_dir: str | Path = "data",
    index_neo4j: bool = True,
) -> Dict[str, Any]:
    """
    Execute the offline ingestion pipeline across all PDFs in input_dir.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Destination directory for FAISS and BM25 indexes
        index_neo4j: Whether to attempt Neo4j indexing

    Returns:
        Summary dictionary with ingestion statistics
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    faiss_index_file = str(out_path / "faiss_index.bin")
    faiss_meta_file = str(out_path / "faiss_metadata.json")
    bm25_index_file = str(out_path / "bm25_index.pkl")
    bm25_meta_file = str(out_path / "bm25_metadata.json")

    pdf_files = list(in_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", in_path)
        return {"status": "no_files_found", "documents": 0, "chunks": 0}

    logger.info("Found %d PDF documents to ingest in %s", len(pdf_files), in_path)

    all_docs: List[ParsedDocument] = []
    all_chunks: List[Chunk] = []
    all_extractions: List[ExtractionResult] = []

    # ── 1. Parse, Chunk, and Extract ────────────────────────────────────
    for pdf_path in pdf_files:
        try:
            logger.info("── Processing %s ──", pdf_path.name)
            doc = parse_pdf(pdf_path)
            chunks = chunk_document(doc)
            extraction = extract_knowledge(doc.doc_id, chunks)

            all_docs.append(doc)
            all_chunks.extend(chunks)
            all_extractions.append(extraction)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_path.name, exc)

    logger.info(
        "Extraction complete: %d documents parsed into %d passage chunks",
        len(all_docs),
        len(all_chunks),
    )

    if not all_chunks:
        raise RuntimeError("No passage chunks were successfully generated.")

    # Prepare texts and metadata for stores
    chunk_texts = [c.content for c in all_chunks]
    chunk_metadatas = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.doc_id,
            "title": c.doc_title,
            "section": c.section_name,
            "page_number": c.page_number,
            "text": c.content,
        }
        for c in all_chunks
    ]

    # ── 2. Index into FAISS Dense Vector Store ──────────────────────────
    logger.info("Indexing %d chunks into FAISS...", len(chunk_texts))
    v_store = VectorStore(
        index_path=faiss_index_file,
        metadata_path=faiss_meta_file,
    )
    v_store.add_chunks(chunk_texts, chunk_metadatas)
    v_store.save()
    logger.info("FAISS vector store saved to %s", faiss_index_file)

    # ── 3. Index into BM25 Sparse Lexical Store ─────────────────────────
    logger.info("Indexing %d chunks into BM25...", len(chunk_texts))
    s_store = SparseStore(
        index_path=bm25_index_file,
        metadata_path=bm25_meta_file,
    )
    s_store.add_documents(chunk_texts, chunk_metadatas)
    s_store.save()
    logger.info("BM25 sparse store saved to %s", bm25_index_file)

    # ── 4. Index into Neo4j Hierarchical Graph (if requested) ───────────
    neo4j_status = "skipped"
    if index_neo4j:
        try:
            from memory.neo4j_client import Neo4jClient
            client = Neo4jClient()
            client.connect()
            client.ensure_indexes()

            for doc in all_docs:
                client.upsert_document(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    url=doc.file_path,
                    authors=doc.authors,
                    publication_year=doc.year,
                )

            for c in all_chunks:
                client.upsert_passage_chunk(
                    chunk_id=c.chunk_id,
                    document_id=c.doc_id,
                    section_label=c.section_name,
                    content=c.content,
                    page_number=c.page_number,
                )

            for extraction in all_extractions:
                for schema in extraction.schemas:
                    client.upsert_ontology_schema(
                        schema_id=schema.schema_id,
                        head_type=schema.head_type,
                        relation=schema.relation,
                        tail_type=schema.tail_type,
                        status=schema.status,
                    )
                for fact in extraction.facts:
                    client.upsert_fact_triple(
                        fact_id=fact.fact_id,
                        head_entity=fact.head_entity,
                        relation=fact.relation,
                        tail_entity=fact.tail_entity,
                        similarity_weight=fact.similarity_weight,
                        schema_id=fact.schema_id,
                    )
                    if fact.chunk_id:
                        client.link_fact_to_passage(fact.fact_id, fact.chunk_id)

            client.close()
            neo4j_status = "populated"
            logger.info("Neo4j knowledge graph populated successfully.")
        except Exception as exc:
            neo4j_status = f"offline ({exc})"
            logger.warning(
                "Neo4j is offline or unavailable (%s). "
                "FAISS and BM25 stores were built successfully.",
                exc,
            )

    summary = {
        "status": "success",
        "documents_ingested": len(all_docs),
        "total_chunks": len(all_chunks),
        "total_facts": sum(len(e.facts) for e in all_extractions),
        "total_schemas": sum(len(e.schemas) for e in all_extractions),
        "faiss_index_path": faiss_index_file,
        "bm25_index_path": bm25_index_file,
        "neo4j_status": neo4j_status,
    }

    logger.info("════════════════ Ingestion Summary ════════════════")
    logger.info("Documents Ingested: %d", summary["documents_ingested"])
    logger.info("Passage Chunks:     %d", summary["total_chunks"])
    logger.info("Facts Extracted:    %d", summary["total_facts"])
    logger.info("FAISS File:         %s", summary["faiss_index_path"])
    logger.info("BM25 File:          %s", summary["bm25_index_path"])
    logger.info("Neo4j Status:       %s", summary["neo4j_status"])
    logger.info("═══════════════════════════════════════════════════")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Project Aletheia Offline Ingestion CLI")
    parser.add_argument(
        "--input",
        type=str,
        default="RAGxamples",
        help="Path to directory containing academic PDFs (default: RAGxamples)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Path to directory for FAISS and BM25 output (default: data)",
    )
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Skip Neo4j graph indexing",
    )
    args = parser.parse_args()

    run_ingestion_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        index_neo4j=not args.no_neo4j,
    )


if __name__ == "__main__":
    main()
