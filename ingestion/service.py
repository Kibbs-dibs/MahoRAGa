"""
Live In-Browser Ingestion Service for Project Aletheia (ingestion/service.py)

Orchestrates real-time document ingestion directly into the Three-Layer Global Memory:
1. File validation & MD5 deduplication
2. PyMuPDF Layout Parsing (ingestion.parser)
3. Structure-preserving Chunking (ingestion.chunker)
4. Knowledge Graph Fact & Schema Extraction (ingestion.extractor)
5. Incremental FAISS VectorStore addition & persistence
6. Incremental BM25 SparseStore addition & persistence
7. Idempotent Neo4j Graph Diffusion (with graceful offline fallback)
8. Persistent Corpus Catalog Registry (data/corpus_catalog.json)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import settings
from ingestion.parser import parse_pdf, ParsedDocument
from ingestion.chunker import chunk_document, Chunk
from ingestion.extractor import extract_knowledge, ExtractionResult
from memory.vector_store import VectorStore
from memory.sparse_store import SparseStore

logger = logging.getLogger("ingestion.service")


class LiveIngestionService:
    """Service handling in-browser PDF uploads and real-time corpus management."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        uploads_dir: str | Path = "data/uploads",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.uploads_dir = Path(uploads_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_index_path = str(self.data_dir / "faiss_index.bin")
        self.faiss_meta_path = str(self.data_dir / "faiss_metadata.json")
        self.bm25_index_path = str(self.data_dir / "bm25_index.pkl")
        self.bm25_meta_path = str(self.data_dir / "bm25_metadata.json")
        self.catalog_path = self.data_dir / "corpus_catalog.json"

        # Initialize or bootstrap catalog
        self.ensure_catalog_initialized()

    # ── Catalog Registry Management ─────────────────────────────────────

    def ensure_catalog_initialized(self) -> Dict[str, Any]:
        """Ensure corpus_catalog.json exists; seed from existing indexes if needed."""
        if self.catalog_path.is_file():
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as fh:
                    catalog = json.load(fh)
                if isinstance(catalog, dict) and "documents" in catalog:
                    return catalog
            except Exception as exc:
                logger.warning("Corpus catalog corrupt, rebuilding: %s", exc)

        return self._bootstrap_catalog_from_existing_stores()

    def _bootstrap_catalog_from_existing_stores(self) -> Dict[str, Any]:
        """Bootstrap catalog entries from existing FAISS/BM25 metadata sidecars."""
        catalog: Dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_documents": 0,
            "total_chunks": 0,
            "documents": [],
        }

        # Inspect existing FAISS metadata
        if os.path.isfile(self.faiss_meta_path):
            try:
                with open(self.faiss_meta_path, "r", encoding="utf-8") as fh:
                    faiss_meta = json.load(fh)

                # Group chunks by document_id
                doc_groups: Dict[str, List[Dict[str, Any]]] = {}
                for item in faiss_meta:
                    doc_id = item.get("document_id") or "unknown_doc"
                    doc_groups.setdefault(doc_id, []).append(item)

                for doc_id, items in doc_groups.items():
                    title = items[0].get("title") or doc_id
                    max_page = max((it.get("page_number") or 1) for it in items)
                    catalog["documents"].append({
                        "doc_id": doc_id,
                        "filename": f"{doc_id}.pdf",
                        "title": title,
                        "authors": ["Author(s) Ingested"],
                        "page_count": max_page,
                        "chunk_count": len(items),
                        "md5_hash": f"seed_{doc_id}",
                        "ingested_at": catalog["updated_at"],
                        "storage_path": f"RAGxamples/{doc_id}.pdf",
                    })

                catalog["total_documents"] = len(catalog["documents"])
                catalog["total_chunks"] = len(faiss_meta)
            except Exception as exc:
                logger.warning("Failed to bootstrap catalog from FAISS metadata: %s", exc)

        self._save_catalog(catalog)
        return catalog

    def _save_catalog(self, catalog: Dict[str, Any]) -> None:
        """Persist catalog to disk."""
        catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.catalog_path, "w", encoding="utf-8") as fh:
            json.dump(catalog, fh, indent=2, ensure_ascii=False)

    def get_catalog(self) -> Dict[str, Any]:
        """Retrieve the current corpus catalog."""
        return self.ensure_catalog_initialized()

    @staticmethod
    def compute_file_md5(file_path: str | Path) -> str:
        """Calculate hexadecimal MD5 hash for deduplication."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # ── Ingestion Pipeline ──────────────────────────────────────────────

    def ingest_files(
        self,
        file_paths: List[str | Path],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a list of PDF file paths directly into the Three-Layer Global Memory.

        Args:
            file_paths: List of uploaded PDF paths (temporary or permanent).
            progress_cb: Optional callback receiving (progress_ratio: float, status: str).

        Returns:
            Structured summary dictionary.
        """
        def update_progress(ratio: float, msg: str):
            if progress_cb:
                try:
                    progress_cb(ratio, msg)
                except Exception as exc:
                    logger.debug("Progress callback error: %s", exc)

        catalog = self.ensure_catalog_initialized()
        known_hashes = {doc.get("md5_hash") for doc in catalog["documents"]}
        known_ids = {doc.get("doc_id") for doc in catalog["documents"]}

        valid_files: List[Tuple[Path, str]] = []
        skipped_duplicates: List[str] = []

        update_progress(0.05, "Validating uploaded files...")

        # 1. Validation & Deduplication
        for raw_p in file_paths:
            path = Path(raw_p)
            if not path.is_file() or path.suffix.lower() != ".pdf":
                logger.warning("Skipping invalid or non-PDF file: %s", path)
                continue

            file_hash = self.compute_file_md5(path)
            if file_hash in known_hashes:
                logger.info("Skipping duplicate file (MD5 match): %s", path.name)
                skipped_duplicates.append(path.name)
                continue

            valid_files.append((path, file_hash))

        if not valid_files:
            if skipped_duplicates:
                msg = f"All {len(skipped_duplicates)} uploaded paper(s) are already ingested in the corpus."
                return {
                    "status": "skipped",
                    "ingested_count": 0,
                    "skipped_duplicates": len(skipped_duplicates),
                    "new_chunks": 0,
                    "total_documents": catalog["total_documents"],
                    "total_chunks": catalog["total_chunks"],
                    "neo4j_status": "skipped",
                    "message": msg,
                    "catalog": catalog,
                }
            return {
                "status": "no_files",
                "ingested_count": 0,
                "skipped_duplicates": 0,
                "new_chunks": 0,
                "total_documents": catalog["total_documents"],
                "total_chunks": catalog["total_chunks"],
                "neo4j_status": "skipped",
                "message": "No valid PDF documents were selected.",
                "catalog": catalog,
            }

        # 2. Parse, Chunk, and Extract Triples
        ingested_docs: List[ParsedDocument] = []
        new_chunks: List[Chunk] = []
        new_extractions: List[ExtractionResult] = []
        processed_records: List[Dict[str, Any]] = []

        total_files = len(valid_files)
        for idx, (pdf_path, md5_hash) in enumerate(valid_files, 1):
            file_ratio_start = 0.1 + (idx - 1) / total_files * 0.4
            file_ratio_mid = file_ratio_start + 0.2 / total_files
            update_progress(file_ratio_start, f"[{idx}/{total_files}] Parsing {pdf_path.name}...")

            # Copy to persistent data/uploads/
            safe_name = pdf_path.name
            dest_path = self.uploads_dir / safe_name
            # If a file with the same name exists, prepend timestamp
            if dest_path.exists() and dest_path != pdf_path:
                dest_path = self.uploads_dir / f"{int(datetime.now().timestamp())}_{safe_name}"
            shutil.copy2(pdf_path, dest_path)

            try:
                # Layout Parsing
                doc = parse_pdf(dest_path)
                # Ensure unique doc_id if collides
                if doc.doc_id in known_ids:
                    doc.doc_id = f"{doc.doc_id}_{md5_hash[:6]}"

                update_progress(file_ratio_mid, f"[{idx}/{total_files}] Chunking & extracting facts...")
                chunks = chunk_document(doc)
                extractions = extract_knowledge(doc.doc_id, chunks)

                ingested_docs.append(doc)
                new_chunks.extend(chunks)
                new_extractions.append(extractions)

                record = {
                    "doc_id": doc.doc_id,
                    "filename": dest_path.name,
                    "title": doc.title or dest_path.stem,
                    "authors": doc.authors or ["Author(s) Unknown"],
                    "page_count": getattr(doc, "total_pages", getattr(doc, "page_count", 1)),
                    "chunk_count": len(chunks),
                    "md5_hash": md5_hash,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "storage_path": str(dest_path),
                }
                processed_records.append(record)
                known_hashes.add(md5_hash)
                known_ids.add(doc.doc_id)

            except Exception as exc:
                logger.error("Failed to parse %s: %s", pdf_path.name, exc, exc_info=True)

        if not new_chunks:
            return {
                "status": "error",
                "ingested_count": 0,
                "skipped_duplicates": len(skipped_duplicates),
                "new_chunks": 0,
                "total_documents": catalog["total_documents"],
                "total_chunks": catalog["total_chunks"],
                "neo4j_status": "skipped",
                "message": "Failed to extract readable text passages from uploaded PDFs.",
                "catalog": catalog,
            }

        # Prepare payloads for stores
        chunk_texts = [c.content for c in new_chunks]
        chunk_metadatas = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.doc_id,
                "title": c.doc_title,
                "section": c.section_name,
                "page_number": c.page_number,
                "text": c.content,
            }
            for c in new_chunks
        ]

        # 3. Incremental Index into FAISS Dense Vector Store
        update_progress(0.60, f"Embedding {len(chunk_texts)} chunks into FAISS...")
        v_store = VectorStore(
            index_path=self.faiss_index_path,
            metadata_path=self.faiss_meta_path,
        )
        v_store.load_if_exists()
        v_store.add_chunks(chunk_texts, chunk_metadatas)
        v_store.save()

        # 4. Incremental Index into BM25 Sparse Lexical Store
        update_progress(0.75, f"Tokenizing & updating BM25 lexical index...")
        s_store = SparseStore(
            index_path=self.bm25_index_path,
            metadata_path=self.bm25_meta_path,
        )
        s_store.load_if_exists()
        s_store.add_documents(chunk_texts, chunk_metadatas)
        s_store.save()

        # 5. Idempotent Index into Neo4j Hierarchical Graph
        update_progress(0.90, "Syncing Neo4j knowledge graph...")
        neo4j_status = "synced"
        try:
            from memory.neo4j_client import Neo4jClient
            with Neo4jClient() as client:
                client.ensure_indexes()

                for doc in ingested_docs:
                    client.upsert_document(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        url=doc.file_path,
                        authors=doc.authors,
                        publication_year=doc.year,
                    )

                for c in new_chunks:
                    client.upsert_passage_chunk(
                        chunk_id=c.chunk_id,
                        document_id=c.doc_id,
                        section_label=c.section_name,
                        content=c.content,
                        page_number=c.page_number,
                    )

                for extraction in new_extractions:
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
        except Exception as exc:
            logger.warning("Neo4j offline or unreachable during live ingest: %s", exc)
            neo4j_status = "offline"

        # 6. Update & Persist Catalog
        update_progress(0.98, "Updating corpus catalog...")
        catalog["documents"].extend(processed_records)
        catalog["total_documents"] = len(catalog["documents"])
        catalog["total_chunks"] = len(catalog["documents"]) and sum(d.get("chunk_count", 0) for d in catalog["documents"])
        self._save_catalog(catalog)

        update_progress(1.0, "Ingestion complete!")

        summary_msg = (
            f"Successfully ingested {len(ingested_docs)} document(s) "
            f"(+{len(new_chunks)} chunks). Total corpus: {catalog['total_documents']} papers ({catalog['total_chunks']} chunks)."
        )
        if skipped_duplicates:
            summary_msg += f" Skipped {len(skipped_duplicates)} duplicate file(s)."

        return {
            "status": "success",
            "ingested_count": len(ingested_docs),
            "skipped_duplicates": len(skipped_duplicates),
            "new_chunks": len(new_chunks),
            "total_documents": catalog["total_documents"],
            "total_chunks": catalog["total_chunks"],
            "neo4j_status": neo4j_status,
            "message": summary_msg,
            "catalog": catalog,
        }

    # ── Memory Wipe ─────────────────────────────────────────────────────

    def wipe_all(self) -> Dict[str, Any]:
        """
        Complete wipe: Clears all FAISS, BM25, uploads, and Neo4j memory completely (0 documents).
        """
        logger.info("Executing complete memory wipe...")

        # 1. Reset FAISS index to 0 vectors
        v_store = VectorStore(
            index_path=self.faiss_index_path,
            metadata_path=self.faiss_meta_path,
        )
        v_store.save()  # Saves empty index

        # 2. Reset BM25 index to 0 documents
        s_store = SparseStore(
            index_path=self.bm25_index_path,
            metadata_path=self.bm25_meta_path,
        )
        s_store.save()  # Saves empty index

        # 3. Clean uploads directory
        if self.uploads_dir.is_dir():
            for f in self.uploads_dir.glob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                except Exception as exc:
                    logger.warning("Failed to delete upload file %s: %s", f, exc)

        # 4. Reset Neo4j if reachable
        neo4j_status = "purged"
        try:
            from memory.neo4j_client import Neo4jClient
            with Neo4jClient() as client:
                client.execute_query("MATCH (n) DETACH DELETE n")
        except Exception as exc:
            neo4j_status = f"skipped ({exc})"

        # 5. Reset Catalog
        empty_catalog = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_documents": 0,
            "total_chunks": 0,
            "documents": [],
        }
        self._save_catalog(empty_catalog)

        return {
            "status": "wiped",
            "total_documents": 0,
            "total_chunks": 0,
            "neo4j_status": neo4j_status,
            "message": "All memory stores (FAISS, BM25, uploads, catalog) have been wiped to a clean slate (0 documents).",
            "catalog": empty_catalog,
        }


# Singleton service instance
ingestion_service = LiveIngestionService()
