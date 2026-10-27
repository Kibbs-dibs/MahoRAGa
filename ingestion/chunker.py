"""
Structure-Aware Chunker for Project Aletheia.

Enforces section-boundary preservation and paragraph coherence:
- Target chunk size: ~3,500 characters (settings.chunk_size)
- Never merges content across distinct academic sections
- Attaches provenance metadata headers to every chunk
"""

from __future__ import annotations

import logging
import re
from typing import List
from pydantic import BaseModel, Field

from core.config import settings
from ingestion.parser import ParsedDocument, ParsedSection

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """Represents a coherent passage chunk ready for indexing in FAISS, BM25, and Neo4j."""
    chunk_id: str
    doc_id: str
    doc_title: str
    section_name: str
    page_number: int
    content: str
    character_count: int = 0
    token_estimate: int = 0


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on double newlines or substantial line breaks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


def _split_into_sentences(text: str) -> List[str]:
    """Split large paragraphs into sentences on punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc: ParsedDocument,
    target_chunk_size: int | None = None,
) -> List[Chunk]:
    """
    Chunk an academic document while strictly respecting section boundaries.

    Args:
        doc: ParsedDocument from ingestion.parser
        target_chunk_size: Maximum character target for each chunk (default: settings.chunk_size ~3500)

    Returns:
        List of structured Chunk objects
    """
    max_size = target_chunk_size or settings.chunk_size
    chunks: List[Chunk] = []
    chunk_counter = 1

    for section in doc.sections:
        # Ignore empty sections or pure headers with no body
        if not section.content or len(section.content.strip()) < 30:
            continue

        paragraphs = _split_into_paragraphs(section.content)
        current_paras: List[str] = []
        current_len = 0

        for para in paragraphs:
            # If paragraph itself is larger than max_size, split by sentences
            if len(para) > max_size:
                # Flush existing buffer first
                if current_paras:
                    chunk_text = "\n\n".join(current_paras)
                    header = f"[{doc.title} | {section.section_name} | Page {section.page_number}]\n"
                    full_content = f"{header}{chunk_text}"
                    chunk_id = f"{doc.doc_id}_c{chunk_counter}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc.doc_id,
                            doc_title=doc.title,
                            section_name=section.section_name,
                            page_number=section.page_number,
                            content=full_content,
                            character_count=len(full_content),
                            token_estimate=len(full_content) // 4,
                        )
                    )
                    chunk_counter += 1
                    current_paras = []
                    current_len = 0

                # Process oversized paragraph by sentences
                sentences = _split_into_sentences(para)
                # A single sentence can still exceed max_size (long enumerations,
                # tables, missing punctuation). Hard-split those into <=max_size
                # slices so no emitted unit overflows the budget (Issue 30).
                normalized_sentences: List[str] = []
                for s in sentences:
                    if len(s) > max_size:
                        for i in range(0, len(s), max_size):
                            normalized_sentences.append(s[i:i + max_size])
                    else:
                        normalized_sentences.append(s)
                sentences = normalized_sentences
                sent_buffer: List[str] = []
                sent_len = 0

                for sent in sentences:
                    if sent_len + len(sent) > max_size and sent_buffer:
                        chunk_text = " ".join(sent_buffer)
                        header = f"[{doc.title} | {section.section_name} | Page {section.page_number}]\n"
                        full_content = f"{header}{chunk_text}"
                        chunk_id = f"{doc.doc_id}_c{chunk_counter}"
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                doc_id=doc.doc_id,
                                doc_title=doc.title,
                                section_name=section.section_name,
                                page_number=section.page_number,
                                content=full_content,
                                character_count=len(full_content),
                                token_estimate=len(full_content) // 4,
                            )
                        )
                        chunk_counter += 1
                        sent_buffer = [sent]
                        sent_len = len(sent)
                    else:
                        sent_buffer.append(sent)
                        sent_len += len(sent)

                if sent_buffer:
                    current_paras = [" ".join(sent_buffer)]
                    current_len = sent_len

            elif current_len + len(para) > max_size and current_paras:
                # Flush current buffer
                chunk_text = "\n\n".join(current_paras)
                header = f"[{doc.title} | {section.section_name} | Page {section.page_number}]\n"
                full_content = f"{header}{chunk_text}"
                chunk_id = f"{doc.doc_id}_c{chunk_counter}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        section_name=section.section_name,
                        page_number=section.page_number,
                        content=full_content,
                        character_count=len(full_content),
                        token_estimate=len(full_content) // 4,
                    )
                )
                chunk_counter += 1
                current_paras = [para]
                current_len = len(para)
            else:
                current_paras.append(para)
                current_len += len(para)

        # Flush trailing section content
        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            header = f"[{doc.title} | {section.section_name} | Page {section.page_number}]\n"
            full_content = f"{header}{chunk_text}"
            chunk_id = f"{doc.doc_id}_c{chunk_counter}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section_name=section.section_name,
                    page_number=section.page_number,
                    content=full_content,
                    character_count=len(full_content),
                    token_estimate=len(full_content) // 4,
                )
            )
            chunk_counter += 1

    logger.info(
        "Chunked document %s into %d passage chunks (target: %d chars)",
        doc.doc_id,
        len(chunks),
        max_size,
    )
    return chunks
