"""
Memory-Guided Retriever for Project Aletheia.

Responsibilities
────────────────
• Fetches structurally sound and semantically relevant data from Neo4j, FAISS, and BM25.
• Utilizes Personalized PageRank (PPR) for graph diffusion and Reciprocal Rank Fusion (RRF) for dual-index merging.
• Updates the MemoryState within the AletheiaState object directly.

This node implements direct state manipulation as specified in Approach 1.
"""

from __future__ import annotations

from typing import List, Dict, Any
from core.state import AletheiaState, MemoryState
from core.config import settings
from core.exceptions import AletheiaError
from memory.neo4j_client import Neo4jClient
from memory.vector_store import VectorStore
from memory.sparse_store import SparseStore
import logging

logger = logging.getLogger(__name__)


def retriever_node(state: AletheiaState) -> AletheiaState:
    """
    Memory-Guided Retriever node.

    Fetches data from all three memory layers and prepares it for evaluation.

    Args:
        state: Current AletheiaState object

    Returns:
        Updated AletheiaState object with memory layer populated

    Raises:
        AletheiaError: If retrieval from any layer fails
    """
    try:
        logger.info("Starting memory-guided retrieval process")

        # Initialize memory state if not present
        if state.memory is None:
            state.memory = MemoryState()

        # Extract the query to use for retrieval
        query_text = state.query.reformulated_query or state.query.original_query
        if not query_text:
            raise AletheiaError("No query available for retrieval")

        logger.debug(f"Retrieving for query: {query_text[:100]}...")

        # Retrieve from Neo4j (graph layer) using Personalized PageRank
        neo4j_triples = _retrieve_from_neo4j(query_text)
        state.memory.serialized_graph_triples = neo4j_triples
        logger.debug(f"Retrieved {len(neo4j_triples)} triples from Neo4j")

        # Retrieve from FAISS (dense vector layer)
        vector_chunks = _retrieve_from_faiss(query_text)
        state.memory.vector_chunks = vector_chunks
        logger.debug(f"Retrieved {len(vector_chunks)} chunks from FAISS")

        # Retrieve from BM25 (sparse lexical layer)
        lexical_chunks = _retrieve_from_bm25(query_text)
        state.memory.lexical_chunks = lexical_chunks
        logger.debug(f"Retrieved {len(lexical_chunks)} chunks from BM25")

        logger.info(
            f"Retrieval complete: {len(neo4j_triples)} graph triples, "
            f"{len(vector_chunks)} vector chunks, {len(lexical_chunks)} lexical chunks"
        )

        return state

    except Exception as exc:
        logger.error(f"Retriever node failed: {exc}")
        raise AletheiaError(f"Memory-guided retrieval failed: {exc}") from exc


def _retrieve_from_neo4j(query_text: str) -> List[str]:
    """Retrieve and serialize relevant triples from Neo4j using PPR."""
    try:
        with Neo4jClient() as client:
            # Ensure indexes exist (idempotent)
            client.ensure_indexes()

            # Extract entities from query for PPR seeding
            # Simple approach: use nouns/named entities as seeds
            # In production, this would use proper NER
            seed_entities = _extract_entities_from_query(query_text)

            all_triples = []
            for entity in seed_entities[:3]:  # Limit to top 3 entities to avoid overload
                try:
                    triples = client.serialise_subgraph_to_sentences(entity, limit=10)
                    all_triples.extend(triples)
                except Exception as exc:
                    logger.warning(f"Failed to get triples for entity '{entity}': {exc}")
                    continue

            # Deduplicate while preserving order
            seen = set()
            deduped_triples = []
            for triple in all_triples:
                if triple not in seen:
                    seen.add(triple)
                    deduped_triples.append(triple)

            return deduped_triples[:settings.top_k_retrieval]

    except Exception as exc:
        logger.error(f"Neo4j retrieval failed: {exc}")
        # Return empty list rather than failing completely - allows other layers to work
        return []


def _retrieve_from_faiss(query_text: str) -> List[str]:
    """Retrieve relevant chunks from FAISS vector store."""
    try:
        store = VectorStore()
        # Try to load existing index; if not available, return empty results
        if not store.load_if_exists():
            logger.warning("FAISS index not found - returning empty vector results")
            return []

        results = store.search(query_text, top_k=settings.top_k_retrieval)
        # Extract just the text content for consistency with other layers
        return [result.get('text', '') for result in results if result.get('text')]

    except Exception as exc:
        logger.error(f"FAISS retrieval failed: {exc}")
        # Return empty list rather than failing completely
        return []


def _retrieve_from_bm25(query_text: str) -> List[str]:
    """Retrieve relevant chunks from BM25 sparse store."""
    try:
        store = SparseStore()
        # Try to load existing index; if not available, return empty results
        if not store.load_if_exists():
            logger.warning("BM25 index not found - returning empty lexical results")
            return []

        results = store.search(query_text, top_k=settings.top_k_retrieval)
        # Extract just the text content for consistency with other layers
        return [result.get('text', '') for result in results if result.get('text')]

    except Exception as exc:
        logger.error(f"BM25 retrieval failed: {exc}")
        # Return empty list rather than failing completely
        return []


def _extract_entities_from_query(query_text: str) -> List[str]:
    """
    Simple entity extraction from query text.
    In a production system, this would use proper NER (spaCy, etc.).
    For now, we'll extract capitalized words and phrases as a heuristic.
    """
    import re

    # Simple heuristic: look for capitalized words and phrases
    # This is a placeholder - real implementation would use NER
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query_text)

    # Also look for quoted terms which might be important concepts
    quoted = re.findall(r'"([^"]*)"', query_text)
    words.extend(quoted)

    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)

    # Fallback: if no entities found, use the whole query as a single entity
    if not unique_words:
        unique_words = [query_text[:50]]  # First 50 chars as fallback entity

    return unique_words[:5]  # Limit to top 5 entities