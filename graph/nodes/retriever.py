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
from core.exceptions import AletheiaError, SparseStoreError
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

        # Retrieve from Neo4j (graph layer) using Personalized PageRank.
        # We keep the full result dicts so RRF can fuse across layers, and
        # derive the text-only list consumed by the per-layer MemoryState field.
        neo4j_dicts = _retrieve_from_neo4j(query_text)
        neo4j_triples = [d.get("text", "") for d in neo4j_dicts if d.get("text")]
        state.memory.serialized_graph_triples = neo4j_triples
        logger.debug(f"Retrieved {len(neo4j_triples)} triples from Neo4j")

        # Retrieve from FAISS (dense vector layer)
        faiss_dicts = _retrieve_from_faiss(query_text)
        vector_chunks = [d.get("text", "") for d in faiss_dicts if d.get("text")]
        state.memory.vector_chunks = vector_chunks
        logger.debug(f"Retrieved {len(vector_chunks)} chunks from FAISS")

        # Retrieve from BM25 (sparse lexical layer)
        bm25_dicts = _retrieve_from_bm25(query_text)
        lexical_chunks = [d.get("text", "") for d in bm25_dicts if d.get("text")]
        state.memory.lexical_chunks = lexical_chunks
        logger.debug(f"Retrieved {len(lexical_chunks)} chunks from BM25")

        # ── Hybrid fusion + reranking across the dense/sparse/graph layers ──
        state.memory.fused_chunks = _fuse_and_rerank(
            query_text, faiss_dicts, bm25_dicts, neo4j_dicts
        )
        logger.debug(f"Fused/reranked to {len(state.memory.fused_chunks)} chunks")

        logger.info(
            f"Retrieval complete: {len(neo4j_triples)} graph triples, "
            f"{len(vector_chunks)} vector chunks, {len(lexical_chunks)} lexical chunks, "
            f"{len(state.memory.fused_chunks)} fused chunks"
        )

        return state

    except Exception as exc:
        logger.error(f"Retriever node failed: {exc}")
        raise AletheiaError(f"Memory-guided retrieval failed: {exc}") from exc


def _fuse_and_rerank(
    query_text: str,
    faiss_dicts: List[Dict[str, Any]],
    bm25_dicts: List[Dict[str, Any]],
    graph_dicts: List[Dict[str, Any]],
) -> List[str]:
    """Merge the three retrieval layers via Reciprocal Rank Fusion (RRF) and,
    when the cross-encoder is available, apply a final precision rerank.

    The RRF and rerank primitives live in ``memory.sparse_store.SparseStore``;
    this is their production call site. Each input is a list of result dicts
    keyed by ``text`` (graph triples are already wrapped as such).

    Graceful degradation: the cross-encoder model (``sentence_transformers``)
    may be unavailable offline. If ``rerank`` raises ``SparseStoreError``
    (model download/load failure) we fall back to the RRF ordering rather
    than crashing the node.
    """
    fused = SparseStore.reciprocal_rank_fusion(
        faiss_dicts, bm25_dicts, graph_dicts
    )
    if not fused:
        return []

    try:
        reranked = SparseStore().rerank(
            query_text, fused, top_k=settings.top_k_retrieval
        )
        ordered = reranked
    except SparseStoreError as exc:
        # Cross-encoder model unavailable (e.g. offline). Degrade to RRF order.
        logger.warning(
            "Cross-encoder rerank unavailable, falling back to RRF order: %s",
            exc,
        )
        ordered = fused[: settings.top_k_retrieval]

    return [d.get("text", "") for d in ordered if d.get("text")]


def _retrieve_from_neo4j(query_text: str) -> List[Dict[str, Any]]:
    """Retrieve and serialize relevant triples from Neo4j using PPR.

    Returns a list of result dicts keyed by ``text`` (the serialized
    sentence) so the fusion layer can treat graph evidence uniformly with
    the dense/sparse layers.
    """
    try:
        with Neo4jClient() as client:
            # Ensure indexes exist (idempotent)
            client.ensure_indexes()

            # Extract seed entities from the query for PPR seeding. This is a
            # lightweight heuristic NER, not a production NER model.
            seed_entities = _extract_entities_from_query(query_text)

            all_triples: List[str] = []
            for entity in seed_entities[:3]:  # Limit to top 3 entities to avoid overload
                try:
                    # Personalized PageRank (PPR) seeding: rank the neighbourhood
                    # of the seed entity, then serialise the top-scored
                    # FACT_TRIPLE nodes into sentences. This makes the module's
                    # "Utilizes Personalized PageRank (PPR)" claim accurate.
                    # If PPR yields nothing (empty graph / no GDS) or errors, we
                    # gracefully fall back to plain subgraph serialisation.
                    triples = _ppr_triples_for_entity(client, entity)
                    if not triples:
                        triples = client.serialise_subgraph_to_sentences(
                            entity, limit=10
                        )
                    all_triples.extend(triples)
                except Exception as exc:
                    logger.warning(
                        f"Failed to get triples for entity '{entity}': {exc}"
                    )
                    continue

            # Deduplicate while preserving order
            seen = set()
            deduped_triples = []
            for triple in all_triples:
                if triple not in seen:
                    seen.add(triple)
                    deduped_triples.append(triple)

            deduped_triples = deduped_triples[: settings.top_k_retrieval]
            return [{"text": t} for t in deduped_triples]

    except Exception as exc:
        # Degrade gracefully when Neo4j is genuinely offline/unreachable so the
        # other memory layers still work, but log the full traceback so hidden
        # programming errors (TypeError/AttributeError) are not invisible.
        logger.error(f"Neo4j retrieval failed: {exc}", exc_info=True)
        return []


def _ppr_triples_for_entity(client: Neo4jClient, entity: str) -> List[str]:
    """Run Personalized PageRank for *entity* and serialise the top-ranked
    FACT_TRIPLE nodes into natural-language sentences.

    Returns an empty list if PPR returns no FACT_TRIPLE nodes, letting the
    caller fall back to plain subgraph serialisation.
    """
    ppr_results = client.run_personalised_pagerank(entity)

    sentences: List[str] = []
    for record in ppr_results:
        labels = record.get("labels") or []
        if "FACT_TRIPLE" not in labels:
            continue
        props = record.get("properties") or {}
        head = props.get("head_entity", "?")
        rel = props.get("relation", "relates to")
        tail = props.get("tail_entity", "?")
        weight = props.get("similarity_weight", 0.0) or 0.0
        try:
            weight_str = f"{float(weight):.2f}"
        except (TypeError, ValueError):
            weight_str = "0.00"
        sentences.append(f"{head} {rel} {tail} (weight={weight_str}).")

    return sentences


def _retrieve_from_faiss(query_text: str) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from FAISS vector store.

    Returns the full result dicts (containing ``text``/``score``/``rank``)
    so the fusion layer can operate on them; the caller derives the
    text-only list for the per-layer MemoryState field.
    """
    try:
        store = VectorStore()
        # Try to load existing index; if not available, return empty results
        if not store.load_if_exists():
            logger.warning("FAISS index not found - returning empty vector results")
            return []

        results = store.search(query_text, top_k=settings.top_k_retrieval)
        return [r for r in results if r.get("text")]

    except Exception as exc:
        # Degrade gracefully for a genuinely-missing/unreadable index, but log
        # the full traceback so hidden programming errors surface.
        logger.error(f"FAISS retrieval failed: {exc}", exc_info=True)
        return []


def _retrieve_from_bm25(query_text: str) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from BM25 sparse store.

    Returns the full result dicts (containing ``text``/``score``/``rank``)
    so the fusion layer can operate on them; the caller derives the
    text-only list for the per-layer MemoryState field.
    """
    try:
        store = SparseStore()
        # Try to load existing index; if not available, return empty results
        if not store.load_if_exists():
            logger.warning("BM25 index not found - returning empty lexical results")
            return []

        results = store.search(query_text, top_k=settings.top_k_retrieval)
        return [r for r in results if r.get("text")]

    except Exception as exc:
        # Degrade gracefully for a genuinely-missing/unreadable index, but log
        # the full traceback so hidden programming errors surface.
        logger.error(f"BM25 retrieval failed: {exc}", exc_info=True)
        return []


def _extract_entities_from_query(query_text: str) -> List[str]:
    """
    Heuristic entity extraction from query text used to seed Personalized
    PageRank. This is a lightweight heuristic (capitalized words/phrases and
    quoted terms), not a production NER model (spaCy, etc.).
    """
    import re

    # Heuristic: look for capitalized words and phrases
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
