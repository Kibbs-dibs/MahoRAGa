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
        degraded: List[str] = []
        neo4j_triples, neo4j_failed = _retrieve_from_neo4j(query_text)
        if neo4j_failed:
            degraded.append("neo4j")
        state.memory.serialized_graph_triples = neo4j_triples
        logger.debug(f"Retrieved {len(neo4j_triples)} triples from Neo4j")

        # Retrieve from FAISS (dense vector layer) — ranked dicts
        vector_results, faiss_failed = _retrieve_from_faiss(query_text)
        if faiss_failed:
            degraded.append("faiss")
        # Retrieve from BM25 (sparse lexical layer) — ranked dicts
        lexical_results, bm25_failed = _retrieve_from_bm25(query_text)
        if bm25_failed:
            degraded.append("bm25")
        state.memory.degraded_layers = degraded
        if degraded:
            logger.warning("Retrieval ran degraded — layers unavailable: %s", ", ".join(degraded))

        # Reciprocal Rank Fusion (RRF) over graph + dense + lexical ranked lists
        # (Features 1.1/4): graph triples join the fusion as a third ranked list,
        # producing one merged ranking that becomes the primary retrieval output.
        graph_results = [{"text": t} for t in neo4j_triples if t]
        fused = SparseStore.reciprocal_rank_fusion(graph_results, vector_results, lexical_results)

        # Optional cross-encoder re-ranking of the top fused candidates for final
        # precision (Feature 1.2). Degrades gracefully to RRF-only on any failure
        # (e.g. model can't load offline).
        if settings.cross_encoder_rerank and fused:
            try:
                pool = fused[: settings.rerank_candidate_pool]
                reranked = SparseStore().rerank(query_text, pool, top_k=len(pool))
                # Re-ranked pool first (in CE order), then any remaining fused tail.
                fused = reranked + fused[settings.rerank_candidate_pool:]
                logger.debug(f"Cross-encoder re-ranked {len(pool)} fused candidates")
            except Exception as exc:
                logger.warning(f"Cross-encoder re-rank skipped (RRF-only): {exc}")

        # Primary retrieval output: the fused ranked texts (Feature 1.1).
        state.memory.fused_chunks = [d.get("text", "") for d in fused if d.get("text")]

        fused_order = {doc.get("text", ""): rank for rank, doc in enumerate(fused)}

        def _ordered_texts(results):
            texts = [r.get("text", "") for r in results if r.get("text")]
            return sorted(texts, key=lambda t: fused_order.get(t, len(fused_order)))

        state.memory.vector_chunks = _ordered_texts(vector_results)
        logger.debug(f"Retrieved {len(state.memory.vector_chunks)} chunks from FAISS")

        state.memory.lexical_chunks = _ordered_texts(lexical_results)
        logger.debug(f"Retrieved {len(state.memory.lexical_chunks)} chunks from BM25")

        vector_chunks = state.memory.vector_chunks
        lexical_chunks = state.memory.lexical_chunks

        logger.info(
            f"Retrieval complete: {len(neo4j_triples)} graph triples, "
            f"{len(vector_chunks)} vector chunks, {len(lexical_chunks)} lexical chunks"
        )

        return state

    except Exception as exc:
        logger.error(f"Retriever node failed: {exc}")
        raise AletheiaError(f"Memory-guided retrieval failed: {exc}") from exc


def _retrieve_from_neo4j(query_text: str) -> tuple[List[str], bool]:
    """Retrieve and serialize relevant triples from Neo4j using PPR.

    Returns (triples, failed): ``failed`` is True when the graph layer was
    unreachable/errored (distinct from simply finding nothing), so the caller
    can flag degraded retrieval.
    """
    try:
        with Neo4jClient() as client:
            # NOTE: index/constraint creation is done once during ingestion
            # (LiveIngestionService.ingest_files -> client.ensure_indexes()); it is
            # intentionally NOT called here, to keep the per-query read path free of
            # a schema round-trip (Issue 5).

            # Extract entities from query for PPR seeding
            # Simple approach: use nouns/named entities as seeds
            # In production, this would use proper NER
            seed_entities = _extract_entities_from_query(query_text)

            all_triples = []
            for entity in seed_entities[:3]:  # Limit to top 3 entities to avoid overload
                try:
                    triples = client.serialise_subgraph_to_sentences(
                        entity, limit=settings.ppr_top_k
                    )
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

            return deduped_triples[:settings.top_k_retrieval], False

    except Exception as exc:
        logger.error(f"Neo4j retrieval failed: {exc}")
        # Return empty list rather than failing completely - allows other layers to work
        return [], True


def _retrieve_from_faiss(query_text: str) -> tuple[List[Dict[str, Any]], bool]:
    """Retrieve relevant chunks from FAISS vector store.

    Returns (results, failed): ``failed`` is True only on a load/search ERROR
    (a missing index is treated as an empty-but-healthy corpus, not a failure).
    """
    try:
        store = VectorStore()
        # Try to load existing index; if not available, return empty results
        if not store.load_if_exists():
            logger.warning("FAISS index not found - returning empty vector results")
            return [], False

        results = store.search(query_text, top_k=settings.top_k_retrieval)
        # Return ranked result dicts (order == rank) so the caller can RRF-fuse.
        return [r for r in results if r.get('text')], False

    except Exception as exc:
        logger.error(f"FAISS retrieval failed: {exc}")
        # Return empty list rather than failing completely
        return [], True


def _retrieve_from_bm25(query_text: str) -> tuple[List[Dict[str, Any]], bool]:
    """Retrieve relevant chunks from BM25 sparse store.

    Returns (results, failed): ``failed`` is True only on a load/search ERROR
    (a missing index is treated as an empty-but-healthy corpus, not a failure).
    """
    try:
        store = SparseStore()
        # Try to load existing index; if not available, return empty results
        if not store.load_if_exists():
            logger.warning("BM25 index not found - returning empty lexical results")
            return [], False

        results = store.search(query_text, top_k=settings.top_k_retrieval)
        # Return ranked result dicts (order == rank) so the caller can RRF-fuse.
        return [r for r in results if r.get('text')], False

    except Exception as exc:
        logger.error(f"BM25 retrieval failed: {exc}")
        # Return empty list rather than failing completely
        return [], True


def _extract_entities_from_query(query_text: str) -> List[str]:
    """Extract PPR seed entities from the query (Feature 1.3).

    Tries spaCy NER when available (captures real named/technical entities),
    then falls back to an improved keyphrase heuristic that ALSO picks up
    lowercase multi-word technical terms (e.g. "retrieval augmented generation"),
    not just capitalized tokens.
    """
    import re

    # 1) spaCy NER if installed (optional dependency).
    try:
        import spacy  # type: ignore
        if not hasattr(_extract_entities_from_query, "_nlp"):
            _extract_entities_from_query._nlp = spacy.load("en_core_web_sm")  # type: ignore
        doc = _extract_entities_from_query._nlp(query_text)  # type: ignore
        ents = [e.text.strip() for e in doc.ents if e.text.strip()]
        nouns = [c.text.strip() for c in doc.noun_chunks if len(c.text.strip()) > 2]
        seeds = ents + nouns
        if seeds:
            seen, out = set(), []
            for s in seeds:
                if s.lower() not in seen:
                    seen.add(s.lower()); out.append(s)
            return out[:5]
    except Exception:
        pass  # spaCy not installed or model missing -> heuristic below

    # 2) Heuristic fallback: capitalized phrases + quoted terms + salient
    #    lowercase keyphrases (stopword-filtered), so technical terms are kept.
    candidates: List[str] = []
    candidates += re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query_text)
    candidates += re.findall(r'"([^"]*)"', query_text)

    stop = {
        "what", "which", "how", "why", "when", "where", "who", "the", "a", "an",
        "is", "are", "do", "does", "did", "can", "of", "in", "to", "for", "and",
        "or", "with", "about", "tell", "me", "explain", "describe", "compare",
        "between", "difference", "latest", "recent",
    }
    tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", query_text.lower()) if t not in stop]
    # salient single terms
    candidates += tokens
    # salient adjacent bigrams (capture multi-word technical terms)
    candidates += [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

    seen = set()
    unique_words = []
    for word in candidates:
        w = word.strip()
        if w and w.lower() not in seen:
            seen.add(w.lower())
            unique_words.append(w)

    # Fallback: if no entities found, use the whole query as a single entity
    if not unique_words:
        unique_words = [query_text[:50]]

    return unique_words[:5]  # Limit to top 5 entities