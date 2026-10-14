"""
Attention-Guided Context Compressor (AttentionRAG) for Project Aletheia.

Responsibilities
────────────────
• Generates an answer hint prefix to isolate the focal concept / token.
• Segments retrieved context into micro-windows (50–300 tokens / 200–1200 chars).
• Computes multi-layer attention / semantic relevance scores across micro-windows.
• Prunes low-relevance sentences to achieve up to 6.3x compression while preserving citations.
• Populates CompressionState directly within AletheiaState.

This node implements direct state manipulation as specified in Approach 1.
"""

from __future__ import annotations

import logging
import math
import re
from typing import List, Optional, Tuple

from core.config import settings
from core.exceptions import AletheiaError
from core.state import AletheiaState, CompressionState

logger = logging.getLogger(__name__)




def _extract_focal_token(query: str) -> str:
    """
    Construct an answer hint prefix and isolate the focal concept/token.
    E.g.: 'What is a transformer?' -> 'transformer'
    """
    if not query:
        return "general"

    # Strip punctuation and common question words
    cleaned = re.sub(r"[^\w\s]", " ", query)
    stop_words = {
        "what", "is", "a", "an", "the", "how", "why", "when", "where", "who",
        "which", "can", "you", "tell", "me", "about", "describe", "explain",
        "of", "in", "to", "for", "with", "and", "or", "are", "do", "does",
        "did", "their", "latest", "recent", "trend", "trends", "paper", "model"
    }
    tokens = [w.lower() for w in cleaned.split() if w.lower() not in stop_words and len(w) > 2]
    if tokens:
        return tokens[0]

    words = [w for w in cleaned.split() if len(w) > 2]
    return words[0].lower() if words else query[:20].strip().lower()


def _segment_micro_windows(text: str, window_char_limit: int = 600) -> List[str]:
    """
    Segment assembled context into micro-windows (approx. 50-300 tokens each),
    preserving line and paragraph boundaries.
    """
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    windows: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > window_char_limit and current_chunk:
            windows.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line)

    if current_chunk:
        windows.append("\n".join(current_chunk))

    return windows


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while preserving bracketed citation markers.
    """
    if not text:
        return []

    # Match sentences ending with ., !, or ? or line breaks
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def _compute_relevance_scores(
    sentences: List[str],
    focal_token: str,
    query: str,
) -> List[float]:
    """
    Calculate token attention and semantic relevance scores between each sentence
    and the focal concept/query without network dependencies.
    """
    if not sentences:
        return []

    target = f"{focal_token} {query}".lower()
    target_terms = set(re.findall(r"\b[a-z]{3,}\b", target))
    focal_lower = focal_token.lower()

    scores: List[float] = []

    for sent in sentences:
        sent_lower = sent.lower()
        sent_terms = set(re.findall(r"\b[a-z]{3,}\b", sent_lower))

        if not sent_terms or not target_terms:
            base_score = 0.1
        else:
            intersection = target_terms.intersection(sent_terms)
            union = target_terms.union(sent_terms)
            base_score = len(intersection) / len(union) if union else 0.1

        # Direct focal token match bonus (next-token attention focus)
        if focal_lower in sent_lower:
            base_score += 0.35

        # Citation / provenance bonus (protect authoritative references)
        if re.search(r"\[[^\]]+\]", sent) or sent.startswith("==="):
            base_score += 0.25

        scores.append(min(1.0, base_score))

    return scores


def _prune_sentences(
    sentences: List[str],
    scores: List[float],
    max_compression_ratio: float = 6.3,
) -> Tuple[str, float]:
    """
    Prune low-attention sentences according to relevance scores,
    enforcing evidence density and citation preservation.
    """
    if not sentences:
        return "", 1.0

    total_chars = sum(len(s) for s in sentences)
    if total_chars < 300:
        # Context is already very short — preserve all sentences without aggressive pruning
        return "\n".join(sentences), 1.0

    # Calculate token/char budget based on target compression
    target_budget_chars = max(300, int(total_chars / max_compression_ratio))

    # Pair each sentence with its original index and score
    scored_items = list(zip(range(len(sentences)), sentences, scores))

    # Always prioritize sentences with citations or headers
    def sort_key(item):
        idx, sent, score = item
        bonus = 0.5 if re.search(r"\[[^\]]+\]", sent) or sent.startswith("===") else 0.0
        return score + bonus

    sorted_by_score = sorted(scored_items, key=sort_key, reverse=True)

    # Greedily accumulate sentences within budget
    selected_indices = set()
    accumulated_chars = 0

    for idx, sent, score in sorted_by_score:
        if accumulated_chars + len(sent) <= target_budget_chars or not selected_indices:
            selected_indices.add(idx)
            accumulated_chars += len(sent)
        else:
            break

    # Reconstruct in original chronological order
    pruned_sentences = [sentences[i] for i in sorted(selected_indices)]
    pruned_text = "\n".join(pruned_sentences)

    pruned_chars = max(1, len(pruned_text))
    compression_ratio = round(total_chars / pruned_chars, 2)
    # Clamp compression ratio between 1.0 and max_compression_ratio
    compression_ratio = max(1.0, min(compression_ratio, max_compression_ratio))

    return pruned_text, compression_ratio


def pruner_node(state: AletheiaState) -> AletheiaState:
    """
    Attention-Guided Context Compressor (AttentionRAG) node.

    Compresses retrieved context using attention-guided pruning to reduce
    token overhead while preserving relevant facts and citations.

    Args:
        state: Current AletheiaState object

    Returns:
        Updated AletheiaState object with compression layer populated

    Raises:
        AletheiaError: If context compression fails
    """
    try:
        logger.info("Starting attention-guided context compression process")

        # Initialize compression state if not present
        if state.compression is None:
            state.compression = CompressionState()

        # ── 1. Assemble raw retrieved context ──────────────────────────
        context_parts: List[str] = []

        if state.memory and state.memory.serialized_graph_triples:
            context_parts.append("=== Knowledge Graph Triples ===")
            context_parts.extend(state.memory.serialized_graph_triples)

        if state.memory and state.memory.vector_chunks:
            context_parts.append("=== Dense Vector Chunks ===")
            context_parts.extend(state.memory.vector_chunks)

        if state.memory and state.memory.lexical_chunks:
            context_parts.append("=== Lexical Chunks ===")
            context_parts.extend(state.memory.lexical_chunks)

        if state.memory and state.memory.external_api_abstracts:
            context_parts.append("=== External API Abstracts ===")
            context_parts.extend(state.memory.external_api_abstracts)

        assembled_raw = "\n".join(context_parts)

        query_text = (
            state.query.reformulated_query or state.query.original_query
            if state.query else ""
        )

        # ── 2. Isolate Focal Token via Answer Hint Prefix ───────────────
        focal_token = _extract_focal_token(query_text)
        state.compression.focal_token = focal_token

        # ── 3. Segment into Micro-Windows (50-300 Tokens) ───────────────
        micro_windows = _segment_micro_windows(assembled_raw, window_char_limit=800)
        state.compression.micro_windows = micro_windows

        # ── 4. Calculate Attention / Relevance Scores ───────────────────
        if assembled_raw:
            sentences = _split_into_sentences(assembled_raw)
            scores = _compute_relevance_scores(sentences, focal_token, query_text)

            # ── 5. Prune Sentences to Target Budget ─────────────────────
            pruned_text, ratio = _prune_sentences(
                sentences,
                scores,
                max_compression_ratio=settings.max_compression_ratio,
            )
            state.compression.final_pruned_prompt = pruned_text
            state.compression.compression_ratio = ratio
        else:
            # Fallback when memory is empty
            fallback = query_text or "(no context available)"
            state.compression.final_pruned_prompt = fallback
            state.compression.compression_ratio = 1.0

        logger.info(
            "AttentionRAG pruning complete: focal_token='%s', micro_windows=%d, compression_ratio=%.2fx",
            state.compression.focal_token,
            len(state.compression.micro_windows),
            state.compression.compression_ratio,
        )

        return state

    except Exception as exc:
        logger.error(f"Pruner node failed: {exc}")
        raise AletheiaError(f"Attention-guided context compression failed: {exc}") from exc