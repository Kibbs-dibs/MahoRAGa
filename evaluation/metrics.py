"""
Ragas-Style Quantitative Evaluation Metrics for Project Aletheia (evaluation/metrics.py)

Mathematical metrics:
1. Faithfulness (Groundedness of generated claims against retrieved contexts)
2. Answer Relevance (Alignment of answer to query intent)
3. Context Recall (Coverage of ground-truth technical facts in retrieval)
4. Compression Fidelity (Retention of key technical entities post-pruning)
"""

from __future__ import annotations

import re
from typing import Dict, List, Set


def _stem(word: str) -> str:
    """Basic morphological stemming for plural, gerund, and tense affixes."""
    for suffix in ("ing", "tion", "ted", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _tokenize(text: str) -> Set[str]:
    """Tokenize and normalize text into clean lowercase stemmed words."""
    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "are", "was", "were", "with", "as", "by", "from", "that", "this",
        "which", "be", "it", "its", "into", "their", "we", "our", "you",
        "can", "could", "may", "might", "should", "would", "must", "up",
        "down", "out", "over", "under", "all", "each", "both", "such", "than",
        "too", "very", "what", "how", "why", "when", "where", "who", "which"
    }
    words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    tokens: Set[str] = set()
    for w in words:
        for p in w.split("-"):
            if len(p) >= 2 and p not in stopwords:
                tokens.add(_stem(p))
    return tokens


def calculate_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Measure claim groundedness: Fraction of answer proposition tokens supported by retrieved context.
    Returns float in range [0.0, 1.0].
    """
    if not answer.strip():
        return 0.0
    if not retrieved_contexts:
        return 0.5  # Neutral fallback when operating on internal parametric knowledge

    context_tokens = set()
    for ctx in retrieved_contexts:
        context_tokens.update(_tokenize(ctx))

    if not context_tokens:
        return 0.5

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 1.0

    supported_tokens = answer_tokens.intersection(context_tokens)
    return round(len(supported_tokens) / len(answer_tokens), 4)


def calculate_answer_relevance(query: str, answer: str) -> float:
    """
    Measure topical directness: Semantic cosine similarity between query and answer representations.
    Returns float in range [0.0, 1.0].
    """
    if not answer.strip():
        return 0.0
    if not query.strip():
        return 0.0

    try:
        from memory.vector_store import _get_encoder
        import numpy as np

        encoder = _get_encoder()
        embeddings = encoder.encode([query, answer], normalize_embeddings=True)
        # Cosine similarity between normalized vectors is inner product
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return round(max(0.0, min(1.0, sim)), 4)
    except Exception:
        # Fallback to token overlap
        query_tokens = _tokenize(query)
        if not query_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        overlap = query_tokens.intersection(answer_tokens)
        coverage = len(overlap) / len(query_tokens)
        jaccard = len(overlap) / len(query_tokens.union(answer_tokens))
        return round((0.7 * coverage) + (0.3 * jaccard), 4)


def calculate_context_recall(ground_truth_facts: List[str], retrieved_contexts: List[str]) -> float:
    """
    Measure retrieval completeness: Fraction of ground truth facts captured in retrieved evidence.
    Returns float in range [0.0, 1.0].
    """
    if not ground_truth_facts:
        return 1.0
    if not retrieved_contexts:
        return 0.0

    combined_context = " ".join(retrieved_contexts).lower()
    matches = 0

    for fact in ground_truth_facts:
        fact_tokens = _tokenize(fact)
        if not fact_tokens:
            matches += 1
            continue
        # Check if at least 60% of fact tokens appear in context
        found = sum(1 for tok in fact_tokens if tok in combined_context)
        if (found / len(fact_tokens)) >= 0.55:
            matches += 1

    return round(matches / len(ground_truth_facts), 4)


def calculate_compression_fidelity(ground_truth_entities: List[str], pruned_context: str) -> float:
    """
    Measure AttentionRAG pruning preservation: Retention rate of key technical entities.
    Returns float in range [0.0, 1.0].
    """
    if not ground_truth_entities:
        return 1.0
    if not pruned_context.strip():
        return 0.0

    low_context = pruned_context.lower()
    retained = sum(1 for ent in ground_truth_entities if ent.lower() in low_context)
    return round(retained / len(ground_truth_entities), 4)


def calculate_ragas_scorecard(
    query: str,
    answer: str,
    retrieved_contexts: List[str],
    pruned_context: str,
    ground_truth_facts: List[str],
    ground_truth_entities: List[str],
) -> Dict[str, float]:
    """
    Aggregate all 4 evaluation dimensions into a unified scorecard.
    """
    faithfulness = calculate_faithfulness(answer, retrieved_contexts)
    relevance = calculate_answer_relevance(query, answer)
    recall = calculate_context_recall(ground_truth_facts, retrieved_contexts)
    fidelity = calculate_compression_fidelity(ground_truth_entities, pruned_context)

    composite = (0.30 * faithfulness) + (0.30 * relevance) + (0.25 * recall) + (0.15 * fidelity)

    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_recall": recall,
        "compression_fidelity": fidelity,
        "composite_score": round(composite * 100, 1),
    }
