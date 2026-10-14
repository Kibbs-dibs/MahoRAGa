"""
Evidence Evaluator & Scorer for Project Aletheia.

Responsibilities
────────────────
• Audits retrieved context using a 100-Point Evidence Sufficiency Rubric:
  - Retrieval Confidence (0-40 points)
  - Answer Specificity (0-25 points)
  - Source Diversity (0-15 points)
  - Metadata Completeness (0-10 points)
  - Recency (0-10 points)
• Triggers external search if the score is below 80.
• Updates the EvaluationState within the AletheiaState object directly.

This node implements direct state manipulation as specified in Approach 1.
"""

from __future__ import annotations

from core.state import AletheiaState, EvaluationState
from core.config import settings
from core.exceptions import AletheiaError
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


def evaluator_node(state: AletheiaState) -> AletheiaState:
    """
    Evidence Evaluator & Scorer node.

    Audits retrieved context using the 100-Point Evidence Sufficiency Rubric.

    Args:
        state: Current AletheiaState object

    Returns:
        Updated AletheiaState object with evaluation scores set

    Raises:
        AletheiaError: If evaluation fails
    """
    try:
        logger.info("Starting evidence evaluation process")

        # Initialize evaluation state if not present
        if state.evaluation is None:
            state.evaluation = EvaluationState()

        # Reset scores for fresh evaluation
        state.evaluation.retrieval_confidence = 0
        state.evaluation.answer_specificity = 0
        state.evaluation.source_diversity = 0
        state.evaluation.metadata_completeness = 0
        state.evaluation.recency = 0

        # Extract retrieved content from memory state
        memory = state.memory
        if memory is None:
            logger.warning("No memory state found - setting all scores to 0")
            state.evaluation.is_evidence_weak = True
            return state

        # Calculate each component of the rubric
        state.evaluation.retrieval_confidence = _score_retrieval_confidence(memory)
        state.evaluation.answer_specificity = _score_answer_specificity(state.query, memory)
        state.evaluation.source_diversity = _score_source_diversity(memory)
        state.evaluation.metadata_completeness = _score_metadata_completeness(memory)
        state.evaluation.recency = _score_recency(memory)

        # Calculate total score
        state.evaluation.rubric_score = (
            state.evaluation.retrieval_confidence +
            state.evaluation.answer_specificity +
            state.evaluation.source_diversity +
            state.evaluation.metadata_completeness +
            state.evaluation.recency
        )

        # Determine if evidence is weak (triggers external search)
        state.evaluation.is_evidence_weak = state.evaluation.rubric_score < settings.evidence_threshold

        logger.info(
            f"Evidence evaluation complete: {state.evaluation.rubric_score}/100 "
            f"(Confidence: {state.evaluation.retrieval_confidence}/40, "
            f"Specificity: {state.evaluation.answer_specificity}/25, "
            f"Diversity: {state.evaluation.source_diversity}/15, "
            f"Completeness: {state.evaluation.metadata_completeness}/10, "
            f"Recency: {state.evaluation.recency}/10)"
        )
        logger.info(f"Evidence weak: {state.evaluation.is_evidence_weak} (threshold: {settings.evidence_threshold})")

        return state

    except Exception as exc:
        logger.error(f"Evaluator node failed: {exc}")
        raise AletheiaError(f"Evidence evaluation failed: {exc}") from exc


def _score_retrieval_confidence(memory: MemoryState) -> int:
    """
    Score retrieval confidence (0-40 points).
    Based on quantity and quality of retrieved results across all layers.
    """
    score = 0

    # Points for having results from each layer (max 15 points)
    layers_with_results = 0
    if memory.serialized_graph_triples and len(memory.serialized_graph_triples) > 0:
        layers_with_results += 1
    if memory.vector_chunks and len(memory.vector_chunks) > 0:
        layers_with_results += 1
    if memory.lexical_chunks and len(memory.lexical_chunks) > 0:
        layers_with_results += 1

    # 5 points per layer with results, up to 15 points
    score += min(layers_with_results * 5, 15)

    # Points for quantity of results (up to 25 points)
    total_results = len(memory.serialized_graph_triples) + len(memory.vector_chunks) + len(memory.lexical_chunks)

    # Scale: 0 results = 0 points, 5+ results = 25 points
    if total_results >= 5:
        score += 25
    elif total_results > 0:
        # Linear scaling: 1 result = 5 points, 2 = 10, 3 = 15, 4 = 20
        score += total_results * 5

    return min(score, 40)  # Cap at 40 points


def _score_answer_specificity(query_state: QueryState, memory: MemoryState) -> int:
    """
    Score answer specificity (0-25 points).
    Based on how specific and focused the retrieved information is to the query.
    """
    if not query_state.reformulated_query and not query_state.original_query:
        return 0

    query_text = (query_state.reformulated_query or query_state.original_query).lower()
    query_terms = set(re.findall(r'\b[a-z]{3,}\b', query_text))  # Terms 3+ chars

    if not query_terms:
        return 0

    score = 0
    max_score = 25

    # Check specificity in each layer's results
    specificity_scores = []

    # Graph triples
    if memory.serialized_graph_triples:
        graph_specificity = _calculate_text_specificity(
            memory.serialized_graph_triples, query_terms
        )
        specificity_scores.append(graph_specificity)

    # Vector chunks
    if memory.vector_chunks:
        vector_specificity = _calculate_text_specificity(
            memory.vector_chunks, query_terms
        )
        specificity_scores.append(vector_specificity)

    # Lexical chunks
    if memory.lexical_chunks:
        lexical_specificity = _calculate_text_specificity(
            memory.lexical_chunks, query_terms
        )
        specificity_scores.append(lexical_specificity)

    # Average specificity across layers that have results
    if specificity_scores:
        avg_specificity = sum(specificity_scores) / len(specificity_scores)
        score = int(avg_specificity * max_score)
        score = min(score, max_score)  # Cap at max

    return score


def _score_source_diversity(memory: MemoryState) -> int:
    """
    Score source diversity (0-15 points).
    Based on variety of sources and types of information retrieved.
    """
    score = 0
    max_score = 15

    # Points for diversity across the three layers (max 9 points)
    layers_used = 0
    if memory.serialized_graph_triples and len(memory.serialized_graph_triples) > 0:
        layers_used += 1
    if memory.vector_chunks and len(memory.vector_chunks) > 0:
        layers_used += 1
    if memory.lexical_chunks and len(memory.lexical_chunks) > 0:
        layers_used += 1

    # 3 points per layer used, up to 9 points
    score += layers_used * 3

    # Points for diversity within results (up to 6 points)
    # Simple heuristic: check for varied content length and structure
    all_results = (
        memory.serialized_graph_triples +
        memory.vector_chunks +
        memory.lexical_chunks
    )

    if len(all_results) >= 3:
        # Check if we have varied content (different lengths, structures)
        lengths = [len(text) for text in all_results if text]
        if len(set(lengths)) >= 2:  # At least 2 different lengths
            score += 3
        if len(all_results) >= 5:  # Good variety of results
            score += 3

    return min(score, max_score)


def _score_metadata_completeness(memory: MemoryState) -> int:
    """
    Score metadata completeness (0-10 points).
    Based on richness of metadata associated with retrieved results.
    """
    score = 0
    max_score = 10

    # For now, we'll give points based on having any results
    # In a full implementation, this would check actual metadata fields
    total_results = len(memory.serialized_graph_triples) + len(memory.vector_chunks) + len(memory.lexical_chunks)

    if total_results > 0:
        # Basic points for having results
        score += 5
        # Additional points for substantial results
        if total_results >= 3:
            score += 3
        if total_results >= 5:
            score += 2

    return min(score, max_score)


def _score_recency(memory: MemoryState) -> int:
    """
    Score recency (0-10 points).
    Based on how recent the information is.
    """
    score = 0
    max_score = 10

    # For now, we'll use a simple heuristic based on content analysis
    # In a full implementation, this would check timestamps, dates in content, etc.
    all_results = (
        memory.serialized_graph_triples +
        memory.vector_chunks +
        memory.lexical_chunks
    )

    if all_results:
        # Basic points for having any results (assumes some recency)
        score += 5

        # Additional points if content seems to contain recent terminology
        recent_indicators = [
            '2023', '2024', '2025', '2026',  # Recent years
            'latest', 'recent', 'new', 'emerging', 'current',
            'state-of-the-art', 'cutting-edge', 'novel'
        ]

        combined_text = ' '.join(all_results).lower()
        recent_matches = sum(1 for indicator in recent_indicators if indicator in combined_text)

        if recent_matches >= 2:
            score += 3
        elif recent_matches >= 1:
            score += 2

        # Bonus for multiple recent indicators
        if recent_matches >= 4:
            score += 2

    return min(score, max_score)


def _calculate_text_specificity(texts: List[str], query_terms: set) -> float:
    """
    Calculate how specific the texts are to the query terms.
    Returns a score between 0.0 and 1.0.
    """
    if not texts or not query_terms:
        return 0.0

    specificity_scores = []

    for text in texts:
        if not text:
            specificity_scores.append(0.0)
            continue

        text_terms = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        if not text_terms:
            specificity_scores.append(0.0)
            continue

        # Calculate Jaccard similarity: intersection over union
        intersection = len(query_terms.intersection(text_terms))
        union = len(query_terms.union(text_terms))

        if union > 0:
            specificity = intersection / union
            specificity_scores.append(specificity)
        else:
            specificity_scores.append(0.0)

    # Return average specificity
    return sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0.0