"""
Async External Search Agent for Project Aletheia.

Responsibilities
────────────────
• Reformulates queries and performs iterative searches across external academic APIs
  (Crossref, OpenAlex, and Semantic Scholar) when internal evidence is insufficient.
• Updates the MemoryState within the AletheiaState object directly with external API abstracts.

This node implements direct state manipulation as specified in Approach 1.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
import time
from typing import List, Dict, Any, Optional
import httpx
from core.state import AletheiaState, MemoryState, EvaluationState
from core.config import settings
from core.exceptions import AletheiaError
import logging
import random

logger = logging.getLogger(__name__)


def _calculate_term_overlap(query_terms: set, text_terms: set) -> float:
    """
    Calculate Jaccard similarity (term overlap) between query and text.
    Returns a score between 0.0 and 1.0.
    """
    if not query_terms or not text_terms:
        return 0.0

    intersection = len(query_terms.intersection(text_terms))
    union = len(query_terms.union(text_terms))

    if union > 0:
        return intersection / union
    return 0.0


def _extract_query_terms(query_text: str) -> set:
    """
    Extract meaningful terms from query text for overlap calculation.
    """
    # Extract terms 3+ characters, convert to lowercase
    terms = set(re.findall(r'\b[a-z]{3,}\b', query_text.lower()))
    return terms


def _extract_text_terms(text: str) -> set:
    """
    Extract meaningful terms from text for overlap calculation.
    """
    # Extract terms 3+ characters, convert to lowercase
    terms = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    return terms


async def _search_crossref(query: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    Search Crossref API for academic works.
    """
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": min(settings.top_k_retrieval * 2, 20),  # Get more to filter later
            "sort": "relevance",
            "order": "desc"
        }

        headers = {
            "User-Agent": "ProjectAletheia/1.0 (mailto:research@aletheia.example)"
        }

        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("message", {}).get("items", []):
            # Extract abstract if available
            abstract = item.get("abstract")
            if not abstract:
                # Try to get from other fields
                abstract = item.get("snippet", "")

            # Clean up HTML/XML tags from abstract if present
            if abstract:
                abstract = re.sub(r'<[^>]+>', ' ', abstract)
                abstract = re.sub(r'\s+', ' ', abstract).strip()

            if abstract and len(abstract) > 50:  # Only keep substantial abstracts
                results.append({
                    "title": item.get("title", [""])[0] if item.get("title") else "",
                    "abstract": abstract,
                    "source": "crossref",
                    "doi": item.get("DOI", ""),
                    "year": item.get("issued", {}).get("date-parts", [[None]])[0][0] if item.get("issued") else None,
                    "authors": [author.get("given", "") + " " + author.get("family", "")
                               for author in item.get("author", []) if author.get("given") and author.get("family")]
                })

        return results

    except Exception as exc:
        logger.warning(f"Crossref search failed: {exc}")
        return []


async def _search_openalex(query: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    Search OpenAlex API for academic works.
    """
    try:
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per-page": min(settings.top_k_retrieval * 2, 20),
            "sort": "relevance_score:desc"
        }

        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("results", []):
            # Extract abstract from inverted index if available
            abstract_inverted = item.get("abstract_inverted_index")
            abstract = ""

            if abstract_inverted and isinstance(abstract_inverted, dict):
                # Reconstruct abstract from inverted index
                try:
                    # Create array of position -> word
                    max_pos = max([pos for positions in abstract_inverted.values() for pos in positions]) if abstract_inverted else 0
                    if max_pos > 0:
                        words = [""] * (max_pos + 1)
                        for word, positions in abstract_inverted.items():
                            for pos in positions:
                                if pos < len(words):
                                    words[pos] = word
                        abstract = " ".join(words).strip()
                except Exception:
                    # Fallback if reconstruction fails
                    abstract = ""

            # If no abstract from inverted index, try other fields
            if not abstract:
                abstract = item.get("summary", "")

            if abstract and len(abstract) > 50:  # Only keep substantial abstracts
                results.append({
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "source": "openalex",
                    "doi": item.get("doi", ""),
                    "year": item.get("publication_year"),
                    "authors": [author.get("author", {}).get("display_name", "")
                               for author in item.get("authorships", [])
                               if author.get("author", {}).get("display_name")]
                })

        return results

    except Exception as exc:
        logger.warning(f"OpenAlex search failed: {exc}")
        return []


async def _search_semantic_scholar(query: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar API for academic works.
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(settings.top_k_retrieval * 2, 20),
            "fields": "title,abstract,year,venue,authors,doi"
        }

        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key

        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("data", []):
            abstract = item.get("abstract", "")

            if abstract and len(abstract) > 50:  # Only keep substantial abstracts
                results.append({
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "source": "semantic_scholar",
                    "doi": item.get("doi", ""),
                    "year": item.get("year"),
                    "authors": [author.get("name", "") for author in item.get("authors", []) if author.get("name")]
                })

        return results

    except Exception as exc:
        logger.warning(f"Semantic Scholar search failed: {exc}")
        return []


async def _perform_external_search_with_retry(
    query: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Perform external search with exponential backoff and jitter for rate limiting.
    """
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(1 + max_retries):
            try:
                # Search all three APIs concurrently
                crossref_task = _search_crossref(query, client)
                openalex_task = _search_openalex(query, client)
                semantic_scholar_task = _search_semantic_scholar(query, client)

                crossref_results, openalex_results, semantic_scholar_results = await asyncio.gather(
                    crossref_task, openalex_task, semantic_scholar_task,
                    return_exceptions=True
                )

                # Handle exceptions from individual APIs
                all_results = []
                for results in [crossref_results, openalex_results, semantic_scholar_results]:
                    if isinstance(results, Exception):
                        logger.warning(f"One API search failed: {results}")
                        continue
                    all_results.extend(results)

                return all_results

            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code != 429:  # Not a rate limit error
                    raise

                if attempt >= max_retries:
                    logger.warning(f"HTTP 429 retry exhausted after {max_retries} retries")
                    break

                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay += random.uniform(0, jitter)
                logger.warning(
                    f"HTTP 429 from external API, attempt {attempt + 1}/{max_retries + 1}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                last_exc = e
                if attempt >= max_retries:
                    logger.warning(f"External search retry exhausted after {max_retries} retries")
                    break

                # Exponential backoff with jitter for other errors too
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay += random.uniform(0, jitter)
                logger.warning(
                    f"External search failed, attempt {attempt + 1}/{max_retries + 1}. "
                    f"Retrying in {delay:.1f}s... Error: {e}"
                )
                await asyncio.sleep(delay)

    # If we exhausted all retries, raise the last exception
    assert last_exc is not None
    raise last_exc


def _reformulate_query(
    original_query: str,
    reformulated_query: Optional[str],
    iteration: int,
    max_iterations: int = 3
) -> str:
    """
    Reformulate query for better API results, applying drift guard.
    In a full implementation, this would use an LLM.
    For now, we'll use simple heuristics with drift guard checking.
    """
    # Use the most recent reformulation, or fall back to original
    base_query = reformulated_query or original_query

    if iteration == 0:
        # First iteration - use the query as-is or apply minimal enhancement
        return base_query

    # For subsequent iterations, apply simple expansion techniques
    # In production, this would use LLM-based query expansion/reformulation

    # Extract key terms from the query
    query_terms = _extract_query_terms(base_query)

    # Simple expansion: add related terms based on common academic prefixes/suffixes
    expanded_terms = set(query_terms)

    # Add some common academic modifiers (very basic expansion)
    academic_modifiers = {
        "review", "survey", "analysis", "study", "research", "approach",
        "method", "technique", "algorithm", "framework", "model"
    }

    # Only add modifiers if we haven't exceeded max iterations
    if iteration < max_iterations:
        for term in list(query_terms)[:3]:  # Only expand based on first few terms
            if len(term) > 4:  # Only for substantial terms
                expanded_terms.add(term + "s")  # Plural form
                if term.endswith("y"):
                    expanded_terms.add(term[:-1] + "ies")
                else:
                    expanded_terms.add(term + "ing")

    # Reconstruct query from expanded terms
    # This is a simplified approach - real implementation would be more sophisticated
    expanded_query = " ".join(sorted(list(expanded_terms)[:10]))  # Limit expansion

    # Apply drift guard: check term overlap with original query
    original_terms = _extract_query_terms(original_query)
    expanded_terms_set = _extract_query_terms(expanded_query)

    overlap = _calculate_term_overlap(original_terms, expanded_terms_set)

    if overlap >= settings.drift_guard_min_overlap:
        logger.debug(f"Query reformulation iteration {iteration}: overlap={overlap:.2f} (≥{settings.drift_guard_min_overlap}) - accepting")
        return expanded_query
    else:
        logger.debug(f"Query reformulation iteration {iteration}: overlap={overlap:.2f} (<{settings.drift_guard_min_overlap}) - rejecting drift")
        # Return previous query to prevent drift
        return base_query


def _vet_and_filter_results(
    results: List[Dict[str, Any]],
    original_query: str,
    max_results: int = 10
) -> List[str]:
    """
    Vet and filter search results for quality and relevance.
    Returns list of vetted abstracts.
    """
    if not results:
        return []

    # Extract original query terms for relevance checking
    query_terms = _extract_query_terms(original_query)

    vetted_abstracts = []

    for result in results:
        abstract = result.get("abstract", "")
        if not abstract or len(abstract) < 50:
            continue

        # Calculate term overlap with original query
        text_terms = _extract_text_terms(abstract)
        overlap = _calculate_term_overlap(query_terms, text_terms)

        # Apply ≥30% term overlap guardrail to prevent semantic drift
        if overlap >= settings.drift_guard_min_overlap:
            vetted_abstracts.append(abstract)
        else:
            logger.debug(f"Filtered result due to low term overlap: {overlap:.2f}")

    # Limit results and remove duplicates
    unique_abstracts = []
    seen = set()

    for abstract in vetted_abstracts:
        # Simple deduplication based on first 100 chars
        abstract_key = abstract[:100].lower()
        if abstract_key not in seen and len(unique_abstracts) < max_results:
            seen.add(abstract_key)
            unique_abstracts.append(abstract)

    return unique_abstracts


def search_apis_node(state: AletheiaState) -> AletheiaState:
    """
    External Search Agent node (synchronous wrapper for LangGraph compatibility).

    Reformulates queries and performs iterative searches across external academic APIs
    (Crossref, OpenAlex, and Semantic Scholar) when internal evidence is insufficient.

    Args:
        state: Current AletheiaState object

    Returns:
        Updated AletheiaState object with external API abstracts added to memory state

    Raises:
        AletheiaError: If external search fails
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _search_apis_node_async(state)).result()
    else:
        return asyncio.run(_search_apis_node_async(state))


async def _search_apis_node_async(state: AletheiaState) -> AletheiaState:
    """Internal async implementation of the search APIs node."""
    try:
        logger.info("Starting external search process")

        # Initialize memory state if not present
        if state.memory is None:
            state.memory = MemoryState()

        # Check if evidence is weak before searching - FIXED: trigger when evidence IS weak
        if state.evaluation is None or not state.evaluation.is_evidence_weak:
            logger.info("Evidence is strong enough - skipping external search")
            return state

        # Get the query to search for
        original_query = state.query.original_query
        if not original_query:
            raise AletheiaError("Original query is missing or empty")

        current_query = state.query.reformulated_query or original_query
        logger.info(f"Performing external search for query: {current_query[:100]}...")

        # Initialize external API abstracts list if not present
        if not hasattr(state.memory, 'external_api_abstracts') or state.memory.external_api_abstracts is None:
            state.memory.external_api_abstracts = []

        # Iterative optimize-search-vet loop (max 3 iterations)
        max_iterations = 3
        all_vetted_abstracts = []

        for iteration in range(max_iterations):
            logger.info(f"External search iteration {iteration + 1}/{max_iterations}")

            # Reformulate query for this iteration (with drift guard)
            search_query = _reformulate_query(
                original_query,
                state.query.reformulated_query,
                iteration,
                max_iterations
            )

            logger.debug(f"Search query for iteration {iteration + 1}: {search_query[:100]}...")

            # Perform external search with retry logic
            try:
                raw_results = await _perform_external_search_with_retry(search_query)
                logger.info(f"Retrieved {len(raw_results)} raw results from external APIs")

                # Vet and filter results
                vetted_abstracts = _vet_and_filter_results(
                    raw_results,
                    original_query,
                    max_results=settings.top_k_retrieval
                )

                logger.info(f"Vetted {len(vetted_abstracts)} high-quality abstracts after filtering")
                all_vetted_abstracts.extend(vetted_abstracts)

                # If we got good results, we can stop early
                if len(vetted_abstracts) >= settings.top_k_retrieval:
                    logger.info(f"Sufficient results obtained in iteration {iteration + 1}")
                    break

            except Exception as exc:
                logger.warning(f"External search iteration {iteration + 1} failed: {exc}")
                # Continue to next iteration or fail if this was the last attempt
                if iteration == max_iterations - 1:
                    raise AletheiaError(f"All external search iterations failed: {exc}") from exc
                continue

        # Remove duplicates while preserving order
        seen_abstracts = set()
        final_abstracts = []

        for abstract in all_vetted_abstracts:
            # Create a simple hash for deduplication
            abstract_hash = hash(abstract.lower().strip())
            if abstract_hash not in seen_abstracts:
                seen_abstracts.add(abstract_hash)
                final_abstracts.append(abstract)

        # Update state with vetted external API abstracts
        state.memory.external_api_abstracts = final_abstracts

        logger.info(
            f"External search complete: added {len(final_abstracts)} vetted abstracts to memory state"
        )
        logger.debug(f"Total external API abstracts in memory: {len(state.memory.external_api_abstracts)}")

        return state

    except Exception as exc:
        logger.error(f"Search APIs node failed: {exc}")
        raise AletheiaError(f"Failed to perform external search: {exc}") from exc