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


# ── External-API circuit breaker ─────────────────────────────────────────
# Mirrors the LLM client's _disable_backend pattern: after N consecutive
# failures a source is "opened" (skipped) for a cooldown, so a down or
# rate-limiting API (429/5xx storms) is not hammered on every query. State is
# process-local and self-healing — the next call after the cooldown probes once.
_BREAKER_FAILS: Dict[str, int] = {}
_BREAKER_OPEN_UNTIL: Dict[str, float] = {}
_BREAKER_FAIL_THRESHOLD = 3
_BREAKER_COOLDOWN_SECONDS = 60.0


def _breaker_is_open(source: str) -> bool:
    """True if this source is in cooldown and should be skipped."""
    return time.time() < _BREAKER_OPEN_UNTIL.get(source, 0.0)


def _breaker_record_success(source: str) -> None:
    _BREAKER_FAILS[source] = 0
    _BREAKER_OPEN_UNTIL.pop(source, None)


def _breaker_record_failure(source: str) -> None:
    n = _BREAKER_FAILS.get(source, 0) + 1
    _BREAKER_FAILS[source] = n
    if n >= _BREAKER_FAIL_THRESHOLD:
        _BREAKER_OPEN_UNTIL[source] = time.time() + _BREAKER_COOLDOWN_SECONDS
        logger.warning(
            "Circuit breaker OPEN for '%s' after %d consecutive failures; "
            "skipping for %.0fs.", source, n, _BREAKER_COOLDOWN_SECONDS,
        )


async def _guarded_search(source: str, coro_factory):
    """Run an external search under the breaker: skip if open, trip on failure.

    ``coro_factory`` is a zero-arg callable returning the search coroutine, so a
    skipped source never even constructs the request.
    """
    if _breaker_is_open(source):
        logger.info("Skipping '%s' (circuit breaker open).", source)
        return []
    try:
        results = await coro_factory()
        _breaker_record_success(source)
        return results
    except Exception as exc:
        _breaker_record_failure(source)
        logger.warning("External source '%s' failed: %s", source, exc)
        return []


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

async def _search_arxiv(query: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Search the arXiv Atom API for preprints (Feature 5.4: broader coverage).

    Reuses the watcher's Atom parser so abstract/title/author/year extraction
    stays consistent with ingestion.
    """
    try:
        import urllib.parse
        from agents.arxiv_watcher import parse_arxiv_atom_feed

        params = {
            "search_query": f"all:{query.strip()}",
            "start": 0,
            "max_results": min(settings.top_k_retrieval * 2, 20),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        response = await client.get(url)
        response.raise_for_status()

        results: List[Dict[str, Any]] = []
        for paper in parse_arxiv_atom_feed(response.text):
            if paper.abstract and len(paper.abstract) > 50:
                results.append({
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "source": "arxiv",
                    "doi": "",
                    "year": int(paper.published[:4]) if paper.published[:4].isdigit() else None,
                    "authors": paper.authors,
                })
        return results
    except Exception as exc:
        logger.warning(f"arXiv search failed: {exc}")
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
                # Search all academic APIs concurrently, each under its circuit
                # breaker so a repeatedly-failing source is skipped for a cooldown.
                sources = [
                    ("crossref", lambda: _search_crossref(query, client)),
                    ("openalex", lambda: _search_openalex(query, client)),
                    ("semantic_scholar", lambda: _search_semantic_scholar(query, client)),
                    ("arxiv", lambda: _search_arxiv(query, client)),
                ]
                gathered = await asyncio.gather(
                    *[_guarded_search(name, factory) for name, factory in sources],
                    return_exceptions=True,
                )

                # Handle exceptions from individual APIs
                all_results = []
                rate_limited = False
                for (name, _factory), results in zip(sources, gathered):
                    if isinstance(results, Exception):
                        # Detect rate-limit errors captured by return_exceptions=True
                        if isinstance(results, httpx.HTTPStatusError) and results.response.status_code == 429:
                            rate_limited = True
                            last_exc = results
                        _breaker_record_failure(name)
                        logger.warning(f"One API search failed: {results}")
                        continue
                    all_results.extend(results)

                # If we got usable results, return them (partial success is fine).
                if all_results:
                    return all_results

                # No results AND at least one API was rate-limited → back off and retry.
                if rate_limited and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay += random.uniform(0, jitter)
                    logger.warning(
                        f"HTTP 429 from external API(s), attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

                # No results and no retryable rate-limit (or retries exhausted) → done.
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

    # Feature 5.2: prefer an LLM-generated academic reformulation for subsequent
    # iterations (the mechanical plural/-ing expansion below rarely helps academic
    # recall). Drift-guard the LLM output and fall back to the heuristic on any
    # failure or drift, so offline / no-LLM operation still works.
    try:
        from core.llm import llm_client
        prompt = (
            f"Rewrite this academic search query to improve recall on scholarly APIs "
            f"(Crossref/OpenAlex/Semantic Scholar). Expand acronyms, add close synonyms, "
            f"keep it one line, no commentary.\n\nQuery: {base_query}"
        )
        resp = llm_client.call(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You reformulate academic search queries. Output only the rewritten query.",
            temperature=0.2,
            max_tokens=60,
        )
        candidate = (resp.content or "").strip().splitlines()[0].strip() if resp and resp.content else ""
        if candidate:
            overlap = _calculate_term_overlap(
                _extract_query_terms(original_query), _extract_query_terms(candidate)
            )
            if overlap >= settings.drift_guard_min_overlap:
                logger.debug(f"LLM reformulation accepted (overlap={overlap:.2f}): {candidate[:80]}")
                return candidate
            logger.debug(f"LLM reformulation rejected for drift (overlap={overlap:.2f}); using heuristic")
    except Exception as exc:
        logger.debug(f"LLM reformulation unavailable, using heuristic: {exc}")

    # Heuristic fallback (also offline path) ----------------------------------
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

        # Drop external abstracts already covered by the internal corpus
        # (Feature 5.3): if an abstract is highly similar to an existing internal
        # chunk, it adds no new evidence — prune it so external results complement
        # rather than duplicate internal retrieval. Graceful no-op offline.
        try:
            internal = (list(state.memory.serialized_graph_triples)
                        + list(state.memory.vector_chunks)
                        + list(state.memory.lexical_chunks))
            if final_abstracts and internal:
                from memory.vector_store import VectorStore
                enc = VectorStore().encode
                import numpy as _np
                iv = enc(internal)                      # L2-normalised
                av = enc(final_abstracts)
                kept = []
                for i, a in enumerate(final_abstracts):
                    max_sim = float(_np.max(iv @ av[i])) if len(internal) else 0.0
                    if max_sim < 0.92:                  # not a near-duplicate of internal
                        kept.append(a)
                if kept:
                    logger.info("5.3 dedup: kept %d/%d external abstracts (dropped near-duplicates of internal corpus)",
                                len(kept), len(final_abstracts))
                    final_abstracts = kept
        except Exception as exc:
            logger.debug(f"External-vs-internal dedup skipped: {exc}")

        # Update state with vetted external API abstracts
        state.memory.external_api_abstracts = final_abstracts

        # Optionally persist vetted abstracts so the corpus self-enriches from use
        # (Feature 5.1). Gated off by default since it mutates the persistent stores.
        if final_abstracts and settings.persist_external_abstracts:
            try:
                from memory.vector_store import VectorStore
                from memory.sparse_store import SparseStore
                metas = [{"text": a, "source": "external_search", "document_id": "external"} for a in final_abstracts]
                v = VectorStore(); v.load_if_exists(); v.add_chunks(list(final_abstracts), list(metas)); v.save()
                s = SparseStore(); s.load_if_exists(); s.add_documents(list(final_abstracts), list(metas)); s.save()
                logger.info("Persisted %d external abstracts into FAISS+BM25 (corpus self-enrichment)", len(final_abstracts))
            except Exception as exc:
                logger.warning(f"External-abstract persistence skipped: {exc}")

        # Re-score evidence now that external abstracts were added (Issue 11): the
        # rubric was computed on internal evidence only, so re-run the evaluator so
        # is_evidence_weak / rubric_score reflect the external results before pruning.
        # Done inline (not via a graph edge) to keep the StateGraph acyclic.
        if final_abstracts and state.evaluation is not None:
            try:
                from graph.nodes.evaluator import evaluator_node
                evaluator_node(state)
                logger.info(
                    "Re-evaluated after external search: rubric=%d, weak=%s",
                    state.evaluation.rubric_score,
                    state.evaluation.is_evidence_weak,
                )
            except Exception as exc:
                logger.warning(f"Post-search re-evaluation skipped: {exc}")

        logger.info(
            f"External search complete: added {len(final_abstracts)} vetted abstracts to memory state"
        )
        logger.debug(f"Total external API abstracts in memory: {len(state.memory.external_api_abstracts)}")

        return state

    except Exception as exc:
        logger.error(f"Search APIs node failed: {exc}")
        raise AletheiaError(f"Failed to perform external search: {exc}") from exc