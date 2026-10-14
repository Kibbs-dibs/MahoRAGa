"""
Query Router & Reformulator Agent for Project Aletheia.

Responsibilities
────────────────
• Classifies user intent into one of four routes: 'content', 'bibliometric', 'trend', or 'current_world'.
• Normalizes and reformulates the user query for optimal technical retrieval using an LLM.
• Updates the QueryState within the AletheiaState object directly.

This node implements direct state manipulation as specified in Approach 1.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.exceptions import AletheiaError
from core.llm import llm_client
from core.state import AletheiaState, QueryState

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the master routing agent for Project Aletheia, an AI tracking and reasoning system.
Your task is to analyze the user's input, classify its intent, and rewrite it into a highly optimized search query for a FAISS/BM25 dual-index and Neo4j graph.

ROUTING CATEGORIES:
- "content": Questions about model architectures, methodologies, or technical mechanisms.
- "bibliometric": Questions about specific authors, researchers, or publication histories.
- "trend": Questions asking for landscape mapping or temporal evolutions of a technology.
- "current_world": Generic greetings, trivia, or requests outside the AI/CS domain.

REFORMULATION RULES:
1. Strip away conversational filler (e.g., "Can you tell me about...").
2. Expand acronyms if implied by the context (e.g., "RLHF" -> "Reinforcement Learning from Human Feedback").
3. Maintain domain-specific technical jargon.

OUTPUT FORMAT:
You must respond with a raw JSON object containing exactly two keys:
{
  "route": "<category>",
  "reformulated_query": "<optimized search string>"
}
"""

VALID_ROUTES = {"content", "bibliometric", "trend", "current_world"}


def router_node(state: AletheiaState) -> AletheiaState:
    """
    Query Router & Reformulator Agent node.

    Classifies user intent and normalizes queries for optimal retrieval.

    Args:
        state: Current AletheiaState object

    Returns:
        Updated AletheiaState object with route and reformulated_query set

    Raises:
        AletheiaError: If routing or reformulation fails
    """
    try:
        # Initialize query state if not present
        if state.query is None:
            state.query = QueryState(original_query="")

        # Ensure we have the original query
        if not state.query.original_query:
            raise AletheiaError("Original query is missing or empty")

        original_query = state.query.original_query.strip()
        logger.info(f"Routing query: {original_query[:100]}...")

        # Invoke LLM router
        resp = llm_client.call(
            messages=[{"role": "user", "content": original_query}],
            system_prompt=ROUTER_SYSTEM_PROMPT,
            temperature=0.1,
            json_output=True,
        )

        route = "content"
        reformulated = original_query

        if resp.parsed_json:
            parsed_route = str(resp.parsed_json.get("route", "")).lower().strip()
            if parsed_route in VALID_ROUTES:
                route = parsed_route

            parsed_reformulated = resp.parsed_json.get("reformulated_query")
            if parsed_reformulated and isinstance(parsed_reformulated, str) and parsed_reformulated.strip():
                reformulated = parsed_reformulated.strip()

        # Track pipeline cost
        if resp.cost_usd > 0.0:
            state.pipeline_cost += resp.cost_usd

        state.query.route = route
        state.query.reformulated_query = reformulated

        logger.info(f"Route determined: {route}")
        logger.debug(f"Reformulated query: {reformulated[:100]}...")

        return state

    except Exception as exc:
        logger.error(f"Router node failed: {exc}")
        raise AletheiaError(f"Failed to route query: {exc}") from exc