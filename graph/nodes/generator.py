"""
Generator & Critic for Project Aletheia.

Responsibilities
────────────────
• Synthesizes the final answer from pruned context using the configured LLM.
• Enforces bracketed source citations on all technical claims.
• Audits output using The Critic (checking for hallucinations and uncited claims).
• Executes up to 1 automatic regeneration loop if issues are detected.
• Directly updates GenerationState within AletheiaState and tracks pipeline cost.

This node implements direct state manipulation as specified in Approach 1.
"""

from __future__ import annotations

import logging
import re
import time
import hashlib
from collections import OrderedDict
from typing import Optional

from core.exceptions import AletheiaError
from core.llm import llm_client
from core.state import AletheiaState, GenerationState

logger = logging.getLogger(__name__)

# ── Answer cache (Feature 4.4) ──────────────────────────────────────────
# Bounded LRU cache keyed on (normalized query + pruned-context hash). The
# pruned context already encodes the corpus state that produced it, so a changed
# corpus -> different context -> different key, which keeps the cache correct
# without an explicit corpus version. Cuts repeat LLM cost + latency.
_ANSWER_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_ANSWER_CACHE_MAX = 256


def _answer_cache_key(query: str, pruned_context: str) -> str:
    norm = " ".join((query or "").lower().split())
    h = hashlib.sha256((norm + "\x00" + (pruned_context or "")).encode("utf-8")).hexdigest()
    return h


def _answer_cache_get(key: str) -> Optional[dict]:
    hit = _ANSWER_CACHE.get(key)
    if hit is not None:
        _ANSWER_CACHE.move_to_end(key)  # LRU touch
    return hit


def _answer_cache_put(key: str, value: dict) -> None:
    _ANSWER_CACHE[key] = value
    _ANSWER_CACHE.move_to_end(key)
    while len(_ANSWER_CACHE) > _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.popitem(last=False)

GENERATOR_SYSTEM_PROMPT = """You are the primary technical synthesis agent for Project Aletheia, an AI research assistant.
Synthesize an authoritative, thorough answer to the user's query based STRICTLY on the provided <PRUNED_CONTEXT>.

CITATION REQUIREMENTS:
- The context lines are prefixed with source tags like [S1], [S2]. Every technical fact, metric, architecture mechanism, or methodology MUST end with the bracketed source tag(s) it came from (e.g., [S3]), so citations link back to the exact source.
- Do NOT hallucinate claims not present in the context. If details are missing, state what is known from the context.
"""

CRITIC_SYSTEM_PROMPT = """You are the final Quality Assurance Critic for a zero-hallucination research pipeline.
Your job is to audit a drafted answer against the pruned context it was generated from.

You will be provided with:
1. <PRUNED_CONTEXT> (The ground-truth facts)
2. <DRAFT_ANSWER> (The LLM-generated response)

AUDIT RULES:
1. Citation Check: Every technical claim, metric, or architectural description in the draft MUST end with a bracketed citation matching a source in the context (e.g., [Smith et al., 2024]).
2. Hallucination Check: The draft cannot contain ANY technical information, frameworks, or metrics that are not explicitly stated in the <PRUNED_CONTEXT>.

EVALUATION:
If the draft passes both checks, set "passed" to true.
If it fails, set "passed" to false, and provide specific "feedback" detailing exactly which claim is uncited or hallucinated so the Generator can fix it on the next iteration.
Also provide a "confidence" score in [0,1] (how confident you are the answer is fully grounded) and a "spans" array listing the exact offending claim substrings (empty when passed).

OUTPUT FORMAT:
Respond with a raw JSON object:
{
  "passed": boolean,
  "feedback": "string or null",
  "confidence": number,
  "spans": ["offending claim text", ...]
}
"""


def _run_critic_audit(pruned_context: str, draft_answer: str) -> tuple[bool, Optional[str], float, float, list]:
    """
    Run The Critic audit on the drafted answer.
    Returns (passed, feedback, cost_usd).
    """
    prompt = (
        f"<PRUNED_CONTEXT>\n{pruned_context}\n</PRUNED_CONTEXT>\n\n"
        f"<DRAFT_ANSWER>\n{draft_answer}\n</DRAFT_ANSWER>"
    )

    resp = llm_client.call(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=CRITIC_SYSTEM_PROMPT,
        temperature=0.1,
        json_output=True,
    )

    passed = True
    feedback = None
    confidence = 1.0
    spans: list = []

    if resp.parsed_json:
        passed = bool(resp.parsed_json.get("passed", True))
        feedback = resp.parsed_json.get("feedback")
        try:
            confidence = float(resp.parsed_json.get("confidence", 1.0 if passed else 0.3))
        except (TypeError, ValueError):
            confidence = 1.0 if passed else 0.3
        confidence = max(0.0, min(1.0, confidence))
        raw_spans = resp.parsed_json.get("spans") or []
        spans = [str(s) for s in raw_spans if str(s).strip()] if isinstance(raw_spans, list) else []
    else:
        # Heuristic fallback if JSON parsing failed
        has_citations = bool(re.search(r"\[[^\]]+\]", draft_answer))
        if not has_citations:
            passed = False
            feedback = "Missing bracketed citations for technical claims."
            confidence = 0.3

    return passed, feedback, resp.cost_usd, confidence, spans


def generator_node(state: AletheiaState) -> AletheiaState:
    """
    Generator & Critic node for final answer synthesis and self-correction.

    Synthesizes the final answer from pruned context using an LLM,
    then performs self-correction by checking for hallucinations and missing citations.
    If issues are detected, triggers a single automatic regeneration.

    Args:
        state: Current AletheiaState object

    Returns:
        Updated AletheiaState object with generation state populated

    Raises:
        AletheiaError: If generation or self-correction fails
    """
    try:
        logger.info("Generating final answer from pruned context...")
        start_time = time.time()

        # Initialize generation state if not present
        if state.generation is None:
            state.generation = GenerationState()

        # Get pruned context
        if state.compression is None or not state.compression.final_pruned_prompt:
            raise AletheiaError("No pruned context available for generation")

        pruned_context = state.compression.final_pruned_prompt.strip()
        if not pruned_context:
            raise AletheiaError("No pruned context available for generation")

        original_query = state.query.original_query if state.query else ""
        reformulated_query = (
            state.query.reformulated_query or original_query if state.query else ""
        )

        # ── 0. Answer cache lookup (Feature 4.4) ──────────────────────
        _hist = getattr(state.query, "conversation_history", None) or [] if state.query else []
        _key_query = (reformulated_query or original_query) + " || " + " | ".join(_hist)
        cache_key = _answer_cache_key(_key_query, pruned_context)
        cached = _answer_cache_get(cache_key)
        if cached is not None:
            state.generation.draft_answer = cached["answer"]
            state.generation.critic_feedback = cached.get("critic_feedback")
            logger.info("Answer cache HIT — served without LLM calls (%.0f%% full)",
                        100.0 * len(_ANSWER_CACHE) / _ANSWER_CACHE_MAX)
            return state

        # ── 1. Synthesize Draft Answer ────────────────────────────────
        history = getattr(state.query, "conversation_history", None) or [] if state.query else []
        history_block = ("\n<CONVERSATION_HISTORY>\n" + "\n".join(history) + "\n</CONVERSATION_HISTORY>\n") if history else ""
        gen_prompt = (
            f"<USER_QUERY>\n{reformulated_query or original_query}\n</USER_QUERY>\n"
            f"{history_block}\n"
            f"<PRUNED_CONTEXT>\n{pruned_context}\n</PRUNED_CONTEXT>"
        )

        resp = llm_client.call(
            messages=[{"role": "user", "content": gen_prompt}],
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=1500,
        )

        draft_answer = resp.content.strip()
        state.generation.draft_answer = draft_answer
        state.pipeline_cost += resp.cost_usd

        # ── 2. Run The Critic Audit ───────────────────────────────────
        passed, critic_feedback, critic_cost, critic_confidence, critic_spans = _run_critic_audit(
            pruned_context, draft_answer
        )
        state.pipeline_cost += critic_cost
        # Persist critic confidence + offending spans (Feature 4.3); overwritten
        # below with the re-audit values if regeneration occurs.
        state.generation.critic_confidence = critic_confidence
        state.generation.critic_spans = critic_spans

        # ── 3. Self-Correction Loop (Max 1 Retry) ──────────────────────
        if not passed and state.generation.retry_count < 1:
            logger.info(
                "Critic feedback: %s. Triggering automatic regeneration (attempt 1)",
                critic_feedback,
            )
            state.generation.critic_feedback = critic_feedback
            state.generation.retry_count = 1

            retry_prompt = (
                f"<USER_QUERY>\n{reformulated_query or original_query}\n</USER_QUERY>\n\n"
                f"<PRUNED_CONTEXT>\n{pruned_context}\n</PRUNED_CONTEXT>\n\n"
                f"PREVIOUS DRAFT:\n{draft_answer}\n\n"
                f"CRITIC AUDIT FEEDBACK (FIX THESE ISSUES STRICTLY):\n{critic_feedback}\n\n"
                f"Please regenerate the complete answer addressing the feedback and providing bracketed citations."
            )

            retry_resp = llm_client.call(
                messages=[{"role": "user", "content": retry_prompt}],
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=1500,
            )

            state.generation.draft_answer = retry_resp.content.strip()
            state.pipeline_cost += retry_resp.cost_usd
            logger.info("Automatic regeneration completed successfully.")

            # Re-audit the regenerated draft so the FINAL answer is verified
            # (Issue 8). The single-retry cap above already prevents loops.
            recheck_passed, recheck_feedback, recheck_cost, critic_confidence, critic_spans = _run_critic_audit(
                pruned_context, state.generation.draft_answer
            )
            state.pipeline_cost += recheck_cost
            # Refresh feedback to describe the returned draft, not the discarded one (Issue 9).
            state.generation.critic_feedback = (
                None if recheck_passed else recheck_feedback
            )
            state.generation.critic_confidence = critic_confidence
            state.generation.critic_spans = critic_spans
            logger.info(
                "Post-regeneration critic verdict: %s",
                "passed" if recheck_passed else "still failing",
            )
        else:
            state.generation.critic_feedback = critic_feedback

        latency = time.time() - start_time
        logger.info(
            "Answer generation finished in %.2fs. Total pipeline cost: $%.6f",
            latency,
            state.pipeline_cost,
        )

        # Populate the answer cache for future identical (query, context) pairs.
        _answer_cache_put(cache_key, {
            "answer": state.generation.draft_answer,
            "critic_feedback": state.generation.critic_feedback,
        })

        return state

    except Exception as exc:
        logger.error(f"Generator node failed: {exc}")
        raise AletheiaError(f"Failed to generate answer: {exc}") from exc