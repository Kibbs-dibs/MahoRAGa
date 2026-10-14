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
from typing import Optional

from core.exceptions import AletheiaError
from core.llm import llm_client
from core.state import AletheiaState, GenerationState

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """You are the primary technical synthesis agent for Project Aletheia, an AI research assistant.
Synthesize an authoritative, thorough answer to the user's query based STRICTLY on the provided <PRUNED_CONTEXT>.

CITATION REQUIREMENTS:
- Every technical fact, metric, architecture mechanism, or methodology MUST end with a bracketed citation matching the source in context (e.g., [Vaswani et al., 2017], [Ref-1, 2024], or [Source ID]).
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

OUTPUT FORMAT:
Respond with a raw JSON object:
{
  "passed": boolean,
  "feedback": "string or null"
}
"""


def _run_critic_audit(pruned_context: str, draft_answer: str) -> tuple[bool, Optional[str], float]:
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

    if resp.parsed_json:
        passed = bool(resp.parsed_json.get("passed", True))
        feedback = resp.parsed_json.get("feedback")
    else:
        # Heuristic fallback if JSON parsing failed
        has_citations = bool(re.search(r"\[[^\]]+\]", draft_answer))
        if not has_citations:
            passed = False
            feedback = "Missing bracketed citations for technical claims."

    return passed, feedback, resp.cost_usd


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

        # ── 1. Synthesize Draft Answer ────────────────────────────────
        gen_prompt = (
            f"<USER_QUERY>\n{reformulated_query or original_query}\n</USER_QUERY>\n\n"
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
        passed, critic_feedback, critic_cost = _run_critic_audit(
            pruned_context, draft_answer
        )
        state.pipeline_cost += critic_cost

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
        else:
            state.generation.critic_feedback = critic_feedback

        latency = time.time() - start_time
        logger.info(
            "Answer generation finished in %.2fs. Total pipeline cost: $%.6f",
            latency,
            state.pipeline_cost,
        )

        return state

    except Exception as exc:
        logger.error(f"Generator node failed: {exc}")
        raise AletheiaError(f"Failed to generate answer: {exc}") from exc