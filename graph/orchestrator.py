"""
LangGraph orchestrator for Project Aletheia.

Compiles the StateGraph that manages the flow between all pipeline nodes:
1. Query Router & Reformulator Agent
2. Memory-Guided Retriever
3. Evidence Evaluator & Scorer
4. Async External Search Agent
5. Attention-Guided Context Compressor
6. Generator & Critic
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from typing import Dict, Any

from core.state import AletheiaState
from .nodes import (
    router,
    retriever,
    evaluator,
    search_apis,
    pruner,
    generator
)


def create_aletheia_graph() -> StateGraph:
    """
    Create and configure the LangGraph StateGraph for Project Aletheia.

    Returns:
        StateGraph: Compiled graph ready for invocation
    """
    # Initialize the graph with our state type
    workflow = StateGraph(AletheiaState)

    # Add all the nodes to the graph
    workflow.add_node("router", router.router_node)
    workflow.add_node("retriever", retriever.retriever_node)
    workflow.add_node("evaluator", evaluator.evaluator_node)
    workflow.add_node("search_apis", search_apis.search_apis_node)
    workflow.add_node("pruner", pruner.pruner_node)
    workflow.add_node("generator", generator.generator_node)

    # Define the entry point
    workflow.set_entry_point("router")

    # Define the flow edges
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "evaluator")

    # Conditional edge from evaluator: if evidence is weak, go to search APIs
    workflow.add_conditional_edges(
        "evaluator",
        lambda state: "search_apis" if state.evaluation.is_evidence_weak else "pruner",
        {
            "search_apis": "search_apis",
            "pruner": "pruner"
        }
    )

    # After search APIs, always go to pruner (whether we searched or not)
    workflow.add_edge("search_apis", "pruner")
    workflow.add_edge("pruner", "generator")

    # Generator always ends the flow (for now - could loop back for self-correction)
    workflow.add_edge("generator", END)

    # Compile the graph
    return workflow.compile()


# Create a global instance for easy access
aletheia_graph = create_aletheia_graph()


def run_aletheia_pipeline(initial_state: AletheiaState) -> Dict[str, Any]:
    """
    Convenience function to run the Aletheia pipeline synchronously.

    Args:
        initial_state: The starting state for the pipeline

    Returns:
        Dict containing the final state after pipeline execution
    """
    result = aletheia_graph.invoke(initial_state)
    return result


async def run_aletheia_pipeline_async(initial_state: AletheiaState) -> Dict[str, Any]:
    """
    Asynchronous runner for the Aletheia pipeline.

    Args:
        initial_state: The starting state for the pipeline

    Returns:
        Dict containing the final state after pipeline execution
    """
    result = await aletheia_graph.ainvoke(initial_state)
    return result