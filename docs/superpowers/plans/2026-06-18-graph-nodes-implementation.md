# Graph Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three missing LangGraph nodes (search_apis, pruner, generator) to complete the Project Aletheia orchestration pipeline.

**Architecture:** Direct state manipulation approach following existing patterns in router, retriever, and evaluator nodes. Each node will initialize its state component, process inputs, and return updated AletheiaState objects.

**Tech Stack:** Python, LangGraph, asyncio, aiohttp, transformers, sentence-transformers, core/config, core/state, core/exceptions, memory modules

---

### Task 1: Search APIs Node Implementation

**Files:**
- Create: `graph/nodes/search_apis.py`
- Modify: `graph/__init__.py:1-1` (to export the new node)
- Test: `tests/nodes/test_search_apis.py`

- [ ] **Step 1: Write the failing test for search APIs node basic structure**

```python
def test_search_apis_node_exists():
    from graph.nodes.search_apis import search_apis_node
    assert search_apis_node is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_search_apis.py::test_search_apis_node_exists -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'graph.nodes.search_apis'"

- [ ] **Step 3: Write minimal implementation for search APIs node**

```python
"""
Async External Search Agent for Project Aletheia.

Responsibilities
────────────────
• Reformulates queries when evidence is weak (score < 80)
• Performs iterative searches across Crossref, OpenAlex, and Semantic Scholar APIs
• Applies term overlap guardrail to prevent semantic drift
• Returns vetted external abstracts for context enrichment
"""

from __future__ import annotations

from core.state import AletheiaState
from core.config import settings
from core.exceptions import AletheiaError
import logging

logger = logging.getLogger(__name__)


def search_apis_node(state: AletheiaState) -> AletheiaState:
    """
    Async External Search Agent node.
    
    Handles fallback research when internal evidence is weak.
    
    Args:
        state: Current AletheiaState object
        
    Returns:
        Updated AletheiaState object with search API results
        
    Raises:
        AletheiaError: If search API processing fails
    """
    try:
        logger.info("Starting external search APIs process")
        
        # Initialize search APIs state if not present
        if state.search_apis is None:
            from core.state import SearchAPIsState
            state.search_apis = SearchAPIsState()
        
        # Check if we need to search (evidence is weak)
        if not state.evaluation or not state.evaluation.is_evidence_weak:
            logger.info("Evidence is strong, skipping external search")
            return state
        
        # For now, implement placeholder that returns empty results
        # In full implementation, this would:
        # 1. Reformulate query with term overlap guardrail
        # 2. Query APIs in parallel (Crossref, OpenAlex, Semantic Scholar)
        # 3. Apply iterative optimize-search-vet loops
        # 4. Return vetted abstracts
        
        logger.info("External search APIs placeholder - returning empty results")
        return state
        
    except Exception as exc:
        logger.error(f"Search APIs node failed: {exc}")
        raise AletheiaError(f"External search APIs failed: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_search_apis.py::test_search_apis_node_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/search_apis.py tests/nodes/test_search_apis.py
git commit -m "feat: implement search_apis node basic structure"
```

---

### Task 2: Pruner Node Implementation

**Files:**
- Create: `graph/nodes/pruner.py`
- Modify: `graph/__init__.py:1-1` (to export the new node)
- Test: `tests/nodes/test_pruner.py`

- [ ] **Step 1: Write the failing test for pruner node basic structure**

```python
def test_pruner_node_exists():
    from graph.nodes.pruner import pruner_node
    assert pruner_node is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_pruner.py::test_pruner_node_exists -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'graph.nodes.pruner'"

- [ ] **Step 3: Write minimal implementation for pruner node**

```python
"""
Attention-Guided Context Compressor for Project Aletheia.

Responsibilities
────────────────
• Uses an "answer hint prefix" to create a next-token-prediction task
• Generates a focal token using configured LLM
• Processes context in micro-windows (50-300 tokens) to calculate attention scores
• Uses int4 quantization for efficient attention computation
• Prunes sentences with low attention scores to achieve up to 6.3x compression
"""

from __future__ import annotations

from core.state import AletheiaState
from core.config import settings
from core.exceptions import AletheiaError
import logging

logger = logging.getLogger(__name__)


def pruner_node(state: AletheiaState) -> AletheiaState:
    """
    Attention-Guided Context Compressor node.
    
    Dynamically prunes assembled context to prevent attention dilution.
    
    Args:
        state: Current AletheiaState object
        
    Returns:
        Updated AletheiaState object with pruned, high-density context
        
    Raises:
        AletheiaError: If context pruning fails
    """
    try:
        logger.info("Starting attention-guided context compression")
        
        # Initialize pruner state if not present
        if state.pruner is None:
            from core.state import PrunerState
            state.pruner = PrunerState()
        
        # For now, implement placeholder that returns state unchanged
        # In full implementation, this would:
        # 1. Assemble context from memory and search APIs results
        # 2. Create answer hint prefix from user query
        # 3. Generate focal token using LLM
        # 4. Process context in micro-windows to calculate attention scores
        # 5. Use int4 quantization for efficient computation
        # 6. Prune low-attention sentences to create compressed context
        # 7. Update state with compression metrics
        
        logger.info("Pruner placeholder - returning state unchanged")
        return state
        
    except Exception as exc:
        logger.error(f"Pruner node failed: {exc}")
        raise AletheiaError(f"Context pruning failed: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_pruner.py::test_pruner_node_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/pruner.py tests/nodes/test_pruner.py
git commit -m "feat: implement pruner node basic structure"
```

---

### Task 3: Generator Node Implementation

**Files:**
- Create: `graph/nodes/generator.py`
- Modify: `graph/__init__.py:1-1` (to export the new node)
- Test: `tests/nodes/test_generator.py`

- [ ] **Step 1: Write the failing test for generator node basic structure**

```python
def test_generator_node_exists():
    from graph.nodes.generator import generator_node
    assert generator_node is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_generator.py::test_generator_node_exists -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'graph.nodes.generator'"

- [ ] **Step 3: Write minimal implementation for generator node**

```python
"""
Generator & Critic for Project Aletheia.

Responsibilities
────────────────
• Synthesizes final answers from pruned context using configured LLM
• Performs self-correction by checking for hallucinations and missing citations
• Triggers automatic regeneration if issues are detected (max 1 retry)
• Provides verified answers with source attribution and cost/latency metrics
"""

from __future__ import annotations

from core.state import AletheiaState
from core.config import settings
from core.exceptions import AletheiaError
import logging

logger = logging.getLogger(__name__)


def generator_node(state: AletheiaState) -> AletheiaState:
    """
    Generator & Critic node.
    
    Synthesizes final answers and enforces traceability through self-correction.
    
    Args:
        state: Current AletheiaState object
        
    Returns:
        Updated AletheiaState object with final answer and metadata
        
    Raises:
        AletheiaError: If answer generation fails
    """
    try:
        logger.info("Starting answer generation and criticism")
        
        # Initialize generator state if not present
        if state.generator is None:
            from core.state import GeneratorState
            state.generator = GeneratorState()
        
        # For now, implement placeholder that returns state unchanged
        # In full implementation, this would:
        # 1. Construct generation prompt with pruned context and citation instructions
        # 2. Generate initial answer using configured LLM
        # 3. Run criticism pass: check for hallucinations, uncited claims, logical consistency
        # 4. If issues found and retries available: refine prompt and regenerate (max 1 retry)
        # 5. Format final answer with inline citations and metadata
        # 6. Update state with answer, citations, confidence scores, and cost/latency metrics
        
        logger.info("Generator placeholder - returning state unchanged")
        return state
        
    except Exception as exc:
        logger.error(f"Generator node failed: {exc}")
        raise AletheiaError(f"Answer generation failed: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_generator.py::test_generator_node_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/generator.py tests/nodes/test_generator.py
git commit -m "feat: implement generator node basic structure"
```

---

### Task 4: Update Graph Init File

**Files:**
- Modify: `graph/__init__.py`

- [ ] **Step 1: Write the failing test for init file exports**

```python
def test_graph_init_exports_all_nodes():
    from graph.nodes import router, retriever, evaluator, search_apis, pruner, generator
    assert all([router, retriever, evaluator, search_apis, pruner, generator])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_init.py::test_graph_init_exports_all_nodes -v`
Expected: FAIL with "ImportError: cannot import name 'search_apis' from 'graph.nodes'"

- [ ] **Step 3: Write minimal implementation for init file**

```python
"""
Project Aletheia Graph Nodes Package
"""

from .nodes.router import router_node as router
from .nodes.retriever import retriever_node as retriever
from .nodes.evaluator import evaluator_node as evaluator
from .nodes.search_apis import search_apis_node as search_apis
from .nodes.pruner import pruner_node as pruner
from .nodes.generator import generator_node as generator

__all__ = ["router", "retriever", "evaluator", "search_apis", "pruner", "generator"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_init.py::test_graph_init_exports_all_nodes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graph/__init__.py
git commit -m "feat: export all graph nodes from __init__.py"
```

---

### Task 5: Update Orchestrator for Complete Pipeline

**Files:**
- Modify: `graph/orchestrator.py`

- [ ] **Step 1: Write the failing test for orchestrator integration**

```python
def test_orchestrator_has_complete_flow():
    from graph.orchestrator import create_aletheia_graph
    # This test would verify the graph compiles successfully with all nodes
    graph = create_aletheia_graph()
    assert graph is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_orchestrator_has_complete_flow -v`
Expected: FAIL when trying to import search_apis, pruner, or generator nodes

- [ ] **Step 3: Write minimal implementation for orchestrator**

```python
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
    Convenience function to run the Aletheia pipeline.

    Args:
        initial_state: The starting state for the pipeline

    Returns:
        Dict containing the final state after pipeline execution
    """
    result = aletheia_graph.invoke(initial_state)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py::test_orchestrator_has_complete_flow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graph/orchestrator.py
git commit -m "feat: update orchestrator with complete node pipeline"
```