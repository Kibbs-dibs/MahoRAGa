# Graph Nodes Completion Design
**Date**: 2026-06-18
**Topic**: Completion of missing graph nodes for Project Aletheia

## Overview

This design specifies the implementation of the three missing LangGraph nodes in the Project Aletheia orchestrator:
1. **search_apis.py** - Async External Search Agent
2. **pruner.py** - Attention-Guided Context Compressor  
3. **generator.py** - Generator & Critic

These nodes will complete the core orchestration pipeline that currently has router, retriever, and evaluator nodes implemented.

## Design Approach

Following the pattern established by existing nodes, all three missing nodes will use **direct state manipulation** (Approach 1) for consistency with the codebase.

## Node Specifications

### 1. Search APIs Node (Async External Search Agent)

**Purpose**: Handles fallback research when internal evidence is weak by querying external academic APIs.

**Responsibilities**:
- Reformulate queries when evidence evaluation score < 80
- Perform iterative optimize-search-vet loops across Crossref, OpenAlex, and Semantic Scholar APIs
- Apply ≥30% term overlap guardrail to prevent semantic drift during query reformulation
- Return vetted external abstracts for context enrichment
- Handle API authentication, rate limiting, and error recovery

**Key Implementation Details**:
- Uses asynchronous HTTP requests for parallel API querying
- Implements exponential backoff and graceful degradation
- Falls back to empty results rather than failing the pipeline
- Updates state.search_apis with raw API results
- Maintains compatibility with existing state structure

### 2. Pruner Node (Attention-Guided Context Compressor)

**Purpose**: Dynamically prunes assembled context to prevent attention dilution and reduce token overhead by up to 6.3x.

**Responsibilities**:
- Create "answer hint prefix" from user query for next-token-prediction task
- Generate focal token using configured LLM
- Process context in micro-windows (50-300 tokens) to calculate attention scores
- Use int4 quantization for efficient attention computation
- Prune sentences with low attention scores to create high-density context
- Maintain semantic integrity while achieving compression

**Key Implementation Details**:
- Segments context into overlapping micro-windows for attention calculation
- Uses local LLM (per settings) for attention weight computation
- Applies entropy-based pruning thresholds
- Preserves discourse coherence during sentence removal
- Updates state.pruner with compressed context and compression metrics

### 3. Generator Node (Generator & Critic)

**Purpose**: Synthesizes final answers and enforces traceability through self-correction.

**Responsibilities**:
- Synthesize answers from pruned context using configured LLM
- Perform self-correction by checking for hallucinations and missing citations
- Trigger automatic regeneration if citation/factuality standards not met (max 1 retry)
- Provide verified answers with source attribution and cost/latency metrics
- Format output for Gradio UI consumption

**Key Implementation Details**:
- Constructs generation prompt with citation instructions
- Implements self-check for hallucinations and uncited claims
- Provides automatic prompt refinement for regeneration attempts
- Formats final answer with inline citations and metadata
- Updates state.generator with answer, citations, confidence scores

## Integration Points

All nodes follow the established data flow:
- **search_apis**: Receives from evaluator (when evidence weak) → sends to pruner
- **pruner**: Receives from evaluator/search_apis → sends to generator  
- **generator**: Receives from pruner → sends to END/UI

Each node:
- Initializes its respective state component if None
- Preserves all existing state while adding outputs
- Follows naming convention: state.node_name for outputs
- Uses consistent error handling with AletheiaError
- Maintains standard logging patterns
- Respects configuration from core.config.settings

## Quality Attributes

**Consistency**: Matches existing node patterns exactly
**Maintainability**: Follows established codebase conventions
**Performance**: Uses efficient algorithms (int4 quantization, batching, caching)
**Reliability**: Graceful degradation, comprehensive error handling
**Testability**: Clear separation of concerns, deterministic behavior where possible

## Files to be Created

- `graph/nodes/search_apis.py` - Async External Search Agent
- `graph/nodes/pruner.py` - Attention-Guided Context Compressor
- `graph/nodes/generator.py` - Generator & Critic

Each file will follow the exact same structure and patterns as the existing nodes in the graph/nodes/ directory.