# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-Level Architecture

Project Aletheia is an autonomous, continuously updating conversational AI system designed to track, ingest, and explain advancements in artificial intelligence. It employs a **Three-Layer Global Memory architecture** managed by a collaborative **multi-agent society** orchestrated by **LangGraph**.

The system is broadly divided into:

1. **Offline Ingestion Sandbox (MemGraphRAG)**: An isolated multi-agent loop responsible for processing raw PDFs, extracting facts, detecting and resolving conflicts, and populating the global memory.
2. **Online Orchestration (LangGraph)**: The core state machine that manages query routing, parallel retrieval from memory, evidence evaluation, external API fallback research, attention-guided context compression, and self-correcting answer generation.
3. **Attention Compressor (AttentionRAG)**: Dynamically prunes assembled context to prevent attention dilution and reduce token overhead by up to 6.3x.
4. **Storage**: A heterogeneous data layer comprising:
    * **Neo4j**: For topological structures (Ontology, Facts, Passages).
    * **FAISS**: For 384-dimensional dense vector embeddings.
    * **Disk-Backed BM25**: For sparse lexical indexing.
5. **External Academic APIs**: Integration with Crossref, OpenAlex, and Semantic Scholar for fallback research when internal evidence is weak.
6. **LLM Engine**: Supports local inference (e.g., Ollama running Qwen) and cloud APIs for various reasoning tasks.
7. **User Interface**: A Gradio web interface for interaction.

## Core Components and Data Flow

* **Query Router & Reformulator Agent**: Classifies user intent and normalizes queries for optimal retrieval.
* **Memory-Guided Retriever**: Fetches structurally sound and semantically relevant data from Neo4j, FAISS, and BM25, utilizing Personalized PageRank (PPR) for graph diffusion and Reciprocal Rank Fusion (RRF) for dual-index merging.
* **Evidence Evaluator & Scorer**: Audits retrieved context using a 100-Point Evidence Sufficiency Rubric (Retrieval Confidence, Answer Specificity, Source Diversity, Metadata Completeness, Recency). Triggers external search if the score is below 80.
* **Async External Search Agent**: Reformulates queries and performs iterative searches across external academic APIs (Crossref, OpenAlex, Semantic Scholar) when internal evidence is insufficient.
* **Attention-Guided Context Compressor**: Uses an "answer hint prefix" to identify a focal token, then calculates attention weights across micro-windows of context to prune irrelevant sentences, achieving significant compression.
* **Generator & Critic**: Synthesizes the final answer, followed by a self-correction loop where a secondary LLM checker audits for hallucinations and missing citations. If issues are detected, a single automatic regeneration is triggered.

## Project Structure

The project adheres to the following Python directory structure:

```
MahoRAGa/
├── core/
│   ├── __init__.py
│   ├── config.py           # Pydantic BaseSettings (API keys, thresholds, paths)
│   ├── state.py            # Pydantic state models (AletheiaState and sub-states)
│   └── exceptions.py       # Custom error hierarchy (AletheiaError base)
├── graph/
│   ├── orchestrator.py     # The compiled LangGraph StateGraph
│   ├── nodes/              # Individual LangGraph node functions
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── retriever.py
│   │   ├── evaluator.py
│   │   ├── search_apis.py
│   │   ├── pruner.py
│   │   └── generator.py
├── memory/
│   ├── __init__.py
│   ├── neo4j_client.py     # Neo4j graph-database client (PPR, serialisation)
│   ├── vector_store.py     # FAISS IndexFlatIP dense-vector store
│   └── sparse_store.py     # Disk-backed BM25 + cross-encoder re-ranking
├── ui/
│   └── app.py              # Gradio interface
├── .env                    # Environment variables (API keys, Neo4j credentials)
└── requirements.txt
```

## Implemented Components

### `core/` — Foundation Layer
- **`config.py`**: Pydantic `BaseSettings` loading from `.env`. Centralises all config: Neo4j connection, FAISS/BM25 index paths, embedding model (`all-MiniLM-L6-v2`), cross-encoder model (`ms-marco-MiniLM-L-6-v2`), Ollama/OpenAI settings, retrieval thresholds, PPR parameters.
- **`state.py`**: Five Pydantic state models (`QueryState`, `MemoryState`, `EvaluationState`, `CompressionState`, `GenerationState`) composed into `AletheiaState` — the master object passed through all LangGraph nodes.
- **`exceptions.py`**: Typed exception hierarchy rooted in `AletheiaError` with specific classes for each layer (Neo4j, FAISS, BM25, retrieval, ingestion, generation).

### `memory/` — Three-Layer Global Memory
- **`neo4j_client.py`**: Thread-safe Neo4j Bolt driver wrapper. CRUD for four node labels (`ONTOLOGY_SCHEMA`, `FACT_TRIPLE`, `PASSAGE_CHUNK`, `DOCUMENT`) with provenance edges. Includes pure-Cypher PPR, graph→NL serialisation for Cross-Encoder input, and hub-suppression penalty helpers for structure-aware node initialisation.
- **`vector_store.py`**: FAISS `IndexFlatIP` over L2-normalised 384-dim embeddings (cosine similarity). Lazy-loaded SentenceTransformer encoder. Supports `add_chunks`, `search`, `search_by_vector`, and full disk serialisation (binary index + JSON metadata sidecar).
- **`sparse_store.py`**: `BM25Okapi` lexical index with pickle-based disk persistence. Built-in Cross-Encoder re-ranking via `ms-marco-MiniLM-L-6-v2`. Static `reciprocal_rank_fusion()` utility for merging FAISS + BM25 result lists.

## Common Development Tasks

### Environment Setup
```bash
pip install -r requirements.txt
# Or install individually:
pip install pydantic pydantic-settings neo4j faiss-cpu rank-bm25 \
    sentence-transformers pymupdf gradio langchain langgraph
```

### Environment Variables
Configure `.env` at the project root:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OPENAI_API_KEY=sk-...       # optional
SEMANTIC_SCHOLAR_API_KEY=... # optional
```

### Running the UI
```bash
python ui/app.py
```

### Local LLM Setup
Install Ollama and pull the default model:
```bash
ollama pull qwen2.5:7b
```

### Testing
```bash
pytest                           # Unit tests
pytest tests/ -k "test_memory"   # Memory layer tests only
```
- **Retrieval Metrics**: Precision@K, Recall@K, MRR against standard test sets.
- **Generation (LLM-as-a-Judge)**: GPT-4o-mini evaluates hallucination and citation adherence.
- **Agent Flow**: Unit test LangGraph nodes individually, mocking API responses.

## Key Design Decisions

- **IndexFlatIP + L2 normalisation** → exact cosine similarity without approximation, suitable for corpora up to ~1M chunks.
- **Pickle persistence for BM25** → `BM25Okapi` is not natively incremental; the full index is rebuilt on each `add_documents` call and the entire object is serialised.
- **Pure-Cypher PPR** → avoids a hard dependency on Neo4j GDS plugin; a 3-hop neighbourhood expansion with teleport-based scoring approximates PPR.
- **Lazy model loading** → SentenceTransformer and CrossEncoder are loaded on first use, not at import time, so modules that don't need them avoid the startup cost.
