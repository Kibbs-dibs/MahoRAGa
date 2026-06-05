# MahoRAGa

**Project Aletheia** — an autonomous, continuously updating conversational AI system that tracks, ingests, and explains advancements in artificial intelligence.

Combines **MemGraphRAG** (memory-guided graph construction), **TechGraphRAG** (structure-aware retrieval), and **AttentionRAG** (attention-guided context compression) into a unified Three-Layer Global Memory architecture orchestrated by **LangGraph**.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Gradio UI                                 │
├──────────────────────────────────────────────────────────────────┤
│                   LangGraph Orchestrator                         │
│  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐ │
│  │ Router  │→│ Retriever │→│Evaluator │→│ Pruner │→│Generator│ │
│  └─────────┘ └───────────┘ └──────────┘ └────────┘ └─────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                Three-Layer Global Memory                         │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐ │
│  │ Neo4j Graph   │  │ FAISS Dense    │  │ BM25 Sparse (Disk)  │ │
│  │ (Ontology,    │  │ (384-dim,      │  │ (rank_bm25 +        │ │
│  │  Facts,       │  │  all-MiniLM-   │  │  ms-marco cross-    │ │
│  │  Passages)    │  │  L6-v2)        │  │  encoder reranking) │ │
│  └──────────────┘  └────────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Zero-Hallucination Retrieval** — 100-point evidence sufficiency rubric with automatic external search fallback via Crossref, OpenAlex, and Semantic Scholar.
- **Structure-Aware Graph Retrieval** — Personalized PageRank (PPR) with entity, type, and passage node initialization using semantic similarity and hub suppression.
- **Dual-Index Fusion** — Reciprocal Rank Fusion (RRF) merges dense (FAISS) and sparse (BM25) results, finalized by a Cross-Encoder reranker.
- **Attention-Guided Compression** — Prunes context by up to 6.3x using focal-token attention scoring over micro-windows.
- **Self-Correcting Generation** — LLM Critic audits for hallucinations and missing citations with a single automatic regeneration loop.
- **Local-First Privacy** — Supports Ollama with open-weight models (Qwen) for proprietary data.

## Quick Start

### Prerequisites

- Python 3.10+
- Neo4j 5.x (local or Docker)
- Ollama (optional, for local LLM inference)

### Installation

```bash
git clone https://github.com/Kibbs-dibs/MahoRAGa.git
cd MahoRAGa
pip install -r requirements.txt
```

### Configuration

Create a `.env` file at the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
# Optional cloud keys:
OPENAI_API_KEY=sk-...
SEMANTIC_SCHOLAR_API_KEY=...
```

### Run

```bash
# Start Neo4j (Docker example)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password neo4j:5

# Optional: Pull local LLM
ollama pull qwen2.5:7b

# Launch the Gradio interface
python ui/app.py
```

## Project Structure

```
MahoRAGa/
├── core/
│   ├── config.py           # Pydantic BaseSettings (.env loading)
│   ├── state.py            # Pipeline state models (AletheiaState)
│   └── exceptions.py       # Typed exception hierarchy
├── graph/
│   ├── orchestrator.py     # Compiled LangGraph StateGraph
│   └── nodes/              # LangGraph node functions
│       ├── router.py       # Query classification & reformulation
│       ├── retriever.py    # Parallel memory retrieval + RRF fusion
│       ├── evaluator.py    # 100-point evidence scoring
│       ├── search_apis.py  # Async external academic API search
│       ├── pruner.py       # AttentionRAG context compression
│       └── generator.py    # Answer synthesis + critic loop
├── memory/
│   ├── neo4j_client.py     # Neo4j Bolt client (CRUD, PPR, serialisation)
│   ├── vector_store.py     # FAISS IndexFlatIP (384-dim cosine similarity)
│   └── sparse_store.py     # Disk-backed BM25 + cross-encoder reranking
├── ui/
│   └── app.py              # Gradio web interface
├── .env                    # Environment variables
└── requirements.txt        # Python dependencies
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| State Management | Pydantic |
| Knowledge Graph | Neo4j |
| Dense Vectors | FAISS (IndexFlatIP) |
| Sparse Index | rank_bm25 (BM25Okapi) |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) |
| Re-ranking | `ms-marco-MiniLM-L-6-v2` |
| PDF Parsing | PyMuPDF (fitz) |
| Local LLM | Ollama (Qwen) |
| UI | Gradio |

## Testing

```bash
pytest                            # All tests
pytest tests/ -k "test_memory"    # Memory layer
pytest tests/ -k "test_retrieval" # Retrieval pipeline
```

## License

See [LICENSE](LICENSE) for details.
