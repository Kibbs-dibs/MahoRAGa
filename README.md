# Project Aletheia (MahoRAGa)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-52%2F52%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Gradio UI](https://img.shields.io/badge/UI-CodeRabbit%20Glassmorphic-orange.svg)](ui/app.py)
[![FastAPI Backend](https://img.shields.io/badge/API-FastAPI%20REST-009688.svg)](backend/app.py)

**Project Aletheia** is an enterprise-grade, autonomous multi-agent research platform for artificial intelligence literature. It integrates:
- **Three-Layer Global Memory**: Neo4j Hierarchical Indexing (Ontology, Fact Triples, Passages), FAISS Dense Vector Store (384-dim IndexFlatIP), and Disk-Backed BM25 Sparse Lexical Store.
- **AttentionRAG Context Compression**: Micro-window attention pruning reducing prompt length up to 6.3x while maintaining strict citation fidelity.
- **The Critic QA Audit**: Self-correcting hallucination prevention with single-turn regeneration bounds.
- **Multi-Agent LangGraph Pipeline**: Real-time streaming progression across 6 nodes.
- **CodeRabbit-Styled Web Console & Vis.js Graph**: Interactive Gradio 6 web application with collapsible Ingestion Dock, interactive force-directed graph visualizer, and provenance inspector.
- **FastAPI REST Backend & Session Persistence**: SQLite-backed conversation sessions and query audits.
- **Autonomous ArXiv Daily Watcher**: Scheduled and on-demand discovery, paper downloading, and auto-indexing.
- **Ragas-Style Quantitative Benchmark**: Mathematical evaluation (Faithfulness, Answer Relevance, Context Recall, Compression Fidelity) and pipeline observability tracing.

---

## Architecture & Pipeline Flow

```
                      ┌────────────────────────────────────────┐
                      │    Gradio UI & FastAPI REST Backend    │
                      │               (start.py)               │
                      └───────────────────┬────────────────────┘
                                          │
                         [Real-Time LangGraph Stepper]
                                          │
    ┌──────────┐    ┌───────────┐    ┌────┴─────┐    ┌─────────────┐    ┌──────────┐    ┌─────────────┐
    │  Router  ├───►│ Retriever ├───►│Evaluator ├───►│ Search APIs ├───►│  Pruner  ├───►│ Generator   │
    │  (Agent) │    │ (RRF Dual)│    │ (Rubric) │    │ (Fallback)  │    │(AttnRAG) │    │  & Critic   │
    └──────────┘    └─────┬─────┘    └──────────┘    └─────────────┘    └──────────┘    └──────┬──────┘
                          │                                                                    │
              ┌───────────┴──────────────────────────────┐                       [Inline Citations &]
              │            Three-Layer Global Memory     │                       [The Critic Audit  ]
              ├──────────────────────────────────────────┤                                     ▼
              │ • Neo4j Graph  (Ontology, Facts, Passages)│                               Answer Output
              │ • FAISS Dense  (IndexFlatIP, 384-dim)    │
              │ • BM25 Sparse  (BM25Okapi + Cross-Encoder)│
              └──────────────────────────────────────────┘
```

---

## Key Features & Production Milestones

1. **Live In-Browser PDF Ingestion Dock (Milestone 1)**:
   - Drag-and-drop multi-file PDF uploader in Gradio with real-time progress indicators.
   - MD5 deduplication prevents redundant tokenization and embedding recomputation.
   - Incremental FAISS vector insertion and BM25 index rebuilding with dynamic UI catalog updates.

2. **Interactive Force-Directed Knowledge Graph Visualizer (Milestone 2)**:
   - Dynamic Vis.js network visualizer with physics simulation and category color palettes (`#FF570A` Methodology, `#00F2A5` Metric, `#3B82F6` Concept).
   - "Freeze / Float" physics controls and isolated HTML rendering.

3. **FastAPI Production REST API & Session Persistence (Milestone 3)**:
   - SQLite database (`data/aletheia.db`) in WAL mode for persistent multi-turn chat sessions and query audit logs.
   - Comprehensive REST endpoints (`/health`, `/api/v1/query`, `/api/v1/sessions`, `/api/v1/ingest`, `/api/v1/corpus`, `/api/v1/graph`) with interactive Swagger docs at `/docs`.

4. **Autonomous ArXiv Daily Watcher Agent (Milestone 4)**:
   - Queries the official arXiv Atom REST API for AI categories (`cs.AI`, `cs.CL`, `cs.LG`).
   - Deduplicates incoming papers against existing catalog titles, downloads PDFs, and auto-ingests them into memory.

5. **Ragas-Style Quantitative Evaluation & Observability Tracing (Milestone 5)**:
   - Pipeline tracing measuring latency percentiles ($p50$, $p95$), rubric score trends, and compression ratios.
   - Mathematical metric calculators for Faithfulness, Answer Relevance (dense cosine similarity), Context Recall, and Compression Fidelity.
   - CLI benchmark runner (`python -m evaluation.benchmark`) exporting Markdown scorecards and JSON artifacts.

---

## Quick Start

### 1. Launch Unified Server (Web Console + FastAPI REST Endpoints)

Run the multi-service launcher script:

```powershell
python start.py
```

The unified server launches:
- **Web Console**: `http://localhost:7860`
- **FastAPI OpenAPI Docs**: `http://localhost:7860/docs`
- **Health Diagnostics**: `http://localhost:7860/health`
- **Observability Metrics**: `http://localhost:7860/api/v1/metrics`

### 2. Autonomous ArXiv Sync

Trigger via REST API or from the Web Console Ingestion Dock:

```bash
curl -X POST "http://localhost:7860/api/v1/watcher/sync?query=Retrieval-Augmented+Generation&max_papers=3"
```

### 3. Run Ragas Benchmark Suite

```bash
python -m evaluation.benchmark
```

---

## Testing & Quality Assurance

Project Aletheia maintains a **100% test pass rate** with 52 automated tests covering all modules and milestones:

```bash
# Run complete test suite (52/52 passing)
pytest -v

# Run specific milestone suites
pytest tests/test_live_ingestion.py -v          # Milestone 1
pytest tests/test_graph_visualizer.py -v         # Milestone 2
pytest tests/test_fastapi_backend.py -v          # Milestone 3
pytest tests/test_arxiv_watcher.py -v            # Milestone 4
pytest tests/test_evaluation_and_tracing.py -v   # Milestone 5
```

---

## Project Structure

```
MahoRAGa/
├── agents/
│   └── arxiv_watcher.py    # Autonomous ArXiv discovery & auto-ingestion agent
├── backend/
│   ├── app.py              # FastAPI production REST server & endpoint routing
│   ├── database.py         # Thread-safe SQLite session manager (WAL mode)
│   └── schemas.py          # Pydantic request/response validation models
├── core/
│   ├── config.py           # Pydantic BaseSettings (.env loading & defaults)
│   ├── state.py            # LangGraph Master State (AletheiaState)
│   ├── llm.py              # Multi-backend LLM client (Ollama, OpenAI, Proxy, Fallback)
│   ├── tracing.py          # Pipeline observability telemetry & latency percentiles
│   └── exceptions.py       # Domain-specific exception hierarchy
├── evaluation/
│   ├── dataset.py          # Curated academic evaluation benchmark dataset
│   ├── metrics.py          # Ragas mathematical metric formulas (Faithfulness, Recall, etc.)
│   └── benchmark.py        # Automated benchmark runner & markdown scorecard generator
├── graph/
│   ├── orchestrator.py     # Compiled LangGraph StateGraph (sync & async runners)
│   └── nodes/
│       ├── router.py       # Query intent classification & normalization
│       ├── retriever.py    # Hybrid RRF dual-index retrieval (FAISS + BM25 + Neo4j)
│       ├── evaluator.py    # 100-point evidence sufficiency rubric
│       ├── search_apis.py  # Async academic literature search (Semantic Scholar, arXiv)
│       ├── pruner.py       # AttentionRAG context compression (up to 6.3x)
│       └── generator.py    # Answer synthesis + The Critic audit loop
├── memory/
│   ├── neo4j_client.py     # Neo4j Bolt client (CRUD, PPR traversal, serialisation)
│   ├── vector_store.py     # FAISS IndexFlatIP (384-dim normalized inner-product)
│   ├── sparse_store.py     # Disk-backed BM25Okapi + Cross-Encoder re-ranker
│   └── utils.py            # Sidecar JSON metadata serialization helpers
├── ingestion/
│   ├── parser.py           # PyMuPDF academic layout extractor
│   ├── chunker.py          # Structure-aware 3,500-char boundary-preserving chunker
│   ├── extractor.py        # Ontology schema and fact triple extraction
│   ├── service.py          # Live multi-threaded PDF ingestion & deduplication engine
│   └── cli.py              # Batch ingestion pipeline runner
├── ui/
│   ├── app.py              # CodeRabbit-styled Gradio 6 glassmorphic research console
│   └── graph_visualizer.py # Interactive Vis.js physics-simulated network generator
├── tests/                  # 52 automated tests (100% pass rate)
├── RAGxamples/             # Academic literature PDFs (AttentionRAG, MemGraphRAG, TechGraphRAG)
├── data/                   # Dynamic runtime storage (catalog, uploads, indices, sqlite db)
├── Dockerfile              # Production container definition
├── docker-compose.yml      # Full-stack Docker deployment
├── start.py                # Unified FastAPI + Gradio server runner
├── start.ps1               # One-click Windows PowerShell startup script
└── requirements.txt        # Python dependency manifest
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
