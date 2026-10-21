# Project Aletheia — Implementation Status & Gap Analysis

> **Purpose.** This document maps each major claim in the PRD (`Project_Aletheia_Final_PRD.md`),
> System Design (`SystemDesign.md`), and `README.md` to its *verified* status in the current
> codebase. It is intended as an honest engineering reference so that contributors, reviewers,
> and users understand which behaviours are real, which are heuristic approximations, and which
> are stubs / not implemented.
>
> **Method.** Findings below were produced by reading the actual source and cross-checking with
> repository-wide searches (e.g. `grep` for `torch` / `output_attentions` / `quantize`, and
> call-site searches for `run_personalised_pagerank`, `reciprocal_rank_fusion`, `.rerank(`).
> Every row cites the file that backs the claim. No code was changed to produce this report.
>
> **Legend**
> - ✅ **Real** — genuinely implemented as designed.
> - ⚠️ **Heuristic** — runs and produces output, but via a simplified approximation rather than
>   the sophisticated algorithm described in the design.
> - ❌ **Stub / Missing** — not implemented, dead code (defined but never called), or canned output.

---

## 1. Executive summary

The **infrastructure and integrations are genuinely built**: a real LangGraph state machine, real
FAISS and BM25 stores, a real Neo4j driver, real HTTP clients for LLM backends and the arXiv API,
and a working end-to-end ingestion → query flow. The system runs.

However, **the flagship "intelligent" algorithms that constitute the project's research novelty are
heuristic approximations or dead code**, and several headline features do not exist in the running
pipeline at all. Most importantly:

1. **"AttentionRAG" context compression uses no attention and no model.** There is no transformer,
   no `output_attentions`, and no int4 quantization anywhere in the source. Relevance is scored by
   lexical token overlap, and the advertised **"up to 6.3×" compression is an imposed ceiling, not a
   measured result.**
2. **Hybrid fused retrieval is not fused.** Reciprocal Rank Fusion, cross-encoder reranking, and
   Personalized PageRank are all implemented as functions but have **zero call sites** — the
   retriever concatenates the three stores' results independently.
3. **The offline LLM fallback fabricates citations** (`[Ref-1, 2024]`, `[Synthesized Evidence, 2024]`)
   and its "critic" only checks that brackets exist — which the fallback always adds. So with no LLM
   backend connected, the pipeline self-certifies invented citations.

The project is best characterised as a **faithful, runnable architectural skeleton / demo** of the
PRD, not an implementation of the paper-grade algorithms the README advertises.

---

## 2. Claim-by-claim status

### 2.1 Online orchestration — the 6-node LangGraph pipeline

| # | Design claim | Status | Evidence & notes |
|---|---|---|---|
| 1 | LangGraph `StateGraph` orchestrating 6 nodes with a conditional evidence branch | ✅ Real | `graph/orchestrator.py` builds a real `StateGraph`, wires all 6 nodes, and adds a conditional edge evaluator → `search_apis`/`pruner` keyed on `is_evidence_weak`. |
| 2 | **Router**: LLM classifies route (content/bibliometric/trend/current_world) + rewrites query | ✅ Real | `graph/nodes/router.py` makes a real `llm_client.call(..., json_output=True)`, validates the route, and tracks cost. (Degrades to heuristic only through the LLM offline fallback — see §2.4.) |
| 3 | **Retriever**: hybrid retrieval via RRF + PPR + cross-encoder rerank | ⚠️/❌ | `graph/nodes/retriever.py` queries each store and writes three **separate** state fields. **No RRF, no PPR, no rerank is called.** The Neo4j helper claims "using PPR" but calls `serialise_subgraph_to_sentences` (plain lookups). `_extract_entities_from_query` is a regex that the code itself labels *"a placeholder — real implementation would use NER."* Each layer returns `[]` on any failure. |
| 4 | **Evaluator**: 100-point evidence sufficiency rubric across 5 dimensions | ⚠️ Heuristic | `graph/nodes/evaluator.py` — sub-scores sum to 100 and the `< 80` threshold logic is real, but each dimension is a counting heuristic: confidence = #layers that returned results + raw count; specificity = Jaccard overlap; diversity = #layers + distinct lengths; metadata = result count (comment admits a "full implementation would check actual metadata fields"); recency = substring search for literals like `'2023'..'2026'`, `'latest'`, `'novel'`. |
| 5 | **Search APIs**: concurrent Crossref/OpenAlex/Semantic Scholar fallback with term-overlap drift guard | ✅/⚠️ | `graph/nodes/search_apis.py` performs real async external calls; query reformulation / drift guard is a simplified term-overlap rather than the designed agentic optimize-search-vet loop. |
| 6 | **Pruner (AttentionRAG)**: real multi-layer attention over micro-windows via int4-quantized model; up to 6.3× | ❌ Stub (mislabeled) | `graph/nodes/pruner.py` — **no model/attention/tensor/quantization anywhere** (repo-wide `grep` for `torch`/`AutoModel`/`output_attentions`/`quantize` returns nothing in source). `_compute_relevance_scores` is Jaccard token overlap + hardcoded bonuses (`+0.35` focal-token substring, `+0.25` if a `[...]` or `===` marker is present). The reported `compression_ratio` is **clamped to `≤ max_compression_ratio` (6.3)** — a ceiling, not a measurement. |
| 7 | **Generator + Critic**: cited synthesis, then an LLM audit with a single auto-regeneration | ✅ Real (quality = backend) | `graph/nodes/generator.py` does a real draft → `_run_critic_audit` (second LLM JSON call) → one regeneration (`retry_count < 1`) with feedback injected. Genuinely a Critic loop **when a live LLM backend is connected.** The "loop" is in-node, not a graph cycle (orchestrator always routes generator → END). |

### 2.2 Three-Layer Global Memory

| # | Design claim | Status | Evidence & notes |
|---|---|---|---|
| 8 | **FAISS** dense store, `IndexFlatIP`, 384-dim, cosine via normalized IP | ✅ Real | `memory/vector_store.py` — real `faiss.IndexFlatIP(384)`, real `all-MiniLM-L6-v2` encoding with L2 normalization, real binary index + JSON sidecar persistence. |
| 9 | **BM25** disk-backed sparse lexical store | ✅ Real | `memory/sparse_store.py` — real `BM25Okapi` over a tokenized corpus, pickled to disk, real top-k. |
| 10 | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) | ❌ Dead code | `SparseStore.rerank()` really loads the cross-encoder and calls `.predict()` — but **no caller invokes it** (no call sites found). |
| 11 | Reciprocal Rank Fusion of dual indices | ❌ Dead code | `SparseStore.reciprocal_rank_fusion()` is a correct RRF implementation with **zero call sites**. |
| 12 | **Neo4j** graph (Ontology/Fact/Passage), real driver + CRUD | ✅ Real | `memory/neo4j_client.py` — real Bolt driver, `verify_connectivity`, parameterized Cypher, CRUD for the 4 labels, idempotent constraints. |
| 13 | Personalized PageRank graph diffusion | ⚠️/❌ | `run_personalised_pagerank()` is **not iterative PPR** — it's a single-pass Cypher formula over a ≤3-hop neighborhood (no convergence, no GDS). It is also **never called** anywhere. |
| 14 | Graceful behaviour when Neo4j is offline | ✅ Real (by swallowing) | Retriever and ingestion both catch the connection error and continue (`neo4j_status="offline"`, empty triples). Note: `ingestion/service.py` calls some Neo4j methods with signatures that don't match the client, so those writes would throw and be swallowed as "offline." |

### 2.3 Offline ingestion & graph construction

| # | Design claim | Status | Evidence & notes |
|---|---|---|---|
| 15 | PyMuPDF (`fitz`) PDF parsing, structure-aware | ✅/⚠️ | `ingestion/parser.py` — real `fitz` parsing, hyphenation repair, regex section-header detection. Structure detection and author detection are heuristic; **publication year is hardcoded to `2024`**, authors fall back to `["Research Team"]`. |
| 16 | Structure-aware chunking respecting section boundaries (~3500 chars) | ✅ Real | `ingestion/chunker.py`, used by `ingestion/service.py`. |
| 17 | Extraction Agent builds Ontology + Fact + Passage layers (LLM-based) | ❌ Regex, not LLM | `ingestion/extractor.py` **imports `llm_client` but never uses it.** `extract_from_chunk` runs ~6 hardcoded regex patterns ("X proposes Y", "X utilizes Y", …); schema types come from a hardcoded `CANONICAL_RELATIONS` dict. |
| 18 | Conflict Detection + Resolution agents (temporal/granularity/mutual-exclusivity); Pending→Stable schema graduation | ❌ Missing | **No conflict detection/resolution code path exists.** The only trace is an unused `ConflictResolutionError` class in `core/exceptions.py`. The ontology `frequency` increment in the Neo4j client is the closest analogue and is not conflict reconciliation. |
| 19 | Ingestion orchestration: MD5 dedup, incremental index add + persist, catalog registry | ✅ Real | `ingestion/service.py` — real MD5 dedup, incremental FAISS/BM25 add + persist, JSON catalog registry, graceful Neo4j handling. |

### 2.4 LLM engine

| # | Design claim | Status | Evidence & notes |
|---|---|---|---|
| 20 | Multi-backend client: Ollama / OpenAI / Anthropic proxy | ✅ Real | `core/llm.py` — real `httpx` calls to `/api/chat`, `/chat/completions`, `/v1/messages`, real token-cost math, backend-disable-on-failure TTL logic. |
| 21 | "Offline fallback" when no backend is available | ❌ Canned / fabricates citations | `_offline_fallback` is **template text, not inference**: router → keyword if/else emitting canned JSON; critic → regex "does the draft contain `[...]`?"; generator → string-templates the first ~4 context lines and appends **invented citation tags** `[Ref-{i}, 2024]` and `[Synthesized Evidence, 2024]`. Because the generator always adds brackets, the bracket-checking "critic" always passes. |

### 2.5 Autonomous ArXiv watcher

| # | Design claim | Status | Evidence & notes |
|---|---|---|---|
| 22 | Queries the official arXiv API, dedups, downloads PDFs, auto-ingests | ✅ Real | `agents/arxiv_watcher.py` — real `httpx` GET to `export.arxiv.org/api/query`, real Atom XML parsing (ElementTree), real PDF download with `%PDF` magic-byte validation, dedup against catalog, then hands off to the real `ingestion_service.ingest_files`. |

### 2.6 Evaluation & observability

| # | Design claim | Status | Evidence & notes |
|---|---|---|---|
| 23 | Ragas-style metrics: Faithfulness, Answer Relevance, Context Recall, Compression Fidelity | ⚠️ Heuristic | `evaluation/metrics.py` — real deterministic math, but lexical proxies rather than true (LLM-judged) Ragas: Faithfulness = answer-token ∩ context-token fraction with a **hardcoded 0.5** fallback when no context; Answer Relevance = **real** embedding cosine similarity; Context Recall = token-coverage of ground-truth facts; Compression Fidelity = ground-truth entity substring retention. |
| 24 | Benchmark runner over a curated dataset | ✅/⚠️ | `evaluation/benchmark.py` runs the real pipeline per sample. Caveat: on pipeline failure it **falls back to `sample.reference_answer`**, which would inflate scores when the pipeline errors. |
| 25 | Pipeline tracing: latency percentiles (p50/p95), cost, rubric trends | ✅ Real | `core/tracing.py` + `/api/v1/metrics`. |

### 2.7 Interfaces & deployment

| # | Design claim | Status | Evidence & notes |
|---|---|---|---|
| 26 | Gradio web console (streaming stepper, scorecard, source inspector, ingestion dock, graph viz) | ✅ Real | `ui/app.py`, `ui/graph_visualizer.py`. Streams node-by-node, renders metrics and a Vis.js graph. |
| 27 | FastAPI REST backend + SQLite session/audit persistence | ✅ Real | `backend/app.py`, `backend/database.py` (WAL-mode SQLite). |
| 28 | Dockerized deployment | ✅ Present | `Dockerfile`, `docker-compose.yml`. |

---

## 3. Highest-impact gaps (ranked)

1. **AttentionRAG is not attention-based** (`graph/nodes/pruner.py`). This is the central research
   claim; it is a lexical Jaccard heuristic, and the "6.3×" figure is a clamp, not a measurement.
2. **Retrieval is not fused or reranked** (`graph/nodes/retriever.py`). RRF, cross-encoder rerank,
   and PPR exist as functions but are never invoked — so the "hybrid memory-guided retrieval"
   degrades to three independent top-k lists.
3. **Offline fallback fabricates citations and self-approves them** (`core/llm.py`). This is a
   correctness/trust hazard given the product's entire premise is verifiable, grounded answers.
4. **Fact/triple extraction is regex, not LLM** (`ingestion/extractor.py`), and **conflict
   detection/resolution is absent** — so the "conflict-free global memory" and ontology quality
   claims are not backed by code.
5. **Reported scores validate heuristics, not the advertised algorithms.** The UI surfaces
   "Evidence Rubric", "Compression 6.3×", and "Critic: Approved (Zero Hallucinations)" as if they
   were rigorous measurements; with no LLM backend they can be heuristic scores over a
   fabricated-citation answer.

---

## 4. What is solid

- End-to-end runnable LangGraph pipeline with a real conditional evidence branch.
- Real FAISS and BM25 stores with real embeddings and persistence.
- Real Neo4j driver + CRUD, with graceful optional-Neo4j behaviour.
- Real multi-backend LLM HTTP client with cost accounting.
- Real arXiv watcher hitting the live API.
- Real ingestion orchestration (dedup, incremental indexing, catalog).
- Real FastAPI backend, SQLite persistence, tracing, and the Gradio console.

---

*Status as verified against the repository at the time of writing. If the flagship algorithms are
subsequently implemented (real attention-based pruning, wired-in RRF/PPR/rerank, LLM extraction,
conflict resolution, and inference-backed fallback), update the corresponding rows above.*
