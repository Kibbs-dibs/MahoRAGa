# MahoRAGa — Feature Improvement Opportunities

_Goal: identify ways to improve the CURRENT features (enhancements, not bug fixes). Bugs are tracked separately in `docs/code-issues-report.md`; this document assumes those are fixed and asks "how do we make each working feature better?"_

Each item: **what exists → the limitation → the proposed improvement → effort/impact**.

---

## 1. Retrieval (Retriever node + three-layer memory)

### 1.1 Make RRF fusion the real ranking, not just a re-ordering
- **Exists:** `reciprocal_rank_fusion` now fuses FAISS + BM25 (Issue 4 fix), but the node still stores two separate per-layer lists re-ordered by fused rank, and the evaluator/pruner consume them separately.
- **Limitation:** the fused list is advisory; downstream still treats layers independently, so the benefit of fusion is diluted.
- **Improvement:** produce a single `fused_chunks` list as the primary retrieval output and have the pruner consume that; keep per-layer lists only for diagnostics/diversity scoring. Add the graph triples into the fusion too (3-way RRF).
- **Effort:** S–M · **Impact:** High (directly improves answer relevance).

### 1.2 Add cross-encoder re-ranking to the live path
- **Exists:** `SparseStore.rerank()` (ms-marco cross-encoder) is implemented but only reachable manually — the retriever never calls it.
- **Limitation:** final ranking is RRF only; a cross-encoder re-rank of the top-K is the single highest-leverage precision boost in modern RAG.
- **Improvement:** after RRF, re-rank the top ~20 fused candidates with the cross-encoder, keep top-K. Gate behind a tier/flag (it costs latency) — also a natural paid feature (see monetization doc).
- **Effort:** S · **Impact:** High.

### 1.3 Replace capitalized-word entity seeding with real NER
- **Exists:** `_extract_entities_from_query` seeds Neo4j PPR from capitalized words / quoted phrases.
- **Limitation:** misses lowercase technical terms ("transformer", "retrieval-augmented generation"), over-matches proper nouns; weak graph recall.
- **Improvement:** use a lightweight NER (spaCy `en_core_web_sm`) or KeyBERT keyphrase extraction for seeds; fall back to the current heuristic offline.
- **Effort:** M · **Impact:** Med–High (graph layer currently contributes least).

### 1.4 Wire true Personalized PageRank
- **Exists:** `run_personalised_pagerank` is a single-pass teleport+degree approximation and is not called by the retriever (flat 1-hop lookup instead).
- **Limitation:** the headline "PPR graph diffusion" feature is not actually exercised at query time.
- **Improvement:** either (a) integrate Neo4j GDS `gds.pageRank` for exact PPR, or (b) make the pure-Cypher version iterative (power iteration, 3–5 rounds) and call it from the retriever to rank graph nodes before serialization.
- **Effort:** M–L · **Impact:** Med (differentiator; depends on graph density).

### 1.5 Embedding model upgrade + chunk strategy
- **Exists:** `all-MiniLM-L6-v2` (384-d, ~256-token window); chunk size now aligned to it (Issue 31 fix).
- **Limitation:** MiniLM is fast but low-ceiling; small window forces many tiny chunks.
- **Improvement:** offer a configurable higher-quality embedder (e.g. `bge-base`/`gte-base`, 512–1024-d, longer context) with a migration path (re-embed corpus). Keep MiniLM as the fast/local default.
- **Effort:** S (config) + M (re-embed tooling) · **Impact:** High.

---

_Further feature areas (evaluator rubric, AttentionRAG pruner, generator/critic, external search, ingestion, UI/UX, observability) are appended in later passes._

---

## 2. Evidence Evaluator (100-point rubric)

### 2.1 Replace proxy heuristics with semantic scoring
- **Exists:** rubric scores retrieval confidence, specificity (Jaccard term overlap), diversity, metadata completeness, recency (keyword "2024/latest" matching).
- **Limitation:** Jaccard term-overlap and keyword-recency are crude proxies — they reward lexical coincidence, not semantic relevance; recency keys off literal year strings in text.
- **Improvement:** score specificity with cosine similarity between the query embedding and chunk embeddings (already available from FAISS); score recency from actual document `year` metadata (now parsed, Issue 34) instead of string matching.
- **Effort:** M · **Impact:** High (the rubric gates external search — better scoring = fewer wasted API calls and better stop decisions).

### 2.2 Make the threshold adaptive / calibrated
- **Exists:** fixed `evidence_threshold = 80`, now per-request overridable (Issue 25).
- **Limitation:** 80 is arbitrary; the right cutoff depends on corpus size and query type.
- **Improvement:** calibrate the threshold against the evaluation benchmark (`evaluation/benchmark.py`) so the weak/strong boundary maximizes answer quality; optionally vary by route (bibliometric vs content).
- **Effort:** M · **Impact:** Med.

### 2.3 Expose a per-component rubric breakdown in the response
- **Exists:** sub-scores are computed and logged but only the total `rubric_score` reaches the UI.
- **Improvement:** return the 5 sub-scores in `QueryResponse` and render them as a small bar/radar in the UI — turns an opaque number into an explainable trust signal (and a sellable "evidence transparency" feature).
- **Effort:** S · **Impact:** Med (UX/trust).

---

## 3. AttentionRAG Pruner (context compression)

### 3.1 Use real attention/embedding relevance instead of Jaccard
- **Exists:** sentence relevance = term-overlap + focal-token bonus + citation bonus; micro-windows now actually drive scoring (Issue 6).
- **Limitation:** the "attention-guided" name oversells a lexical heuristic; it under-ranks semantically relevant sentences that share few exact words with the query.
- **Improvement:** score each micro-window/sentence by cosine similarity to the query (or focal-token) embedding — reuse the FAISS encoder. This is what makes the compression genuinely attention/semantics-guided and would materially raise pruned-context quality.
- **Effort:** M · **Impact:** High.

### 3.2 Adaptive compression ratio by query complexity
- **Exists:** fixed target `max_compression_ratio` (6.3), per-request overridable.
- **Limitation:** one ratio for all queries; simple questions waste tokens, complex ones get over-pruned.
- **Improvement:** scale the budget by query length / number of sub-claims / evidence spread; keep the ceiling as a cap.
- **Effort:** S–M · **Impact:** Med.

### 3.3 Guaranteed citation preservation
- **Exists:** sentences with bracketed citations get a scoring bonus.
- **Limitation:** a bonus is not a guarantee — a cited sentence can still be pruned under tight budgets, breaking the citation chain the generator depends on.
- **Improvement:** always retain at least the top-N cited/provenance sentences regardless of budget; prune only non-cited filler to hit the ratio.
- **Effort:** S · **Impact:** Med–High (protects the citation-grounding promise).

---

_Further feature areas (generator/critic, external search, ingestion, UI/UX, observability) are appended in later passes._

---

## 4. Generator & Critic (answer synthesis + self-correction)

### 4.1 Streaming responses
- **Exists:** synchronous generate → audit → (regenerate) → return; the UI waits for the whole thing.
- **Limitation:** multi-LLM-call latency (generator + critic + optional regen) with no feedback; feels slow.
- **Improvement:** stream the draft answer token-by-token to the UI (both Ollama and OpenAI support SSE/streaming), then show the critic verdict as a post-step badge. Large perceived-latency win.
- **Effort:** M · **Impact:** High (UX).

### 4.2 Structured citations with click-through to source
- **Exists:** citations are free-text bracketed tags the model emits; critic checks a bracket is present.
- **Limitation:** citations don't link back to the actual chunk/paper; users can't verify.
- **Improvement:** pass chunk ids into the prompt and have the generator cite by id, then resolve ids → clickable source (paper title + page) in the UI. Turns "has a bracket" into true provenance.
- **Effort:** M–L · **Impact:** High (core trust feature; also sellable).

### 4.3 Critic that returns actionable spans, and a confidence score
- **Exists:** critic returns pass/fail + feedback; one regeneration; now re-audited (Issue 8).
- **Improvement:** have the critic return the specific offending claim spans and an overall confidence (0–1); surface confidence to the user; feed spans into a targeted regeneration prompt rather than a full rewrite.
- **Effort:** M · **Impact:** Med–High.

### 4.4 Answer caching
- **Exists:** every query runs the full pipeline fresh.
- **Improvement:** cache (query-embedding, corpus-version) → answer; serve near-duplicate queries from cache (semantic cache with a similarity threshold). Cuts LLM cost and latency materially.
- **Effort:** M · **Impact:** High (cost + speed; directly improves unit economics in the monetization plan).

---

## 5. External Academic Search (fallback agent)

### 5.1 Ingest fetched abstracts into memory (close the loop)
- **Exists:** external abstracts are added to `MemoryState` for the current query and now re-scored (Issue 11), but discarded after.
- **Limitation:** the same external lookup is repeated on similar future queries; knowledge isn't retained.
- **Improvement:** optionally persist high-value fetched abstracts into FAISS/BM25 (and graph) so the corpus self-enriches from external search — the system literally learns from use.
- **Effort:** M · **Impact:** Med–High (compounding corpus value).

### 5.2 LLM-based query reformulation for external search
- **Exists:** `_reformulate_query` uses crude heuristics (add plurals/`-ing`, academic modifier words) with a drift guard.
- **Limitation:** mechanical expansion rarely improves academic-API recall.
- **Improvement:** use the LLM (already wired) to generate 2–3 reformulated academic queries per iteration; keep the drift guard as the safety net.
- **Effort:** S–M · **Impact:** Med.

### 5.3 Dedup + rank external results against internal corpus
- **Exists:** external abstracts vetted by term overlap, deduped by first-100-char.
- **Improvement:** embed external abstracts and (a) drop ones already covered by internal chunks, (b) RRF them into the same fused ranking as internal evidence instead of appending blindly.
- **Effort:** M · **Impact:** Med.

### 5.4 Add citation metadata + more sources
- **Improvement:** capture DOI/authors/year (partly collected) and expose them as structured citations; consider adding arXiv API and PubMed for broader coverage; add per-source result caps so one API can't dominate.
- **Effort:** S–M · **Impact:** Med.

---

_Further feature areas (ingestion, UI/UX, observability) are appended in later passes._

---

## 6. Ingestion (parser, chunker, extractor, arXiv watcher)

### 6.1 LLM-based fact extraction (as the PRD intends)
- **Exists:** regex pattern extraction (~6 verb patterns); `llm_client` import was dead (removed, Issue 32).
- **Limitation:** catches very few facts, misses most relations → sparse graph layer.
- **Improvement:** add an LLM extraction path (structured JSON triples with schema) using the already-available `llm_client`; keep regex as the offline/no-LLM fallback. This is the single biggest lever on graph-layer quality.
- **Effort:** M · **Impact:** High.

### 6.2 Better title/metadata extraction
- **Exists:** title = first line of page 1; authors by heuristic; year now parsed (Issue 34).
- **Improvement:** use PDF embedded metadata (`doc.metadata`) and a largest-font-on-page-1 heuristic for title; cross-check against arXiv/Crossref by DOI when available.
- **Effort:** S–M · **Impact:** Med (clean metadata improves recency scoring, citations, dedup).

### 6.3 Incremental / resumable ingestion + atomicity
- **Exists:** ingestion now reports partial_success honestly (Issue 21) but still has no rollback.
- **Improvement:** stage all-three-layer writes and commit atomically (or mark a doc re-ingestable on partial failure); add a "re-ingest failed docs" action.
- **Effort:** M · **Impact:** Med.

### 6.4 Scheduled autonomous arXiv watching
- **Exists:** `fetch_and_ingest` is manual/endpoint-triggered.
- **Improvement:** a scheduled daily watcher (cron) that harvests new papers for the user's tracked topics and notifies them — turns the corpus into a living feed (recurring-value hook from the monetization doc).
- **Effort:** S–M · **Impact:** Med–High (retention).

---

## 7. UI / UX (Gradio app)

### 7.1 Surface provenance and evidence transparently
- Show retrieved sources (per layer), the rubric sub-scores, and whether external search fired — make the pipeline legible instead of a black box. Pairs with 2.3 and 4.2.
- **Effort:** M · **Impact:** High (trust + differentiation).

### 7.2 Corpus management UX
- Upload progress (callback already exists), per-document view, delete/re-ingest, and a corpus-health panel (counts per layer, last arXiv sync). Partly present — extend it.
- **Effort:** M · **Impact:** Med.

### 7.3 Interactive knowledge-graph view
- `/api/v1/graph` now returns real triples (Issue 24) and `ui/graph_visualizer.py` exists — add an interactive, filterable graph explorer (zoom to a paper's neighborhood).
- **Effort:** M · **Impact:** Med (demo/wow + genuine research utility).

### 7.4 Conversational follow-ups
- Sessions persist (SQLite) but each query is independent. Use prior turns as context for follow-up questions (multi-turn RAG).
- **Effort:** M · **Impact:** High (matches how researchers actually work).

---

## 8. Observability & Evaluation

### 8.1 Act on the metrics already collected
- **Exists:** `core/tracing.py` + `query_audits` capture per-node latency, rubric, cost, critic verdict; `/api/v1/metrics` exposes p50/p95.
- **Improvement:** a dashboard of these (latency per node, cost per query trend, external-search hit rate, critic regeneration rate) — operational visibility that's nearly free to build on existing data.
- **Effort:** S–M · **Impact:** Med.

### 8.2 Continuous evaluation gate
- **Exists:** `evaluation/benchmark.py` (Ragas-style) + 53 passing tests.
- **Improvement:** run the benchmark in CI on every change and track faithfulness/relevance/recall over time; fail the build on regression. Turns quality into a measured, defended property.
- **Effort:** M · **Impact:** High (quality assurance buyers expect).

---

## 9. Priority matrix (impact × effort)

**Do first — high impact, low/med effort (quick wins):**
- 1.2 Cross-encoder re-rank in the live path (S, High)
- 3.3 Guaranteed citation preservation (S, Med–High)
- 4.1 Streaming responses (M, High)
- 4.4 Semantic answer caching (M, High — also cuts cost)
- 2.3 / 4.2 Expose rubric sub-scores + critic confidence in UI (S, trust)

**High impact, bigger effort (flagship bets):**
- 1.1 RRF as primary ranking + 3-way fusion (High)
- 3.1 Embedding-based pruner relevance (High)
- 4.2 Structured click-through citations (High, core trust)
- 6.1 LLM-based fact extraction (High, fixes the weak graph layer)
- 7.1 / 7.4 Provenance UI + multi-turn conversations (High)

**Foundational / enabling:**
- 2.1 Semantic rubric scoring; 2.2 calibrated threshold
- 8.2 Continuous eval gate in CI
- 1.5 Configurable higher-quality embedder

**Later / lower:**
- 1.4 true PPR, 5.x external-search refinements, 6.3 atomic ingest, 7.2/7.3 corpus+graph UX, 8.1 ops dashboard

### Bottom line
The biggest returns cluster in three places: **(1) retrieval precision** (cross-encoder re-rank, semantic/embedding scoring across evaluator+pruner, RRF as primary), **(2) trust/provenance UX** (click-through citations, exposed evidence/confidence, streaming), and **(3) the graph layer** (LLM fact extraction + real PPR). Quick wins (re-rank, citation preservation, streaming, caching) deliver visible quality/latency/cost improvements with modest effort and should lead.



