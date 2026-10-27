# Code Issues Report

Patrol goal: Find issues with the code. Read-only static review (host memory critically low — no test/build runs).

## Executive summary — ranked by impact

Reviewed all 6 graph nodes, both memory stores, config, the LLM client, the full ingestion layer (parser/chunker/extractor/service), the FastAPI backend + schemas, the SQLite layer, the arXiv watcher, and the ingestion test. **40 issues** found. Highest-impact first:

| # | Severity | Location | One-line |
|---|----------|----------|----------|
| 37 | High (confirmed) | `ingestion/service.py` ↔ `memory/neo4j_client.py` | `upsert_document(url=, publication_year=)` kwargs don't match the signature → `TypeError` on every ingest, swallowed as "Neo4j offline"; graph layer never populates. |
| 18 | High | `core/llm.py` offline fallback | Fabricates `[Ref-N, 2024]` citations that match no source; offline critic passes them → silent ungrounded answers (fresh-install default). |
| 8 | Med-High | `graph/nodes/generator.py` | Regenerated answer is never re-audited → breaks the "zero-hallucination" guarantee. |
| 12 | Medium (confirmed) | `graph/nodes/search_apis.py` | `asyncio.gather(return_exceptions=True)` + `return all_results` makes the whole 429 retry/backoff loop unreachable dead code. |
| 4 | Medium | `graph/nodes/retriever.py` | Promised RRF dual-index fusion is never called; FAISS/BM25 kept unmerged. |
| 6 | Medium | `graph/nodes/pruner.py` | `micro_windows` computed but never used for scoring — AttentionRAG step is decorative. |
| 10 | Medium | `graph/orchestrator.py` | `route` classified but never branched on → greetings run the full pipeline. |
| 13 | Medium | `memory/vector_store.py` | `add_vectors` may index un-normalised vectors (normalise-then-copy bug) → corrupts cosine scores. |
| 15 | Medium | `memory/sparse_store.py` | BM25 index loaded via `pickle.load` → untrusted-deserialization/RCE risk. |
| 21 | Medium | `ingestion/service.py` | No atomicity across FAISS/BM25/Neo4j/catalog → layers diverge, MD5-dedup can lock out retry. |
| 1 | Medium (latent) | `graph/nodes/evaluator.py` | `MemoryState`/`QueryState`/`List` used in annotations but never imported. |
| 11,16,17,19,20 | Low-Med | various | re-score gap, weak default password, dead PPR config, opaque backend errors, drift-prone chunk count. |
| 2,3,5,7,9,14 | Low | various | dead None-guards, missing `ge=0`, hot-path `ensure_indexes`, header scoring, stale feedback, RRF empty-key collapse. |

Full details below.

## How to verify the two confirmed, highest-impact bugs

```bash
# Issue 37 — upsert_document signature mismatch (TypeError on every Neo4j ingest)
python -c "import inspect; from memory.neo4j_client import Neo4jClient; print(inspect.signature(Neo4jClient.upsert_document))"
#   -> (self, doc_id, title='', authors='', year=None)   # no 'url', no 'publication_year'
grep -n "publication_year=\|url=doc.file_path" ingestion/service.py   # the offending call (lines 322-324)

# Issue 12 — unreachable 429 retry loop
grep -n "return_exceptions=True" graph/nodes/search_apis.py           # line 243
grep -n "return all_results" graph/nodes/search_apis.py               # line 254 (fires before any except)
```

## Issue 1 — Missing imports in `graph/nodes/evaluator.py`

**Severity:** Medium (latent). Does not crash at import because `from __future__ import annotations` makes all annotations lazy strings, but it is a real bug.

**Details:** The module uses three names that are never imported:

- `MemoryState` — used as a parameter annotation in `_score_retrieval_confidence`, `_score_answer_specificity`, `_score_source_diversity`, `_score_metadata_completeness`, `_score_recency`.
- `QueryState` — used as a parameter annotation in `_score_answer_specificity`.
- `List` — used as `List[str]` in `_calculate_text_specificity`.

Only `AletheiaState` and `EvaluationState` are imported from `core.state`; nothing is imported from `typing`.

**Breaks when:** `typing.get_type_hints()` is called on these functions, a tool resolves annotations eagerly, or a developer removes `from __future__ import annotations`. Static type checkers (mypy/pyright) flag undefined names.

**Fix:** add to the imports:
```python
from typing import List
from core.state import AletheiaState, EvaluationState, MemoryState, QueryState
```
**Confirmed:** `core/state.py` exports both `MemoryState` and `QueryState`, so the fix above is valid as written.

---

## Issue 2 — `evaluator.py` init guard is dead code

`evaluator_node` starts with `if state.evaluation is None: state.evaluation = EvaluationState()`. But `AletheiaState.evaluation` is declared `Optional[EvaluationState] = Field(default_factory=EvaluationState)`, so it is never `None` in practice. Harmless but misleading — signals an intent (nullable evaluation) the model does not actually allow. Same pattern applies to `state.memory` (also `default_factory=MemoryState`), yet the code treats `memory is None` as a real branch.

**Severity:** Low (dead/misleading code).

---

## Issue 3 — Rubric sub-scores allow negative values

In `core/state.py`, `EvaluationState` sub-score fields use only an upper bound:
```python
retrieval_confidence: int = Field(default=0, le=40)
answer_specificity: int = Field(default=0, le=25)
source_diversity: int = Field(default=0, le=15)
metadata_completeness: int = Field(default=0, le=10)
recency: int = Field(default=0, le=10)
```
None declare `ge=0`, so a negative assignment would pass Pydantic validation and produce a `rubric_score` outside 0–100 logic. The scoring helpers never emit negatives today, so this is latent.

**Fix:** add `ge=0` to each sub-score field.

**Severity:** Low (latent, defensive).

---

## Issue 4 — Promised RRF fusion is not applied in the retrieval path

**Severity:** Medium (functional gap vs. spec).

The README/architecture promise "Reciprocal Rank Fusion (RRF) for dual-index merging" and the retriever docstring repeats it. But `graph/nodes/retriever.py::retriever_node` stores each layer's results into separate `MemoryState` fields (`serialized_graph_triples`, `vector_chunks`, `lexical_chunks`) and **never calls** `reciprocal_rank_fusion`. FAISS and BM25 results are kept side by side, not fused/re-ranked. The downstream evaluator just concatenates them. So the headline "RRF dual-index merge" feature is effectively absent from the live query path even though `memory/sparse_store.py` exposes the utility.

**Fix:** fuse FAISS + BM25 (and optionally graph) ranked lists via `SparseStore.reciprocal_rank_fusion(...)` before writing to `MemoryState`, or document that fusion happens elsewhere.

---

## Issue 5 — `ensure_indexes()` runs on every query (hot path)

`_retrieve_from_neo4j` opens a new `Neo4jClient()` and calls `client.ensure_indexes()` for **every** query. Index creation is idempotent but is a schema round-trip that does not belong on the per-request retrieval path; it should run once at startup/ingestion. Also every layer swallows all exceptions and returns `[]`, so a Neo4j/FAISS/BM25 outage is invisible to the caller (graceful degradation, but no signal — consider logging at ERROR + a health flag).

**Severity:** Low–Medium (performance + observability).

---

## Issue 6 — Micro-windows are computed but never used for scoring

**Severity:** Medium (feature is decorative, not functional).

In `graph/nodes/pruner.py::pruner_node`, `_segment_micro_windows(assembled_raw, 800)` builds `micro_windows` and stores them in `CompressionState.micro_windows`, but pruning actually scores a **separate** segmentation from `_split_into_sentences(assembled_raw)`. The micro-window segmentation plays no role in relevance scoring or pruning — it is written to state and otherwise ignored. The spec describes "attention weights across micro-windows of context"; the implementation scores plain sentences instead. So the micro-window step is decorative.

Also note the two segmentations use different limits (`_segment_micro_windows` default `600`, called with `800`; sentences split on `[.!?]`/newlines), so `micro_windows` count in logs does not correspond to what was pruned.

---

## Issue 7 — Section headers are scored and can crowd out content

`assembled_raw` interleaves literal header lines (`=== Knowledge Graph Triples ===`, etc.). `_split_into_sentences` turns these into "sentences", and both `_compute_relevance_scores` (`sent.startswith("===")` → +0.25) and `_prune_sentences` sort_key (+0.5) give them a strong bonus. On small contexts these headers can win budget over real evidence sentences. Headers carry no evidence; they should be excluded from scoring or re-attached without consuming budget.

**Severity:** Low–Medium.

---

## Issue 8 — Regenerated answer is never re-audited (breaks "zero-hallucination" claim)

**Severity:** Medium–High (correctness vs. core promise).

In `graph/nodes/generator.py::generator_node`, when the Critic fails the first draft the node regenerates once (`retry_count = 1`) and stores `retry_resp.content` as the final `draft_answer` — but **never runs `_run_critic_audit` again** on the corrected draft. The regenerated answer is returned unverified. If the second attempt still hallucinates or drops citations, nothing catches it. For a pipeline whose stated purpose is a "zero-hallucination" self-correction loop, the final output escaping the audit is a real defect.

**Fix:** after regeneration, re-run the critic on the new draft (and record the second verdict). The single-retry cap already prevents infinite loops.

---

## Issue 9 — Stale `critic_feedback` after regeneration

When regeneration runs, `state.generation.critic_feedback` is set to the feedback about the **first** (discarded) draft and left unchanged after the new draft replaces `draft_answer`. The stored feedback therefore no longer describes the returned answer — misleading for any UI/log that surfaces it. Tied to Issue 8: a re-audit would also refresh this field correctly.

**Severity:** Low (stale state / observability).

---

## Issue 10 — `route` is computed but never used (dead classification)

**Severity:** Medium (wasted work / cost).

`router_node` classifies every query into `content` / `bibliometric` / `trend` / `current_world` and stores it in `QueryState.route`, but `graph/orchestrator.py` has a fixed path `router → retriever → evaluator → (search_apis) → pruner → generator` with **no conditional edge on `route`**. Consequences:

- A `current_world` query (greeting, trivia, out-of-domain — e.g. "hi") still runs full Neo4j + FAISS + BM25 retrieval, 100-point evidence scoring, likely external API search, and two LLM calls. It should short-circuit to a direct/cheap answer.
- `bibliometric` / `trend` routes get no specialized retrieval despite being distinct intents.

The classification is effectively dead — computed, logged, and ignored. Either branch on `route` in the orchestrator or drop the field.

**Fix:** add a conditional edge after `router` that sends `current_world` straight to `generator` (or a lightweight responder), and optionally specialize retrieval per route.

---

## Issue 11 — Evidence is never re-scored after external search

**Severity:** Low–Medium.

Orchestrator path is `evaluator → (is_evidence_weak) → search_apis → pruner → generator`. After `search_apis_node` adds `external_api_abstracts`, the evaluator does **not** run again, so `EvaluationState.is_evidence_weak` / `rubric_score` remain stale and never reflect the externally fetched evidence. The sufficiency rubric is advertised as the gate but is only ever computed on internal evidence. Consider a re-evaluation edge after `search_apis`.

---

## Issue 12 — 429 retry/backoff loop is unreachable dead code

**Severity:** Medium (resilience feature does not work).

In `graph/nodes/search_apis.py::_perform_external_search_with_retry`, the three API calls run via:
```python
await asyncio.gather(crossref_task, openalex_task, semantic_scholar_task, return_exceptions=True)
```
`return_exceptions=True` means a per-API failure (including `httpx.HTTPStatusError` 429) is **returned as a value**, not raised. The surrounding `try/except httpx.HTTPStatusError` / `except Exception` therefore almost never fires — `gather` itself does not raise. As a result the entire exponential-backoff + jitter retry loop (the `for attempt in range(1 + max_retries)` block, the 429 handling, `await asyncio.sleep(delay)`) is effectively unreachable: the function returns `all_results` on the first attempt every time, silently dropping any rate-limited API. The documented rate-limit resilience does not actually operate.

**Fix:** inspect the gathered results for `httpx.HTTPStatusError` with status 429 and trigger the backoff/retry explicitly (per-API or whole-batch), rather than relying on `gather` to propagate.

---

## Issue 13 — `VectorStore.add_vectors` may index un-normalised vectors

**Severity:** Medium (silent correctness — breaks cosine assumption).

In `memory/vector_store.py::add_vectors`:
```python
faiss.normalize_L2(vectors)
self._index.add(vectors.astype(np.float32))
```
`faiss.normalize_L2` normalises **in place** and requires a C-contiguous float32 array. If `vectors` is float64 (numpy's default dtype) or non-contiguous, `normalize_L2` may raise, or normalise a buffer that is then discarded — because `.astype(np.float32)` makes a **new copy**. In the dtype-mismatch path the copy added to the index is NOT the normalised array, so vectors enter an `IndexFlatIP` without unit norm. Since the index assumes L2-normalised vectors to approximate cosine similarity, un-normalised rows corrupt all similarity scores involving them, silently.

**Fix:** cast first, then normalise the exact array you add:
```python
vectors = np.ascontiguousarray(vectors.astype(np.float32))
faiss.normalize_L2(vectors)
self._index.add(vectors)
```
(`add_chunks` is safe because `encode(...)` already requests `normalize_embeddings=True` and returns float32.)

---

## Issue 14 — RRF dedup key collapses on empty/missing `text`

**Severity:** Low.

`SparseStore.reciprocal_rank_fusion` dedupes documents by `doc.get("text", "")`. Any results lacking a `text` key (or with empty text) all hash to the key `""` and get merged into a single fused entry, incorrectly summing their RRF contributions and keeping only the first's metadata. Graph triples in particular may not always carry a `text` field. Use a stable unique id (chunk_id/doi) with text as fallback.

---

## Issue 15 — BM25 index loaded via `pickle.load` (untrusted deserialization)

**Severity:** Medium (security / robustness).

`memory/sparse_store.py::load` reads `bm25_index.pkl` with `pickle.load`. Pickle executes arbitrary code during deserialization, so if the on-disk index file is replaced or tampered with, loading it is remote-code-execution on the host. The file lives under `data/` and is regenerated by ingestion, but there is no integrity check. At minimum: document the trust boundary; better, store the tokenised corpus as JSON and rebuild `BM25Okapi` on load (fast, and `_rebuild_index` already exists), eliminating the pickle entirely.

**Fix:** persist `tokenised_corpus` as JSON and call `_rebuild_index()` on load instead of unpickling the `BM25Okapi` object.

---

## Issue 16 — Insecure hardcoded default for `neo4j_password`

**Severity:** Low–Medium (security smell).

`core/config.py` sets `neo4j_password` default to `"password"`. If `.env`/env vars are not provided the system silently connects (or attempts to) with a trivially guessable credential, and nothing warns the operator. Secret-like settings should default to `None`/empty and fail loudly when required, not to a usable weak value.

---

## Issue 17 — PPR tuning settings (`ppr_alpha`, `ppr_top_k`) are never used

**Severity:** Low (dead config vs. spec).

`settings.ppr_alpha` and `settings.ppr_top_k` are defined and advertised as PPR parameters, but `graph/nodes/retriever.py::_retrieve_from_neo4j` never reads them — it calls `client.serialise_subgraph_to_sentences(entity, limit=10)` with a hardcoded `limit=10` and later slices by `settings.top_k_retrieval`. The Personalized PageRank damping factor and top-k are not threaded into the actual graph retrieval call, so the documented PPR knobs have no effect. Either wire them into the Neo4j PPR query or remove them.

---

## Issue 18 — Offline fallback fabricates citations (`[Ref-N, 2024]`)

**Severity:** Medium–High (citation-integrity violation).

`core/llm.py::_offline_fallback` (answer-generation branch) emits lines like `- {fact} [Ref-{idx}, 2024]` and a closing `[Synthesized Evidence, 2024]`. These bracketed labels are **invented** — they do not correspond to real source identifiers in the context, and the year `2024` is hardcoded. The whole project is built around "every technical claim MUST end with a bracketed citation matching the source." The offline generator manufactures citations that match nothing, and because the offline **critic** (`_offline_fallback` critic branch) passes any draft that merely *contains* `[...]`, these fabricated citations sail through the audit. When no LLM backend is reachable (the default on a fresh install — Ollama not pulled, no API keys), the pipeline silently produces plausible-looking but ungrounded, fake-cited answers.

**Fix:** offline-generated citations should reference actual source ids extracted from the context (or be clearly marked as unverified, e.g. `[offline-synthesis — unverified]`), and the offline critic should not pass fabricated-citation drafts.

---

## Issue 19 — Backend errors are indistinguishable from "backend down"

**Severity:** Low–Medium (observability).

`_try_ollama` / `_try_openai` / `_try_anthropic_proxy` all return `None` on *any* non-200 status or exception, and `call()` then disables that backend for 30s and silently moves to the next (ultimately the offline fallback). A real misconfiguration (Ollama model not pulled → 404, bad API key → 401, proxy 500) is swallowed with only a `logger.debug` and looks identical to "server not running." Operators get silent degradation to deterministic offline output with no surfaced cause. Log non-200 responses at WARNING with status + body snippet, and distinguish auth/config errors (do not silently fall through) from connectivity errors.

---

## Issue 20 — `total_chunks` recomputed from catalog, can drift from real index size

**Severity:** Low.

In `ingestion/service.py::ingest_files`:
```python
catalog["total_chunks"] = len(catalog["documents"]) and sum(d.get("chunk_count", 0) for d in catalog["documents"])
```
The `and` guard works (empty docs → 0) but is a fragile idiom, and the count is derived from per-doc `chunk_count` fields rather than the authoritative FAISS/BM25 index sizes. A record missing `chunk_count` (e.g. bootstrap-seeded entries) silently undercounts. Prefer `v_store.size` / `s_store.size` as the source of truth, or guard explicitly with `if catalog["documents"]`.

---

## Issue 21 — No atomicity across the three stores (partial-ingest / dedup lockout)

**Severity:** Medium (data integrity).

`ingest_files` writes FAISS, then BM25, then Neo4j, then the catalog, with no transaction or rollback:

- Neo4j failure is caught and recorded as `neo4j_status="offline"`, but the overall `status` is still `"success"` and the catalog is committed — so vector+lexical layers contain chunks the graph layer never got, while the catalog claims full success. The three-layer memory silently diverges.
- If `v_store.save()` or `s_store.save()` itself raises (disk full, permission), the exception propagates **after** files were copied into `data/uploads/` and chunks embedded, but **before** the catalog update — a partial write with no cleanup.
- Worse, because dedup is by MD5 against the catalog, a *failed* ingest that left the upload copy / partial index behind can still be re-detected: on retry the file may be skipped as a duplicate (if any prior attempt reached the catalog) or re-copied, with no reconciliation. There is no "mark failed / allow re-ingest" path.

**Fix:** stage all three writes and commit the catalog only after all succeed (or record a per-layer status and expose it, not a blanket `success`); on failure, roll back the upload copy and any partial index append, or mark the doc re-ingestable.

---

## Issue 22 — Insecure CORS: `allow_origins=["*"]` with `allow_credentials=True`

**Severity:** Medium (security).

`backend/app.py` configures:
```python
allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
```
Wildcard origin **with** credentials is an invalid/insecure combination: the CORS spec forbids `Access-Control-Allow-Origin: *` on credentialed requests, so browsers block them, and it advertises intent to accept cookies from any origin (CSRF / credential-leak footgun if origin handling changes). Pin `allow_origins` to the actual dashboard origins, or set `allow_credentials=False` if no cookies are used.

---

## Issue 23 — `/api/v1/query` constructs `CompressionState(max_compression_ratio=...)` — field does not exist

**Severity:** Low (CORRECTED — not a crash).

`execute_query` builds `CompressionState(max_compression_ratio=req.max_compression_ratio)`, but `CompressionState` has no such field. **Confirmed not a crash:** `core/state.py` sets no `model_config`, and Pydantic v2's default is `extra='ignore'`, so the unknown kwarg is silently dropped (not raised). I initially over-rated this as High; corrected here. The kwarg is simply dead — see Issue 25 for the real impact.

**Fix:** drop the kwarg: `CompressionState()`.

---

## Issue 24 — `/api/v1/graph` returns fabricated triples, not the real graph

**Severity:** Low–Medium (misleading output).

`get_graph_topology` does not read Neo4j; it synthesizes `"{title} PROPOSES Methodology"` / `"{title} EVALUATES Benchmarks"` from catalog titles and returns those as the "knowledge graph." The endpoint is documented as "Export current active knowledge graph nodes and edges" but returns invented topology. Either query the real graph via `Neo4jClient` or rename/document it as a placeholder.

---

## Issue 25 — Four `QueryRequest` parameters are accepted but never applied

**Severity:** Medium (API contract lie).

`backend/schemas.py::QueryRequest` exposes `llm_provider`, `local_model`, `evidence_threshold`, `max_compression_ratio`, and `enable_external_search`. In `backend/app.py::execute_query` **none of these reach the pipeline**:

- `max_compression_ratio` → passed to `CompressionState` and silently ignored (Issue 23); the pruner uses `settings.max_compression_ratio`.
- `evidence_threshold` → never read; the evaluator uses `settings.evidence_threshold`.
- `enable_external_search` → never read; the evaluator→search_apis edge ignores it.
- `llm_provider` / `local_model` → never read; `llm_client` always uses its fixed built-in backend order.

The API advertises per-request configurability that has no effect — a client setting `enable_external_search=False` or a custom `evidence_threshold` is silently ignored. Either thread these into the pipeline as per-request overrides or remove them from the schema.

---

## Issue 27 — ArXiv watcher dedups by title substring; `known_hashes` is dead; disagrees with ingest MD5 dedup

**Severity:** Medium (correctness / wasted work).

In `agents/arxiv_watcher.py::fetch_and_ingest`:

- Dedup uses title-substring matching: `any(p.title.lower() in t or t in p.title.lower() for t in known_titles if len(t) > 5)`. Substring matching wrongly skips a paper whose title is a substring of an unrelated longer title (and vice versa). Short/generic titles are especially prone.
- `known_hashes = {d.get("md5_hash") ...}` is computed but **never used** — dead variable. The real MD5 dedup happens later inside `ingestion_service.ingest_files`, which uses a *different* key than the watcher's title check.
- Because the two layers dedup differently, the watcher can download a PDF that passes its title filter only for `ingest_files` to skip it as an MD5 duplicate (wasted download + upload copy), or report `already_synced` for a paper that is not actually in the indexes.

**Fix:** dedup on a stable id (arXiv id / MD5) consistently across watcher and ingestion; remove the dead `known_hashes` or actually use it.

---

## Issue 26 — Fragile arXiv version stripping (`split("v")[0]`)

**Severity:** Low.

`parse_arxiv_atom_feed` derives the id via `raw_id.split("/abs/")[-1].split("v")[0]`. Splitting on the literal `"v"` to drop the version suffix (`2305.12345v2` → `2305.12345`) also truncates at any other `v` in the string (e.g. old-style ids or unexpected formats). Prefer a precise regex like `re.sub(r"v\d+$", "", id_part)` anchored to a trailing `v<digits>`.

---

## Issue 29 — `add_message` auto-create path creates a DIFFERENT session id → FK violation

**Severity:** Medium (bug in the fallback path).

In `backend/database.py::add_message`:
```python
if not self.get_session(session_id):
    title = content[:40].strip() + ("..." if len(content) > 40 else "")
    self.create_session(title=title)   # generates a NEW random sess_... id, return ignored
...
conn.execute("INSERT INTO messages (... session_id ...) VALUES (... session_id ...)")
```
When `session_id` does not exist, the code creates a session with a **brand-new** `sess_<uuid>` id and discards it, then inserts the message with the **original** `session_id` that still does not exist. Since `PRAGMA foreign_keys = ON` and `messages.session_id` has a FK to `sessions(id)`, this insert raises an `IntegrityError`. The intended "ensure session exists" guard therefore fails exactly when it is needed. (In the normal `/api/v1/query` flow this is masked because `execute_query` already created/validated the session first — so it bites any direct `add_message` call with an unknown id.)

**Fix:** insert the session row using the passed `session_id` (e.g. a `create_session_with_id(session_id, title)` or an `INSERT OR IGNORE INTO sessions(id,...)`), not a freshly generated one.

---

## Issue 28 — Thread-local SQLite connections are never closed (resource leak)

**Severity:** Low–Medium.

`DatabaseManager` caches one connection per thread in `self._local.conn` and provides no `close()`/teardown. Under uvicorn/Gradio worker-thread pools each worker opens its own SQLite connection that lives for the thread's lifetime and is never released; WAL checkpoints can be deferred and file handles accumulate on a long-running server. Add a `close()`/context-manager teardown and/or close idle thread-local connections.

---

## Issue 31 — Chunk size (3,500 chars) far exceeds the embedding model's input window

**Severity:** Medium–High (systematic dense-retrieval quality loss).

`settings.chunk_size = 3500` characters, and `chunk_document` targets ~3,500-char chunks. But the dense embedder is `all-MiniLM-L6-v2`, whose max input is **256 word-pieces (~roughly 400–512 characters)**; `sentence-transformers` silently truncates longer input. So FAISS embeds only the first ~1/7th of each chunk — the remainder of every oversized chunk never influences its dense vector. BM25 indexes the full text, so lexical and dense layers represent *different* spans of the same chunk, and dense retrieval is systematically based on truncated openings. This undermines the whole "semantic" retrieval layer.

**Fix:** set `chunk_size` to match the embedder window (e.g. ~256 tokens / ~1,000 chars), or embed sub-windows and pool, or switch to an embedding model with a larger context. At minimum, document the truncation.

---

## Issue 30 — Oversized single sentence still emitted whole; separator mixing

**Severity:** Low.

In `chunk_document`'s oversized-paragraph branch, a single sentence longer than `max_size` is not split further — it is flushed as a new `sent_buffer=[sent]` and can exceed the target. Also the leftover `sent_buffer` is carried into `current_paras` as one element joined with spaces, then later re-joined with `"\n\n"`, mixing sentence-space and paragraph-blank-line separators within a chunk. Cosmetic/robustness, not a crash.

---

## Issue 32 — Knowledge extraction is brittle regex, not the advertised LLM extraction; `llm_client` imported but unused

**Severity:** Medium (spec vs code; weak graph layer).

`ingestion/extractor.py` imports `from core.llm import llm_client` but **never uses it**. `extract_from_chunk` relies entirely on ~6 hand-written regexes matching `[Capitalized] (proposes|utilizes|achieves|…) [short phrase]`. The PRD/architecture describe MemGraphRAG LLM-based fact extraction with conflict detection; the implementation is pattern-matching that only fires on sentences fitting those exact shapes. Consequences:

- The Neo4j graph layer is populated only by whatever these patterns catch — typically very few facts, mostly missing real relations.
- Because chunk content is prefixed with a `[Title | Section | Page]` header (Issue 31 / chunker), the `[A-Z]...` head pattern can match header tokens, producing spurious facts from provenance headers.
- `similarity_weight` is hardcoded to `1.0` for every fact — the "SimilarityWeight" the schema advertises is never computed.

**Fix:** either implement the LLM-based extraction the design calls for (using the already-imported `llm_client`) or drop the dead import and document the extractor as a regex heuristic with its known limits.

---

## Issue 33 — Pronoun-substitution and header-matching edge cases in extractor

**Severity:** Low.

In `extract_from_chunk`, when `head` is a pronoun it is replaced with `chunk.doc_title.split(":")[0]`, but this is done after the `len(head) < 2` filter and the replacement is not re-validated or de-duplicated against existing pairs for the new head. Combined with the header-prefixed chunk text, the capitalized-head regex also matches tokens inside the `[Title | Section | Page]` header line. Strip the provenance header before pattern extraction and re-check substituted heads.

---

## Issue 36 — No-section fallback discards every page after page 1 (data loss)

**Severity:** Medium–High (silent data loss).

In `ingestion/parser.py::parse_pdf`, when `SECTION_HEADER_PATTERN` matches nothing (scanned PDF, unusual layout, non-standard headings), the fallback is:
```python
if not sections:
    full_text = _clean_text(p1_text)   # p1_text = page 1 ONLY
    sections.append(ParsedSection(section_name="Body", content=full_text, page_number=1))
```
`p1_text` is the text of **page 1 only** — every subsequent page is dropped. A multi-page paper whose headings don't fit the regex is silently reduced to its first page, so the vast majority of its content never enters FAISS/BM25/Neo4j. The sequential loop already read all pages; the fallback should concatenate all page texts, not reuse `p1_text`.

**Fix:** accumulate full document text across all pages during the main loop and use that for the fallback body.

---

## Issue 34 — Publication year hardcoded to 2024

**Severity:** Medium (data quality).

`parse_pdf` never extracts a year; `ParsedDocument.year` defaults to `2024` and the return explicitly passes `year=2024`. Every ingested document is stamped 2024 and `Neo4jClient.upsert_document(publication_year=doc.year)` persists that. This corrupts any temporal/recency reasoning — and note the evaluator's recency score keys off the literal string `"2024"`, so the hardcoded year quietly inflates recency. The docstring claims "publication year" extraction that does not exist. Parse the year from the first-page text / PDF metadata, or mark it unknown.

---

## Issue 35 — Title = first non-empty line of page 1 (fragile)

**Severity:** Low–Medium.

Title is taken as `p1_lines[0]`, the first non-empty line of page 1. For many PDFs that is a journal/arXiv banner, watermark, or page number rather than the title, with no font-size/position heuristic. A wrong title propagates: into chunk provenance headers, and into the extractor which uses `title.split(":")[0]` as a graph entity head (Issue 32), so a banner becomes a fabricated entity. Consider PDF metadata title or a largest-font-on-page-1 heuristic.

---

## Issue 37 — `upsert_document` call/signature mismatch silently kills the entire graph layer

**Severity:** High (confirmed — graph layer never populates, misreported as "offline").

`ingestion/service.py::ingest_files` calls:
```python
client.upsert_document(
    doc_id=doc.doc_id, title=doc.title, url=doc.file_path,
    authors=doc.authors, publication_year=doc.year,
)
```
But `memory/neo4j_client.py::upsert_document(doc_id, title="", authors="", year=None)` has **no `url` parameter and no `publication_year` parameter** (it takes `year`). Passing `url=` and `publication_year=` raises `TypeError: unexpected keyword argument` on the **first** document of every ingest. That exception is swallowed by the broad `try/except Exception` around the Neo4j block, which sets `neo4j_status="offline"`. Result: even when Neo4j is running and reachable, the graph layer is **never written** — and the failure is silently mislabeled as "Neo4j offline," so no one notices. The entire three-layer promise degrades to two layers (FAISS + BM25) in every deployment.

Additional mismatch: ingestion passes `authors=doc.authors` (a `List[str]`), while the signature annotates `authors: str`.

**Fix:** align the call and signature — add `url` and rename `year`↔`publication_year` consistently (and decide list-vs-string for authors). This is the highest-value single fix in the report: it silently disables a headline feature.

**Confirmed (cycle 22):** `ingestion/service.py` lines 322–324 pass `url=doc.file_path`, `authors=doc.authors`, `publication_year=doc.year`; `neo4j_client.upsert_document(doc_id, title="", authors="", year=None)` accepts none of `url`/`publication_year`. The sibling calls `upsert_passage_chunk`, `upsert_ontology_schema`, `upsert_fact_triple`, `link_fact_to_passage` were checked and their kwargs DO match their signatures — so the breakage is isolated to `upsert_document`, which is enough to abort the whole Neo4j `with` block on the first document.

---

## Issue 38 — `run_personalised_pagerank` is dead code (never called)

**Severity:** Low (dead code vs. spec).

`Neo4jClient.run_personalised_pagerank` implements the pure-Cypher PPR, but the retriever (`_retrieve_from_neo4j`) calls `serialise_subgraph_to_sentences` (a plain head/tail match), never `run_personalised_pagerank`. So the advertised PPR graph-diffusion step never runs in the live path — consistent with Issue 17 (`ppr_alpha`/`ppr_top_k` unused). Either wire PPR into retrieval or document that graph retrieval is a flat 1-hop lookup.

(Note: the PPR Cypher itself also computes `score` from a static `degree`/teleport formula with no iteration — it is a single-pass approximation, not true iterative PPR, despite the "iterative PPR" docstring.)

---

## Issue 39 — Test suite structurally cannot catch the broken graph layer (Issue 37)

**Severity:** Medium–High (coverage gap hiding a High bug).

`tests/test_live_ingestion.py::test_live_ingestion_end_to_end` asserts `res["status"] == "success"` and verifies FAISS/BM25 files, but runs with **no Neo4j** (no server, no mock). Because `ingest_files` swallows the `upsert_document` `TypeError` (Issue 37) into `neo4j_status="offline"`, the overall status is still `"success"` and the test passes — even though the graph layer never populated. The suite documents "Neo4j offline graceful degradation" as *expected*, and `test_ui_rendering_helpers` explicitly feeds `neo4j_status: "offline"`, so a permanently-broken graph layer is indistinguishable from a legitimately-absent DB.

Gaps:
- No test asserts `upsert_document`/`upsert_fact_triple` were called with **valid** arguments (a `MagicMock` Neo4jClient asserting call kwargs would have caught Issue 37 immediately).
- No test asserts graph triples are retrievable — the three-layer system has effectively zero graph-layer coverage.
- `neo4j_status` is never asserted to be `"synced"` in any end-to-end test, so regressions to "offline" are silent.

**Fix:** add an ingestion test with a mocked `Neo4jClient` that asserts the exact call signatures and that `neo4j_status == "synced"`.

---

## Issue 40 — Wipe test proves overwrite, not clearance

**Severity:** Low.

`test_complete_memory_wipe` checks `wipe_res["total_chunks"] == 0`, but `wipe_all` builds fresh empty `VectorStore`/`SparseStore` objects and `save()`s them without loading existing data, so the result is 0 by construction. The test never re-loads the on-disk index afterward to confirm the previously-ingested vectors are actually gone (`VectorStore(...).load_if_exists(); assert size == 0`). Strengthen the assertion to reload and verify zero vectors/documents.

---

_More findings appended in later patrol cycles._
