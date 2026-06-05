# Design: Project Aletheia - Consolidated Overview

## 1. Introduction & High-Level Architecture

  Project Aletheia is an autonomous, continuously updating conversational AI system
  designed to track, ingest, and explain advancements in artificial intelligence. It
  employs a **Three-Layer Global Memory architecture** managed by a collaborative
  **multi-agent society** orchestrated by **LangGraph**. The system aims to eliminate
  thematic irrelevance, logical inconsistency, and structural fragmentation by utilizing a
  robust memory structure, precise retrieval, and attention-guided context compression.

  The system is divided into:

* **Offline Ingestion Sandbox (MemGraphRAG)**: Processes raw PDFs, extracts facts,
  resolves conflicts, and populates memory.
* **Online Orchestration (LangGraph)**: Manages query routing, retrieval, evidence
  evaluation, external search, context compression, and answer generation.
* **Attention Compressor (AttentionRAG)**: Prunes context to prevent attention
  dilution.
* **Storage**: Heterogeneous data layer (Neo4j, FAISS, Disk-Backed BM25).
* **External Academic APIs**: Integration with Crossref, OpenAlex, Semantic Scholar.  
* **LLM Engine**: Supports local (Ollama) and cloud APIs.
* **User Interface**: Gradio web interface.

## 2. Sub-Project Designs

### 2.1. Offline Ingestion & Graph Construction

  **Objective**: Process raw PDFs into a conflict-free, structured global memory.

* **PDF Parsing & Chunking**: Uses `PyMuPDF (fitz)` for text extraction and
  structure-aware chunking (~3500 chars), respecting section/paragraph boundaries. Chunks
  include metadata like `chunk_id`, `document_id`, `section_label`, `page_number`,
  `content`, `url`.
* **Extraction Agent (`A_ext`)**: Identifies `ONTOLOGY_SCHEMA` (initially "Pending",  
  becomes "Stable" based on frequency $\tau$) and `FACT_TRIPLE`s from chunks using a local
  LLM. Populates Neo4j.
* **Conflict Management**:
  * **Detection (`A_det`)**: Asynchronously scans for mutual exclusivity, temporal,
  and granularity conflicts.
  * **Resolution (`A_res`)**: Uses raw evidence and LLM to correct or merge
  conflicting claims.
* **Memory-Bridging**: Connects subgraphs in Neo4j via shared schema types and
  semantic vector similarity.
* **Persistence & Indexing**:
  * Neo4j: Native disk persistence.
  * FAISS: 384-dim embeddings (`all-MiniLM-L6-v2`) serialized to disk.
  * BM25: Disk-backed index (`rank_bm25`, `ms-marco-MiniLM-L-6-v2` for re-ranking).

### 2.2. Memory-Guided Retrieval

  **Objective**: Fetch structurally sound and semantically relevant data from memory
  stores.

* **Query Router & Reformulator**: Classifies intent (`content`, `bibliometric`,
  `trend`, `current_world`) and reformulates queries for optimal retrieval, stripping
  filler and adding technical jargon. Output: JSON with `route` and `reformulated_query`.
* **Retrieval Strategy**:
  * **Parallel retrieval**: Neo4j (PPR with structure-aware initialization), FAISS  
  (dense vector search), BM25 (sparse lexical search).
  * **Dual-Index Fusion**: Merges FAISS and BM25 results using Reciprocal Rank
  Fusion (RRF).
  * **Cross-Encoder Re-ranking**: Final re-ranking using `ms-marco-MiniLM-L-6-v2`.  
* **Output**: Serialized context strings passed to the Evidence Evaluator.

### 2.3. Evidence Evaluation & External Search

  **Objective**: Audit retrieved context and supplement with external data if
  insufficient.

* **Evidence Evaluator & Scorer**: Audits context against a 100-Point Rubric
  (Retrieval Confidence, Specificity, Diversity, Metadata Completeness, Recency). Sets
  `is_evidence_weak` flag if `rubric_score < 80`.
* **External Search Trigger**: If `is_evidence_weak` is true, the Async External
  Search Agent is invoked.
* **Fallback Query Reformulation**: Refines queries using relevance damping (`damping
  = max(min(retrieval_score / 25, 1.0), 0.2)`) and a drift guard ($\ge 30\%$ term
  overlap).
* **Parallel API Calls**: Concurrently queries Crossref, OpenAlex, and Semantic
  Scholar APIs.
* **Output**: Aggregated abstracts added to `external_api_abstracts` in `MemoryState`.

### 2.4. Attention-Guided Context Compression

  **Objective**: Dynamically prune context to reduce token overhead and prevent attention
  dilution.

* **Component**: Attention-Guided Context Compressor.
* **Logic**:
  * Uses user query to generate an "answer hint prefix" and focal token.
  * Segments context into micro-windows (50-300 tokens).
  * Uses an int4-quantized local LLM to calculate attention scores between
  micro-windows and the focal token.
  * Prunes sentences with low attention scores.
* **Compression Ratio**: Aims for up to 6.3x context reduction.
* **Output**: `final_pruned_prompt` stored in `CompressionState`.

### 2.5. Self-Correcting Answer Generation

  **Objective**: Generate accurate, cited answers with a self-correction mechanism.

* **Generator**: Synthesizes `draft_answer` from pruned context and user query.
* **Critic Agent**: Audits draft for citation presence and hallucinations against
  pruned context. Outputs `passed` (bool) and `feedback` (str/null).
* **Self-Correction Loop**:
  * If Critic's `passed` is false and `retry_count < 1`, Generator retries with
  feedback.
  * `retry_count` is incremented.
* **State**: `draft_answer`, `critic_feedback`, `retry_count` stored in
  `GenerationState`.

### 2.6. User Interface (Gradio)

  **Objective**: Provide an interactive interface for users to query the system.

* **Component**: `ui/app.py` (Gradio application).
* **Functionality**:
  * Text input for user queries.
  * Real-time pipeline status indicators.
  * Display of running LLM costs and latency.
  * Expandable sections for context sources (internal memory, external APIs, pruned
  context).
  * Final answer with citations.
* **Interaction Flow**: User query -> LangGraph orchestrator -> UI updates with status
  and results.

## 3. Common Development Tasks & Project Structure

* **Dependencies**: Install via `pip install -r requirements.txt` (if exists) or
  manually install `langchain`, `pydantic`, `neo4j`, `faiss-cpu`, `rank_bm25`, `pymupdf`,
  `transformers`, `gradio`.
* **Running UI**: Execute `python ui/app.py`.
* **Local LLM**: Configure Ollama for local inference (e.g., Qwen).
* **Environment Variables**: Manage API keys via `.env` file.
* **Testing**: Retrieval metrics (Precision@K, Recall@K, MRR), LLM-as-a-Judge for
  generation, unit tests for LangGraph nodes.
* **Project Structure**: Adheres to `zero-hallucination-rag/` directory structure as  
  detailed in `Structure.md`.
