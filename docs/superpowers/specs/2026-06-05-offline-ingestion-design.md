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
│   ├── config.py           # Pydantic BaseSettings (API keys, thresholds)
│   ├── state.py            # The Pydantic models from State.md
│   └── exceptions.py       # Custom error classes
├── graph/
│   ├── orchestrator.py     # The compiled LangGraph StateGraph
│   ├── nodes/              # Individual LangGraph node functions (router, retriever, evaluator, search_apis, pruner, generator)
├── memory/
│   ├── neo4j_client.py     # Graph database connection
│   ├── vector_store.py     # FAISS implementation
│   └── sparse_store.py     # Disk-backed BM25 implementation
├── ui/
│   └── app.py              # Gradio interface
├── .env                    # Environment variables
└── requirements.txt
```

## Common Development Tasks

Since no `requirements.txt` or explicit build scripts are present, the following are inferred based on the project documentation:

* **Dependency Installation**: You would typically install Python dependencies using a `requirements.txt` file. If one exists, use `pip install -r requirements.txt`. Otherwise, individual dependencies (like `langchain`, `pydantic`, `neo4j`, `faiss-cpu`, `rank_bm25`, `pymupdf`, `transformers`, `gradio`) would need to be installed manually.
* **Running the UI**: The Gradio interface is likely the primary entry point for interaction. To run the UI, you would execute the `app.py` file within the `ui/` directory: `python ui/app.py`.
* **Local LLM Setup**: For local development and privacy, the system supports local inference endpoints like Ollama. This would involve setting up Ollama and configuring the system to use a local model (e.g., Qwen). Refer to Ollama documentation for specific setup instructions.
* **Environment Variables**: API keys for cloud LLMs (OpenAI, Semantic Scholar) and other configurations are managed via `.env` files. Ensure `.env` is correctly configured before running the system.
* **Testing**: The `SystemDesign.md` mentions a testing strategy involving:
  * **Retrieval Metrics**: Offline evaluation using Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
  * **Generation Testing (LLM-as-a-Judge)**: Leveraging GPT-4o-mini in a CI/CD pipeline to evaluate pipeline outputs for hallucination and citation adherence.
  * **Agent Flow Testing**: Unit testing LangGraph nodes individually, mocking API responses.

    Specific commands for running these tests are not provided, but they would typically involve a Python testing framework like `pytest`.

# Design: Offline Ingestion & Graph Construction (Project Aletheia)

## 1. PDF Parsing and Structure-Aware Chunking

**Architecture & Components**:

* Part of the Offline Ingestion Sandbox.
* Utilizes `PyMuPDF (fitz)` for parsing raw PDF documents.

**Logic & Data Flow**:

* PDFs are parsed to extract text and structural elements (headings, paragraphs).
* Structure-Aware Chunking logic segments text into ~3500 character chunks, respecting section/paragraph boundaries.

**Output Format**: Chunks include `chunk_id`, `document_id`, `section_label`, `page_number`, `content`, `url`. Conforms to `PASSAGE_CHUNK` schema.

## 2. Extraction Agent and Initial Memory Population

**Architecture & Components**:

* **Extraction Agent (`A_ext`)**: Processes structured chunks to populate memory layers.
* Uses a local LLM for low-tier reasoning.

**Logic & Data Flow**:

* Identifies `ONTOLOGY_SCHEMA` and `FACT_TRIPLE`s from chunks.
* `ONTOLOGY_SCHEMA`s start as "Pending" and become "Stable" based on frequency threshold ($\tau$).
* `FACT_TRIPLE`s include `head_entity`, `relation`, `tail_entity`, `similarity_weight`, and provenance.
* Populates Neo4j with schemas, triples, and passage links.

**Output**: Structured data for Neo4j.

## 3. Conflict Detection, Resolution, and Memory-Guided Bridging

**Architecture & Components**:

* **Conflict Detection Agent (`A_det`)**: Scans for conflicts (mutual exclusivity, temporal, granularity).
* **Conflict Resolution Agent (`A_res`)**: Uses raw evidence (`PASSAGE_CHUNK`s) and LLM to resolve conflicts.

**Logic & Data Flow**:

* `A_det` flags conflicts in Neo4j data.
* `A_res` evaluates raw text to correct or merge conflicting claims.
* Memory-Guided Bridging connects subgraphs via shared schema types and semantic vector similarity.

**Storage Population**:

* **Neo4j**: Populated with refined schemas, triples, and passages.
* **FAISS**: Stores 384-dim embeddings (`all-MiniLM-L6-v2`) of chunks, serialized to disk.
* **BM25**: Disk-backed index using `rank_bm25` for lexical retrieval.

## 4. Data Persistence and Indexing

**Architecture & Components**:

* Neo4j Client, FAISS, Disk-Backed BM25.

**Logic & Data Flow**:

* **Neo4j**: Native disk-based persistence.
* **FAISS**: Index trained and serialized to disk.
* **BM25**: Index persisted automatically via disk-backing.

**Persistence Strategy**: Ensures data is durably stored and efficiently indexed for retrieval.
