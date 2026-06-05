# Product Requirements Document (PRD)

## Overview

This PRD proposes an autonomous, continuously updating conversational AI system that tracks, ingests, and explains the latest advancements in artificial intelligence.

The overall vision for the system is to eliminate thematic irrelevance, logical inconsistency, and structural fragmentation by utilizing a Three-Layer Global Memory architecture managed by a collaborative multi-agent society. The system dynamically cross-validates internal knowledge through dual-index keyword/vector scoring, supplements it with external academic APIs, and applies an attention-guided context pruning mechanism to produce highly verifiable, domain-specific technical reasoning without suffering from context window bloat.

## Goals

- Autonomously ingest and track rapidly changing AI model architectures without requiring users to manually read dozens of papers.
- Provide highly verifiable, domain-specific technical reasoning.
- Cross-validate internal knowledge dynamically through dual-index keyword/vector scoring to ensure high relevance.
- Supplement internal knowledge seamlessly with external academic APIs when local evidence is deemed weak.
- Minimize computational overhead and prevent attention dilution by compressing retrieved contexts by up to 6.3x prior to final answer generation.
- Deliver transparent auditing capabilities by tracking all agentic LLM calls, exact token counts, financial costs, and execution times.

## Non-Goals

- Answering general trivia or providing information outside the domain of artificial intelligence and computer science research.
- Executing, compiling, or locally running the AI models described in the ingested literature.
- Extracting and processing multimodal visual data, such as charts, graphs, or complex technical diagrams embedded within academic PDFs.

## Audience

The primary audience for this system consists of AI Engineers, Computer Science researchers, and developers. These users need to stay updated on rapidly changing model architectures without manually reading dozens of papers a week.

A secondary audience includes computer science students specializing in AI who are preparing for graduate-level research, as well as technical writers looking for synthesized, accurate summaries of complex AI concepts with direct source citations.

The majority of these users want an engine that allows them to rapidly navigate a large body of literature with confidence in source attribution. They are heavily focused on research-grade accuracy and require verifiable references to support technical discussions and system design.

## Existing solutions and issues

While there are existing retrieval methods for language models, they present critical limitations for specialized engineering domains:

- **Vanilla RAG Systems**
  - These systems struggle with large-scale, unstructured corpora where information is highly fragmented.
  - They rely heavily on surface-level keyword matching, which often overlooks the logical bridges required for multi-hop reasoning.
- **Traditional GraphRAG Methods**
  - These methods rely on isolated, fragment-level extraction for graph construction, lacking a global perspective on the whole corpus.
  - This isolated approach frequently leads to thematically inconsistent, logically conflicting, and structurally fragmented graphs that degrade retrieval performance.
- **Long Context Dilution**
  - As retrieved document chunks accumulate, the context window becomes excessively long, introducing redundant information.
  - Attention scores spread too thin across thousands of tokens, causing the model to lose focus on relevant details and suffer degraded reasoning capabilities.
- **General-Purpose Standalone LLMs**
  - These models exhibit fundamental limitations related to domain specificity, knowledge freshness, context scalability, and traceability.
  - Generated responses are typically uncited, non-verifiable, and difficult to trace back to authoritative sources, increasing the risk of hallucinated information.

## Assumptions

- Target users are working in research-heavy product development or academic environments and need to connect technical literature to generate evidence-backed insights.
- Users require clear provenance and verifiable references to build confidence in engineering decisions derived from LLM-generated content.
- Users will pose distinct categories of queries, including content questions regarding technical methodology, bibliometric queries about specific authors, and trend discovery mapping landscape evolutions.
- Users prefer having the capability to run inference loops via local AI hosting tools—such as Ollama utilizing models like Qwen—for local development and coding assistance to ensure data privacy before scaling to cloud APIs.

## Constraints

The system must be a stable, precise, and auditable architecture. We need to highlight a few constraints of the product:

- The system's orchestration and multi-agent loops must be built using Python and LangGraph.
- The system must use FAISS (IndexFlat IP, cosine similarity) and a persistent, disk-backed serialization architecture or an optimized Lucene engine for scaling dual-indexing safely to 10M+ records.
- The knowledge graph must be implemented using Neo4j.
- The embedding model is restricted to `all-MiniLM-L6-v2` (384 dimensions) and the cross-encoder to `ms-marco-MiniLM-L-6-v2`.
- External literature retrieval must rely on concurrent, asynchronous connections to Crossref, OpenAlex, and Semantic Scholar APIs.
- Document parsing and text extraction must be performed using PyMuPDF (`fitz`).

## Key use cases

- Autonomous Knowledge Ingestion & Graph Construction
- Memory-Guided Retrieval & Cross-Validation
- Evidence Evaluation & External Search
- Attention-Guided Context Compression
- Self-Correcting Answer Generation

### Autonomous Knowledge Ingestion & Graph Construction

The system uses layout markers (e.g., Abstract, Introduction, Methods) to ensure chunks strictly respect section and paragraph boundaries. An Extraction Agent ($A_{ext}$) processes the structured chunks to simultaneously populate an Ontology Layer ($M_{ont}$), Fact Layer ($M_{fac}$), and Passage Layer ($M_{pas}$). A Conflict Detection Agent ($A_{det}$) asynchronously scans for Mutually Exclusive, Temporal, and Granularity conflicts, while a Conflict Resolution Agent ($A_{res}$) accesses raw evidence to resolve these discrepancies. The refined memory layers are then projected into a Neo4j Hierarchical Indexing Graph using Type-based and Similarity-based bridging to connect isolated subgraphs.

### Memory-Guided Retrieval & Cross-Validation

An LLM determines the optimal query route (content, bibliometric, trend, current_world) and translates user intent into normalized technical terminology. The system concurrently queries the Hierarchical Graph and the Dual-Index (FAISS + persistent/serialized Lexical index via Reciprocal Rank Fusion). To guarantee precise re-ranking with text-based Cross-Encoder models, abstract graph structures, topological entities, and facts are explicitly serialized into descriptive natural language sentences prior to merging. The graph diffuses initial importance via Personalized PageRank (PPR) to identify top passages and entities. These results are strictly merged with the structural lexical/vector outputs and finalized through a Cross-Encoder reranker to validate absolute contextual precision.

### Evidence Evaluation & External Search

The cross-validated retrieved context is evaluated using a 100-Point Evidence Sufficiency Scoring rubric across five dimensions: Retrieval Confidence, Answer Specificity, Source Diversity, Metadata Completeness, and Recency. If internal evidence scores below the "Weak" threshold, the system autonomously reformulates the query and strictly enforces a term overlap guardrail to prevent semantic drift. For unresolved weak scores, the system launches optimize-search-vet iterative loops across external academic APIs in parallel to minimize network latency. To address high recency needs or data that has not yet reached cross-document consensus, the multi-layer filtering engine implements an instant cold-start fallback route allowing direct lookup into the probationary "Pending" staging memory layer.

### Attention-Guided Context Compression

Before passing the massive block of retrieved local chunks and external abstracts to the final generation model, the system dynamically prunes the context. The system reformulates the user's query into an "answer hint prefix" conforming to a next-token-prediction format. To prevent attention dilution over large source chunks, the context is dynamically segmented into micro-windows (50–300 tokens). The system handles processing via a heavily optimized int4-quantized local model, computes the attention features connecting the micro-contexts to this focal token, and safely discards sentences that yield low attention scores, creating a highly compressed but semantically dense context prompt.

### Self-Correcting Answer Generation

The system merges the compressed memory-derived facts, externally vetted abstracts, and local passage sentences into a structured LLM prompt. Prior to generation, an LLM analyzes the assembled prompt for logical gaps or direct contradictions among the integrated sources. The final conversational answer is generated and subsequently audited by a secondary LLM checker; if hallucinations or uncited claims are detected, the system triggers a single, automatic regeneration.

## Research

### User Research

**Why is strict citation traceability critical to our users?**

In safety-relevant or domain-critical engineering workflows, generated responses must be grounded in verified evidence. Without clear provenance, it becomes challenging to assess the validity of assumptions, reproduce results, compare methodologies, or build confidence in engineering decisions derived from LLM-generated content. The absence of source grounding inherently increases the risk of hallucinated or conflated information. Consequently, our users demand built-in citation verification and answer quality assessment to ensure grounded analysis over authoritative sources while preserving absolute transparency.

### Technical Research

**How does structure-aware node initialization improve retrieval accuracy over traditional GraphRAG?**

Traditional retrieval blindly activates nodes based on simple keyword overlaps. Our system filters graph nodes and assigns initial reset probabilities based on deep semantic relevance and structural constraints.

- Entity Initialization is calculated as the mean semantic similarity of query-relevant facts: $P_{init}(e) = \frac{1}{|\mathcal{F}_{e}|}\sum_{f\in \mathcal{F}_{e}}Sim(q, f)$.
- Type Initialization incorporates schema relevance while applying a hub suppression penalty to prevent generic nodes from dominating the propagation: $P_{init}(t) = (\frac{1}{|\mathcal{S}_{t}|}\sum_{s\in\mathcal{S}_{t}}Sim(q,s))\times\frac{1}{\log(deg(t)+1)}$.
- Passage Initialization combines semantic alignment with an information density prior using Inverse Document Frequency (IDF): $P_{init}(p) = Sim(q,d_{p})\times\alpha\times\sigma(\frac{\sum_{e\in\mathcal{E}_{p}}IDF(e)}{\log(|\mathcal{E}_{p}|+1)})$.

**How does AttentionRAG prevent context bloat without losing critical data?**

Instead of using question-unaware compression or arbitrary text truncation, the system utilizes an Attention Focus Mechanism. By passing micro-window inputs through an int4-quantized model and summing the attention scores of a specific generated focal token across all model layers, the system successfully captures both syntax (from shallow layers) and deep semantics (from deeper layers). This calculation allows the system to accurately rank and retain only the specific sentences containing the top-K attended tokens, achieving up to 6.3x compression while maintaining or actively improving the final generation accuracy.

**Why incorporate support for local inference tools like Ollama?**

Relying entirely on premium cloud endpoints (e.g., GPT-4o) for the entirety of the multi-agent loop can become cost-prohibitive during high-volume ingestion and testing. By constraining the infrastructure to support local AI hosting tools, developers can run open-weight models (like Qwen) locally. This effectively eliminates external API dependence for low-tier reasoning tasks, reduces per-query testing costs to zero, and safely addresses data privacy concerns when processing proprietary academic corpora.
