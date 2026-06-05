# External API Contracts (ExternalAPI.md)

## Overview

This document defines the strict data contracts and JSON parsing logic for the asynchronous external fallback loop in Project Aletheia. To minimize pipeline latency, all external search nodes execute their calls concurrently.

---

## 1. Crossref API Contract

- **Endpoint:** `https://api.crossref.org/works`
- **Protocol:** REST HTTP GET
- **Target Fields:** `["title", "abstract", "published", "author"]`

### Production Response Payload Sample

` ` `json
{
  "status": "ok",
  "message": {
    "total-results": 1420,
    "items": [
      {
        "DOI": "10.1145/3770855.3818074",
        "title": ["MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation"],
        "abstract": "<jats:p>Retrieval-Augmented Generation (RAG) has become an essential method for mitigating hallucinations...</jats:p>",
        "published": {
          "date-parts": [[2026, 8, 9]]
        },
        "author": [
          {
            "given": "Chuanjie",
            "family": "Wu",
            "sequence": "first"
          },
          {
            "given": "Jinsong",
            "family": "Su",
            "sequence": "additional"
          }
        ]
      }
    ]
  }
}
` ` `

### Strict Python Parsing Paths

When writing the Crossref ingestion engine, the code must extract elements using these explicit selectors:

- **Title Selector:** `item["title"][0]`
- **Abstract Selector:** `item.get("abstract", "")` *(Note: Crossref wraps abstracts in basic JATS XML tags like `<jats:p>`, which must be stripped or cleaned via standard regex).*
- **Year Selector:** `item["published"]["date-parts"][0][0]`
- **Author Selector String:** ` ` `python
  authors = ", ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])])
  ` ` `

---

## 2. OpenAlex API Contract

- **Endpoint:** `https://api.openalex.org/works`
- **Protocol:** REST HTTP GET
- **Target Fields:** `["title", "abstract_inverted_index", "publication_year", "authorships"]`

### Production Response Payload Sample

` ` `json
{
  "meta": {
    "count": 482
  },
  "results": [
    {
      "id": "https://openalex.org/W438721783",
      "title": "AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation",
      "publication_year": 2025,
      "abstract_inverted_index": {
        "While": [0],
        "RAG": [1, 10],
        "demonstrates": [2],
        "remarkable": [3],
        "capabilities": [4]
      },
      "authorships": [
        {
          "author": {
            "id": "https://openalex.org/A50392109",
            "display_name": "Yixiong Fang"
          }
        }
      ]
    }
  ]
}
` ` `

### Strict Python Parsing Paths

OpenAlex uses a highly unique **Inverted Index** format to protect intellectual property from bulk-scraping. Your parsing code *cannot* look for a clean text abstract string. It must reconstruct it dynamically:

- **Title Selector:** `result.get("title", "")`
- **Year Selector:** `result.get("publication_year")`
- **Author Selector String:**
  ` ` `python
  authors = ", ".join([a["author"]["display_name"] for a in result.get("authorships", [])])
  ` ` `
- **Abstract Reconstruction Algorithm:**
  ` ` `python
  inverted_index = result.get("abstract_inverted_index")
  if inverted_index:
      # Map token positions to words
      word_positions = {}
      for word, positions in inverted_index.items():
          for pos in positions:
              word_positions[pos] = word
      # Sort positions and reconstruct string
      abstract = " ".join([word_positions[i] for i in sorted(word_positions.keys())])
  else:
      abstract = ""
  ` ` `

---

## 3. Asynchronous Execution Matrix

To avoid blocking the LangGraph workflow thread, all external search workers must combine their payloads using `asyncio.gather`.

` ` `python

# Expected Orchestrator Branch Execution Flow

async def async_external_fallback_loop(query: str) -> List[Dict[str, Any]]:
    # Launch network queries concurrently across endpoints
    tasks = [
        fetch_crossref(query),
        fetch_openalex(query)
    ]
    crossref_results, openalex_results = await asyncio.gather(*tasks)

    # Deduplicate in staging memory by computing cross-source Title Similarity
    return merge_and_deduplicate(crossref_results, openalex_results)
` ` `
