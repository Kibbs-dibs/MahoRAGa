"""
Entity and Fact Extractor for Project Aletheia.

Performs LLM-based knowledge extraction with a deterministic regex fallback
(used when no LLM backend is reachable, e.g. the offline stub), producing:
1. ONTOLOGY_SCHEMA (HeadType, Relation, TailType)
2. FACT_TRIPLE (HeadEntity, Relation, TailEntity, SimilarityWeight)
3. Provenance associations linking facts back to origin PASSAGE_CHUNK ids.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from core.llm import llm_client
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Canonical domain relations for AI literature
CANONICAL_RELATIONS = {
    "proposes": ("Model", "Methodology"),
    "introduces": ("Paper", "Model"),
    "utilizes": ("Model", "Mechanism"),
    "uses": ("Model", "Component"),
    "evaluates": ("Methodology", "Benchmark"),
    "achieves": ("Model", "PerformanceMetric"),
    "compresses": ("Compressor", "Context"),
    "outperforms": ("Model", "Baseline"),
    "consists_of": ("Architecture", "Subcomponent"),
    "relies_on": ("Algorithm", "Assumption"),
    "retrieves": ("Retriever", "Passage"),
}


class ExtractedSchema(BaseModel):
    """Schema constraining fact triples in the Neo4j ontology layer."""
    schema_id: str
    head_type: str
    relation: str
    tail_type: str
    status: str = "Stable"
    frequency: int = 1


class ExtractedFact(BaseModel):
    """Factual knowledge triple extracted from an academic passage chunk."""
    fact_id: str
    head_entity: str
    relation: str
    tail_entity: str
    similarity_weight: float = 1.0
    schema_id: Optional[str] = None
    chunk_id: str = ""
    natural_language_sentence: str = ""


class ExtractionResult(BaseModel):
    """Container holding all extracted schemas and facts for an ingested document."""
    doc_id: str
    schemas: List[ExtractedSchema] = Field(default_factory=list)
    facts: List[ExtractedFact] = Field(default_factory=list)


def _generate_id(prefix: str, content: str) -> str:
    """Generate a stable deterministic ID from content."""
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge-graph extraction agent for AI research literature.
Read the passage and extract factual subject-relation-object triples.

CONSTRAINTS:
- The "relation" MUST be one of the allowed relations exactly as written: {relations}.
- head_entity and tail_entity must be concrete named entities or concepts drawn from the passage (never pronouns like "we"/"it").
- Only extract triples explicitly supported by the passage. Do NOT invent facts.

OUTPUT FORMAT:
Respond with a raw JSON object of the form:
{{"facts": [{{"head_entity": "...", "relation": "<one of the allowed relations>", "tail_entity": "..."}}]}}
Return an empty list if no reliable triples are present."""


def _build_fact(head: str, rel: str, tail: str, chunk: Chunk) -> ExtractedFact:
    """Construct an ExtractedFact with deterministic id and schema mapping."""
    fact_key = f"{head}_{rel}_{tail}"
    fact_id = _generate_id("fact", fact_key)
    schema_id = f"schema_{rel.upper()}"
    nl_sentence = f"{head} {rel.replace('_', ' ')} {tail}."
    return ExtractedFact(
        fact_id=fact_id,
        head_entity=head,
        relation=rel.upper(),
        tail_entity=tail,
        similarity_weight=1.0,
        schema_id=schema_id,
        chunk_id=chunk.chunk_id,
        natural_language_sentence=nl_sentence,
    )


def _extract_via_llm(chunk: Chunk) -> List[ExtractedFact]:
    """
    Attempt LLM-based triple extraction for a single chunk.

    Returns a list of ExtractedFact objects parsed from the model's JSON output,
    or an empty list when no usable JSON is returned (e.g. under the offline
    deterministic stub, which does not emit extraction JSON). Callers treat an
    empty result as a signal to fall back to the regex extractor.
    """
    allowed = ", ".join(sorted(CANONICAL_RELATIONS.keys()))
    system_prompt = EXTRACTION_SYSTEM_PROMPT.format(relations=allowed)

    try:
        resp = llm_client.call(
            messages=[{"role": "user", "content": f"<PASSAGE>\n{chunk.content}\n</PASSAGE>"}],
            system_prompt=system_prompt,
            temperature=0.0,
            json_output=True,
        )
    except Exception as exc:  # pragma: no cover - defensive, offline path never raises
        logger.debug("LLM extraction call failed for chunk %s: %s", chunk.chunk_id, exc)
        return []

    parsed = resp.parsed_json
    if not isinstance(parsed, dict):
        return []

    raw_facts = parsed.get("facts")
    if not isinstance(raw_facts, list):
        return []

    facts: List[ExtractedFact] = []
    seen_pairs: Set[Tuple[str, str, str]] = set()
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        head = str(item.get("head_entity", "")).strip()
        rel = str(item.get("relation", "")).strip().lower()
        tail = str(item.get("tail_entity", "")).strip()

        # Constrain the relation to the canonical vocabulary.
        if rel not in CANONICAL_RELATIONS:
            continue
        if len(head) < 2 or len(tail) < 2:
            continue

        key = (head.lower(), rel, tail.lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        facts.append(_build_fact(head, rel, tail, chunk))

    return facts


def extract_from_chunk(chunk: Chunk) -> List[ExtractedFact]:
    """
    Extract technical facts from a single passage chunk.

    Uses LLM-based triple extraction when an LLM backend is available, and
    falls back to deterministic regex patterns when the LLM returns no usable
    JSON (which is exactly what happens under the offline deterministic stub).
    """
    llm_facts = _extract_via_llm(chunk)
    if llm_facts:
        return llm_facts

    # ── Regex fallback: pattern-based [Subject] [verb phrase] [Object] ──
    facts: List[ExtractedFact] = []
    text = chunk.content

    # Pattern: [Subject] [verb phrase] [Object]
    relation_patterns = [
        (r"\b([A-Z][A-Za-z0-9_-]+)\s+(?:proposes|introduced|introduces)\s+([A-Za-z0-9_\s-]{3,40}?)(?=[,.]|\s+in\b|\s+to\b)", "proposes"),
        (r"\b([A-Z][A-Za-z0-9_-]+)\s+(?:utilizes|uses|leverages)\s+([A-Za-z0-9_\s-]{3,40}?)(?=[,.]|\s+for\b|\s+to\b)", "utilizes"),
        (r"\b([A-Z][A-Za-z0-9_-]+)\s+(?:achieves|achieved)\s+([A-Za-z0-9_\s-]{3,40}?)(?=[,.]|\s+on\b|\s+with\b)", "achieves"),
        (r"\b([A-Z][A-Za-z0-9_-]+)\s+(?:outperforms|surpasses)\s+([A-Za-z0-9_\s-]{3,40}?)(?=[,.]|\s+by\b)", "outperforms"),
        (r"\b([A-Z][A-Za-z0-9_-]+)\s+(?:compresses|prunes)\s+([A-Za-z0-9_\s-]{3,40}?)(?=[,.]|\s+by\b|\s+to\b)", "compresses"),
        (r"\b([A-Z][A-Za-z0-9_-]+)\s+(?:consists of|is composed of)\s+([A-Za-z0-9_\s-]{3,40}?)(?=[,.])", "consists_of"),
    ]

    seen_pairs: Set[Tuple[str, str, str]] = set()

    for pattern, rel in relation_patterns:
        for match in re.finditer(pattern, text):
            head = match.group(1).strip()
            tail = match.group(2).strip()

            # Filter trivial matches
            if len(head) < 2 or len(tail) < 3 or len(tail) > 50:
                continue
            if head.lower() in ("we", "they", "this", "that", "these", "it", "our"):
                head = chunk.doc_title.split(":")[0].strip()

            key = (head.lower(), rel, tail.lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            facts.append(_build_fact(head, rel, tail, chunk))

    return facts


def extract_knowledge(doc_id: str, chunks: List[Chunk]) -> ExtractionResult:
    """
    Extract all knowledge graph schemas and facts from a document's chunks.

    Uses LLM-based triple extraction per chunk with a deterministic regex
    fallback (see :func:`extract_from_chunk`), then de-duplicates facts and
    derives the ontology schema layer.

    Args:
        doc_id: Identifier of the document
        chunks: List of Chunk objects from chunker

    Returns:
        ExtractionResult with unique schemas and facts
    """
    all_facts: List[ExtractedFact] = []
    seen_fact_ids: Set[str] = set()
    schemas_dict: Dict[str, ExtractedSchema] = {}

    for chunk in chunks:
        chunk_facts = extract_from_chunk(chunk)
        for f in chunk_facts:
            if f.fact_id not in seen_fact_ids:
                seen_fact_ids.add(f.fact_id)
                all_facts.append(f)

                # Track schema
                schema_id = f.schema_id or "schema_GENERAL"
                rel_lower = f.relation.lower()
                head_type, tail_type = CANONICAL_RELATIONS.get(rel_lower, ("Concept", "Property"))

                if schema_id not in schemas_dict:
                    schemas_dict[schema_id] = ExtractedSchema(
                        schema_id=schema_id,
                        head_type=head_type,
                        relation=f.relation,
                        tail_type=tail_type,
                        status="Stable",
                        frequency=1,
                    )
                else:
                    schemas_dict[schema_id].frequency += 1

    logger.info(
        "Extracted %d unique facts and %d ontology schemas for document %s",
        len(all_facts),
        len(schemas_dict),
        doc_id,
    )

    return ExtractionResult(
        doc_id=doc_id,
        schemas=list(schemas_dict.values()),
        facts=all_facts,
    )
