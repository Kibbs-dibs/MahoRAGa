"""
Entity and Fact Extractor for Project Aletheia.

Extracts:
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

# NOTE: extraction here is a regex-based heuristic (not LLM-based). The previous
# `from core.llm import llm_client` import was unused and has been removed (Issue 32).
# If LLM-based extraction is reintroduced, import llm_client where it is actually used.
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


def extract_from_chunk(chunk: Chunk) -> List[ExtractedFact]:
    """Extract technical facts from a chunk.

    Uses the LLM when ``settings.llm_fact_extraction`` is enabled (Feature 6.1),
    falling back to the regex heuristic on failure or offline.
    """
    from core.config import settings as _settings
    if getattr(_settings, "llm_fact_extraction", False):
        try:
            llm_facts = _extract_from_chunk_llm(chunk)
            if llm_facts:
                return llm_facts
        except Exception as exc:
            logger.warning(f"LLM fact extraction failed, using regex fallback: {exc}")
    return _extract_from_chunk_regex(chunk)


def _extract_from_chunk_llm(chunk: Chunk) -> List[ExtractedFact]:
    """LLM-based structured triple extraction (Feature 6.1)."""
    import json as _json
    from core.llm import llm_client

    # Strip provenance header before sending to the LLM.
    text = re.sub(r"^\s*\[[^\]]*\|[^\]]*\|[^\]]*\]\s*\n?", "", chunk.content, count=1)
    if len(text.strip()) < 40:
        return []

    prompt = (
        "Extract factual knowledge triples (head_entity, relation, tail_entity) from the "
        "passage below. Relations should be short verbs/phrases (e.g. proposes, uses, "
        "outperforms, evaluates). Return ONLY a JSON array of objects with keys "
        "head_entity, relation, tail_entity. Max 8 triples.\n\nPassage:\n" + text[:2000]
    )
    resp = llm_client.call(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="You extract knowledge-graph triples as strict JSON. Output only JSON.",
        temperature=0.1,
        max_tokens=400,
        json_output=True,
    )

    raw = resp.parsed_json if resp else None
    # Accept either a bare list or {"triples": [...]}.
    if isinstance(raw, dict):
        raw = raw.get("triples") or raw.get("facts") or next((v for v in raw.values() if isinstance(v, list)), None)
    if not isinstance(raw, list):
        return []

    facts: List[ExtractedFact] = []
    seen: Set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        head = str(item.get("head_entity", "")).strip()
        rel = str(item.get("relation", "")).strip()
        tail = str(item.get("tail_entity", "")).strip()
        if not (head and rel and tail) or len(tail) > 80:
            continue
        key = f"{head.lower()}_{rel.lower()}_{tail.lower()}"
        if key in seen:
            continue
        seen.add(key)
        rel_norm = rel.upper().replace(" ", "_")
        facts.append(ExtractedFact(
            fact_id=_generate_id("fact", key),
            head_entity=head,
            relation=rel_norm,
            tail_entity=tail,
            similarity_weight=1.0,
            schema_id=f"schema_{rel_norm}",
            chunk_id=chunk.chunk_id,
            natural_language_sentence=f"{head} {rel} {tail}.",
        ))
    return facts


def _extract_from_chunk_regex(chunk: Chunk) -> List[ExtractedFact]:
    """
    Extract technical facts from a single passage chunk using NLP patterns.
    """
    facts: List[ExtractedFact] = []
    text = chunk.content

    # Strip the leading provenance header "[Title | Section | Page]\n" that the
    # chunker prepends, so the capitalized-head regexes do not mine spurious facts
    # out of the header line (Issue 33).
    text = re.sub(r"^\s*\[[^\]]*\|[^\]]*\|[^\]]*\]\s*\n?", "", text, count=1)

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

            fact_key = f"{head}_{rel}_{tail}"
            fact_id = _generate_id("fact", fact_key)
            schema_id = f"schema_{rel.upper()}"
            nl_sentence = f"{head} {rel.replace('_', ' ')} {tail}."

            facts.append(
                ExtractedFact(
                    fact_id=fact_id,
                    head_entity=head,
                    relation=rel.upper(),
                    tail_entity=tail,
                    similarity_weight=1.0,
                    schema_id=schema_id,
                    chunk_id=chunk.chunk_id,
                    natural_language_sentence=nl_sentence,
                )
            )

    return facts


def extract_knowledge(doc_id: str, chunks: List[Chunk]) -> ExtractionResult:
    """
    Extract all knowledge graph schemas and facts from a document's chunks.

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
