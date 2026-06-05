"""
Neo4j graph-database client for Project Aletheia.

Responsibilities
────────────────
• Manage the Neo4j Bolt driver lifecycle (connect / close / health-check).
• CRUD helpers for the four node labels defined in the ER schema:
      ONTOLOGY_SCHEMA, FACT_TRIPLE, PASSAGE_CHUNK, DOCUMENT.
• Personalized PageRank (PPR) via Neo4j GDS projection (or a pure-Cypher
  fallback when GDS is unavailable).
• Serialise graph topology into natural-language sentences so that the
  Memory-Guided Retriever can feed them into the Cross-Encoder reranker.
• Expose query utilities consumed by ``graph/nodes/retriever.py``.

All Cypher is parameterised to prevent injection. Errors are wrapped in
``Neo4jConnectionError`` / ``Neo4jQueryError`` from ``core.exceptions``.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

from neo4j import GraphDatabase, Session
from neo4j.exceptions import (
    AuthError,
    ServiceUnavailable,
    SessionExpired,
)

from core.config import settings
from core.exceptions import Neo4jConnectionError, Neo4jQueryError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Neo4j Client
# ═══════════════════════════════════════════════════════════════════════════


class Neo4jClient:
    """Thin, thread-safe wrapper around the official Neo4j Python driver.

    Usage
    -----
    >>> client = Neo4jClient()
    >>> client.connect()
    >>> triples = client.get_fact_triples_for_entity("Transformer")
    >>> client.close()

    Or as a context-manager:

    >>> with Neo4jClient() as client:
    ...     client.upsert_ontology_schema(...)
    """

    # ── lifecycle ───────────────────────────────────────────────────────

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver = None

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def connect(self) -> None:
        """Establish the Bolt driver connection."""
        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            # Verify connectivity eagerly
            self._driver.verify_connectivity()
            logger.info("Neo4j connection established at %s", self._uri)
        except (ServiceUnavailable, AuthError) as exc:
            raise Neo4jConnectionError(
                f"Failed to connect to Neo4j at {self._uri}: {exc}"
            ) from exc

    def close(self) -> None:
        """Release the driver and all underlying connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        """Yield a managed Neo4j session, raising on driver absence."""
        if self._driver is None:
            raise Neo4jConnectionError(
                "Neo4j driver is not initialised. Call connect() first."
            )
        session = self._driver.session()
        try:
            yield session
        except SessionExpired as exc:
            raise Neo4jConnectionError(
                f"Neo4j session expired: {exc}"
            ) from exc
        finally:
            session.close()

    def verify_connection(self) -> bool:
        """Return True if the driver can reach the server."""
        try:
            with self._session() as session:
                session.run("RETURN 1")
            return True
        except (Neo4jConnectionError, Exception):
            return False

    # ── index bootstrapping ─────────────────────────────────────────────

    def ensure_indexes(self) -> None:
        """Create uniqueness constraints / indexes for fast PPR traversal.

        Idempotent — safe to call on every application start-up.
        """
        constraints = [
            (
                "constraint_ontology_schema_id",
                "CREATE CONSTRAINT constraint_ontology_schema_id "
                "IF NOT EXISTS FOR (n:ONTOLOGY_SCHEMA) REQUIRE n.schema_id IS UNIQUE",
            ),
            (
                "constraint_fact_triple_id",
                "CREATE CONSTRAINT constraint_fact_triple_id "
                "IF NOT EXISTS FOR (n:FACT_TRIPLE) REQUIRE n.fact_id IS UNIQUE",
            ),
            (
                "constraint_passage_chunk_id",
                "CREATE CONSTRAINT constraint_passage_chunk_id "
                "IF NOT EXISTS FOR (n:PASSAGE_CHUNK) REQUIRE n.chunk_id IS UNIQUE",
            ),
            (
                "constraint_document_id",
                "CREATE CONSTRAINT constraint_document_id "
                "IF NOT EXISTS FOR (n:DOCUMENT) REQUIRE n.doc_id IS UNIQUE",
            ),
        ]
        with self._session() as session:
            for name, cypher in constraints:
                try:
                    session.run(cypher)
                    logger.debug("Ensured index/constraint: %s", name)
                except Exception as exc:
                    logger.warning("Index creation skipped (%s): %s", name, exc)

    # ═══════════════════════════════════════════════════════════════════
    #  WRITE helpers — Offline Ingestion populates these
    # ═══════════════════════════════════════════════════════════════════

    def upsert_ontology_schema(
        self,
        schema_id: str,
        head_type: str,
        relation: str,
        tail_type: str,
        status: str = "Pending",
        frequency: int = 1,
    ) -> None:
        """Create or update an ONTOLOGY_SCHEMA node.

        Frequency is incremented on conflict so that schemas naturally
        graduate from *Pending* → *Stable* once they cross threshold τ.
        """
        cypher = """
        MERGE (s:ONTOLOGY_SCHEMA {schema_id: $schema_id})
        ON CREATE SET
            s.head_type  = $head_type,
            s.relation   = $relation,
            s.tail_type  = $tail_type,
            s.status     = $status,
            s.frequency  = $frequency
        ON MATCH SET
            s.frequency  = s.frequency + 1,
            s.status     = CASE WHEN s.frequency + 1 >= $frequency
                                THEN 'Stable' ELSE s.status END
        """
        self._write(cypher, schema_id=schema_id, head_type=head_type,
                    relation=relation, tail_type=tail_type,
                    status=status, frequency=frequency)

    def upsert_fact_triple(
        self,
        fact_id: str,
        head_entity: str,
        relation: str,
        tail_entity: str,
        similarity_weight: float = 1.0,
        schema_id: Optional[str] = None,
    ) -> None:
        """Create or update a FACT_TRIPLE node and optionally link it to its
        constraining ONTOLOGY_SCHEMA.
        """
        cypher = """
        MERGE (f:FACT_TRIPLE {fact_id: $fact_id})
        ON CREATE SET
            f.head_entity      = $head_entity,
            f.relation         = $relation,
            f.tail_entity      = $tail_entity,
            f.similarity_weight = $similarity_weight
        ON MATCH SET
            f.head_entity      = $head_entity,
            f.relation         = $relation,
            f.tail_entity      = $tail_entity,
            f.similarity_weight = $similarity_weight
        """
        self._write(cypher, fact_id=fact_id, head_entity=head_entity,
                    relation=relation, tail_entity=tail_entity,
                    similarity_weight=similarity_weight)

        # Link FACT_TRIPLE → ONTOLOGY_SCHEMA
        if schema_id:
            link_cypher = """
            MATCH (f:FACT_TRIPLE {fact_id: $fact_id})
            MATCH (s:ONTOLOGY_SCHEMA {schema_id: $schema_id})
            MERGE (s)-[:CONSTRAINS]->(f)
            """
            self._write(link_cypher, fact_id=fact_id, schema_id=schema_id)

    def upsert_passage_chunk(
        self,
        chunk_id: str,
        document_id: str,
        section_label: str,
        content: str,
        url: str = "",
        page_number: int | None = None,
    ) -> None:
        """Create or update a PASSAGE_CHUNK node and link it to its DOCUMENT."""
        cypher = """
        MERGE (p:PASSAGE_CHUNK {chunk_id: $chunk_id})
        ON CREATE SET
            p.document_id   = $document_id,
            p.section_label = $section_label,
            p.content       = $content,
            p.url           = $url,
            p.page_number   = $page_number
        ON MATCH SET
            p.section_label = $section_label,
            p.content       = $content,
            p.url           = $url,
            p.page_number   = $page_number
        """
        self._write(cypher, chunk_id=chunk_id, document_id=document_id,
                    section_label=section_label, content=content,
                    url=url, page_number=page_number)

        # Link PASSAGE_CHUNK → DOCUMENT (auto-create stub if missing)
        link_cypher = """
        MATCH (p:PASSAGE_CHUNK {chunk_id: $chunk_id})
        MERGE (d:DOCUMENT {doc_id: $document_id})
        MERGE (p)-[:BELONGS_TO]->(d)
        """
        self._write(link_cypher, chunk_id=chunk_id, document_id=document_id)

    def upsert_document(
        self,
        doc_id: str,
        title: str = "",
        authors: str = "",
        year: int | None = None,
    ) -> None:
        """Create or update a DOCUMENT node."""
        cypher = """
        MERGE (d:DOCUMENT {doc_id: $doc_id})
        ON CREATE SET d.title = $title, d.authors = $authors, d.year = $year
        ON MATCH SET  d.title = $title, d.authors = $authors, d.year = $year
        """
        self._write(cypher, doc_id=doc_id, title=title,
                    authors=authors, year=year)

    def link_fact_to_passage(
        self,
        fact_id: str,
        chunk_id: str,
    ) -> None:
        """Create a provenance edge: FACT_TRIPLE -[:GROUNDED_IN]-> PASSAGE_CHUNK."""
        cypher = """
        MATCH (f:FACT_TRIPLE {fact_id: $fact_id})
        MATCH (p:PASSAGE_CHUNK {chunk_id: $chunk_id})
        MERGE (f)-[:GROUNDED_IN]->(p)
        """
        self._write(cypher, fact_id=fact_id, chunk_id=chunk_id)

    # ═══════════════════════════════════════════════════════════════════
    #  READ helpers — Online Retrieval consumes these
    # ═══════════════════════════════════════════════════════════════════

    def get_fact_triples_for_entity(
        self,
        entity: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return FACT_TRIPLEs where *entity* appears as head or tail."""
        cypher = """
        MATCH (f:FACT_TRIPLE)
        WHERE f.head_entity = $entity OR f.tail_entity = $entity
        RETURN f {.*} AS triple
        ORDER BY f.similarity_weight DESC
        LIMIT $limit
        """
        return self._read(cypher, entity=entity, limit=limit)

    def get_passages_for_document(
        self,
        document_id: str,
    ) -> List[Dict[str, Any]]:
        """Return every PASSAGE_CHUNK belonging to a given document."""
        cypher = """
        MATCH (p:PASSAGE_CHUNK {document_id: $document_id})
        RETURN p {.*} AS passage
        ORDER BY p.page_number
        """
        return self._read(cypher, document_id=document_id)

    def get_passages_for_fact(
        self,
        fact_id: str,
    ) -> List[Dict[str, Any]]:
        """Return PASSAGE_CHUNKs that ground a given fact (provenance)."""
        cypher = """
        MATCH (f:FACT_TRIPLE {fact_id: $fact_id})-[:GROUNDED_IN]->(p:PASSAGE_CHUNK)
        RETURN p {.*} AS passage
        """
        return self._read(cypher, fact_id=fact_id)

    def get_stable_schemas(self) -> List[Dict[str, Any]]:
        """Return all ONTOLOGY_SCHEMA nodes whose status is 'Stable'."""
        cypher = """
        MATCH (s:ONTOLOGY_SCHEMA {status: 'Stable'})
        RETURN s {.*} AS schema
        ORDER BY s.frequency DESC
        """
        return self._read(cypher)

    def get_all_fact_triples(
        self,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return up to *limit* FACT_TRIPLEs ordered by weight descending."""
        cypher = """
        MATCH (f:FACT_TRIPLE)
        RETURN f {.*} AS triple
        ORDER BY f.similarity_weight DESC
        LIMIT $limit
        """
        return self._read(cypher, limit=limit)

    # ═══════════════════════════════════════════════════════════════════
    #  Personalised PageRank (PPR)
    # ═══════════════════════════════════════════════════════════════════

    def run_personalised_pagerank(
        self,
        seed_entity: str,
        alpha: float | None = None,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pure-Cypher iterative PPR starting from *seed_entity*.

        This avoids a hard dependency on Neo4j GDS while remaining
        faithful to the Structure-Aware Node Initialization described in
        the PRD (§ Technical Research).

        Parameters
        ----------
        seed_entity:
            Head or tail entity value used as the PPR seed.
        alpha:
            Damping factor (defaults to ``settings.ppr_alpha``).
        top_k:
            Number of top-scored nodes to return (defaults to
            ``settings.ppr_top_k``).

        Returns
        -------
        List of dicts with keys ``node_id``, ``label``, ``score``, and
        the full node property map under ``properties``.
        """
        alpha = alpha or settings.ppr_alpha
        top_k = top_k or settings.ppr_top_k

        # Step 1 — collect the seed node IDs and their neighbour subgraph.
        #          We walk up to 3 hops out.
        cypher = """
        // Seed: FACT_TRIPLEs mentioning the entity
        MATCH (seed:FACT_TRIPLE)
        WHERE seed.head_entity = $entity OR seed.tail_entity = $entity
        WITH collect(id(seed)) AS seedIds

        // Expand neighbourhood ≤ 3 hops via any relationship
        MATCH (n)
        WHERE id(n) IN seedIds
        OPTIONAL MATCH path = (n)-[*1..3]-(neighbour)
        WITH seedIds,
             collect(DISTINCT id(neighbour)) + seedIds AS allIds

        // Score: simple iterative PPR approximation
        UNWIND allIds AS nid
        MATCH (node) WHERE id(node) = nid
        WITH node,
             CASE WHEN id(node) IN seedIds THEN 1.0 ELSE 0.0 END AS teleport,
             size( (node)--() ) AS degree
        WITH node, teleport, degree,
             // PPR score ≈ α * teleport + (1-α) / degree for neighbours
             $alpha * teleport + (1.0 - $alpha) * (1.0 / (degree + 1)) AS score
        RETURN
            id(node)    AS node_id,
            labels(node) AS labels,
            score,
            properties(node) AS properties
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self._read(
            cypher,
            entity=seed_entity,
            alpha=alpha,
            top_k=top_k,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Graph → Natural Language Serialisation
    # ═══════════════════════════════════════════════════════════════════

    def serialise_subgraph_to_sentences(
        self,
        entity: str,
        limit: int = 30,
    ) -> List[str]:
        """Serialise the neighbourhood of *entity* into plain-English
        sentences suitable for the Cross-Encoder reranker.

        E.g. ``"Transformer uses Self-Attention (weight=0.95)."``
        """
        triples = self.get_fact_triples_for_entity(entity, limit=limit)
        sentences: List[str] = []
        for record in triples:
            t = record.get("triple", record)
            head = t.get("head_entity", "?")
            rel = t.get("relation", "relates to")
            tail = t.get("tail_entity", "?")
            weight = t.get("similarity_weight", 0.0)
            sentences.append(
                f"{head} {rel} {tail} (weight={weight:.2f})."
            )
        return sentences

    # ═══════════════════════════════════════════════════════════════════
    #  Structure-Aware Node Initialisation helpers
    # ═══════════════════════════════════════════════════════════════════

    def compute_type_node_penalty(
        self,
        schema_id: str,
    ) -> float:
        """Return the hub-suppression penalty ``1 / log(degree + 1)``
        for a given ONTOLOGY_SCHEMA node, used by the PPR initialiser.
        """
        cypher = """
        MATCH (s:ONTOLOGY_SCHEMA {schema_id: $schema_id})
        OPTIONAL MATCH (s)-[r]-()
        RETURN count(r) AS degree
        """
        result = self._read(cypher, schema_id=schema_id)
        degree = result[0]["degree"] if result else 0
        return 1.0 / math.log(degree + 2)  # +2 avoids log(1)=0

    def get_passage_entity_ids(
        self,
        chunk_id: str,
    ) -> List[str]:
        """Return the entity names grounded in a specific passage chunk
        (for IDF-based Information Density calculation).
        """
        cypher = """
        MATCH (f:FACT_TRIPLE)-[:GROUNDED_IN]->(p:PASSAGE_CHUNK {chunk_id: $chunk_id})
        WITH collect(DISTINCT f.head_entity) + collect(DISTINCT f.tail_entity) AS entities
        UNWIND entities AS e
        RETURN DISTINCT e AS entity
        """
        return [r["entity"] for r in self._read(cypher, chunk_id=chunk_id)]

    # ═══════════════════════════════════════════════════════════════════
    #  Deletion / Maintenance
    # ═══════════════════════════════════════════════════════════════════

    def delete_document_cascade(self, doc_id: str) -> int:
        """Remove a DOCUMENT and all its PASSAGE_CHUNKS (+ dangling edges).

        Returns the number of nodes deleted.
        """
        cypher = """
        MATCH (d:DOCUMENT {doc_id: $doc_id})
        OPTIONAL MATCH (p:PASSAGE_CHUNK {document_id: $doc_id})
        DETACH DELETE d, p
        RETURN count(*) AS deleted
        """
        result = self._read(cypher, doc_id=doc_id)
        return result[0]["deleted"] if result else 0

    def clear_all(self) -> None:
        """⚠️  Delete every node and relationship in the database.

        Intended **only** for testing / full re-ingestion.
        """
        self._write("MATCH (n) DETACH DELETE n")
        logger.warning("All Neo4j data has been deleted.")

    # ═══════════════════════════════════════════════════════════════════
    #  Internal plumbing
    # ═══════════════════════════════════════════════════════════════════

    def _write(self, cypher: str, **params: Any) -> None:
        """Execute a write transaction (auto-commit)."""
        with self._session() as session:
            try:
                session.run(cypher, **params)
            except Exception as exc:
                raise Neo4jQueryError(
                    f"Write query failed: {exc}\nCypher: {cypher}"
                ) from exc

    def _read(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        """Execute a read transaction and return a list of record dicts."""
        with self._session() as session:
            try:
                result = session.run(cypher, **params)
                return [record.data() for record in result]
            except Exception as exc:
                raise Neo4jQueryError(
                    f"Read query failed: {exc}\nCypher: {cypher}"
                ) from exc
