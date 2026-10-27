"""
Regression guards for previously-mitigated findings (Task 11).

These lock in fixes so they cannot silently regress. They do not need the live
server — they exercise the store layer directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.security


class TestBm25PickleGuard:
    """Issue 15: the BM25 index must persist as safe JSON, never as a pickle.

    A pickled index on a tamperable disk path is an arbitrary-code-execution
    risk on load. The store now writes JSON and rebuilds BM25 from the corpus.
    """

    def test_saved_index_is_json_not_pickle(self, tmp_path: Path):
        from memory.sparse_store import SparseStore

        index_path = tmp_path / "bm25_index.json"
        meta_path = tmp_path / "bm25_meta.json"
        store = SparseStore(index_path=str(index_path), metadata_path=str(meta_path))
        store.add_documents(
            ["attention rag compresses context", "bm25 lexical retrieval"],
            [{"chunk_id": "c1", "text": "a"}, {"chunk_id": "c2", "text": "b"}],
        )
        store.save()

        raw = index_path.read_bytes()
        # Pickle protocol-2+ streams begin with b"\x80"; JSON never does.
        assert not raw.startswith(b"\x80"), "BM25 index was written as a pickle stream"
        # And it must parse as JSON with the expected shape.
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        assert "tokenised_corpus" in payload

    def test_save_source_does_not_pickle(self):
        """The save() implementation must not SERIALISE via pickle.

        We check for an actual pickle write call (pickle.dump/dumps), not the
        word "pickle" — the method legitimately mentions pickle in a comment
        explaining why it avoids it.
        """
        import inspect
        from memory.sparse_store import SparseStore

        src = inspect.getsource(SparseStore.save)
        assert "pickle.dump" not in src, "SparseStore.save serialises via pickle — must use JSON"
        assert "json.dump" in src, "SparseStore.save no longer writes JSON"

    def test_roundtrip_rebuilds_from_corpus(self, tmp_path: Path):
        """A JSON index loads and rebuilds the BM25 index from the corpus.

        The security-relevant property is that load() reconstructs state from the
        safe JSON corpus (no pickle), so we assert the corpus + a live BM25 index
        survive the round-trip.
        """
        from memory.sparse_store import SparseStore

        index_path = tmp_path / "bm25_index.json"
        meta_path = tmp_path / "bm25_meta.json"
        s1 = SparseStore(index_path=str(index_path), metadata_path=str(meta_path))
        s1.add_documents(
            ["attention rag compresses context windows", "graph retrieval over triples"],
            [{"chunk_id": "c1", "text": "attention rag"}, {"chunk_id": "c2", "text": "graph"}],
        )
        s1.save()

        s2 = SparseStore(index_path=str(index_path), metadata_path=str(meta_path))
        s2.load()
        assert len(s2._tokenised_corpus) == 2, "corpus did not survive JSON round-trip"
        assert s2._bm25 is not None, "BM25 index was not rebuilt from the JSON corpus"
