"""
Focused unit tests for publication-year extraction in ingestion.parser.

These tests exercise the pure ``_extract_year`` helper directly so they do not
require PyMuPDF (fitz) or a real PDF. ``fitz`` is only imported at module import
time by ingestion.parser, so a minimal stub is installed before import.
"""

import sys
import types

# Stub fitz so importing ingestion.parser does not require PyMuPDF.
if "fitz" not in sys.modules:
    sys.modules["fitz"] = types.ModuleType("fitz")

from ingestion.parser import _extract_year, ParsedDocument


def test_extract_year_from_arxiv_id():
    """arXiv identifiers encode YYMM; the year maps to 20YY."""
    text = "A Great Paper\narXiv:2309.01234v2 [cs.CL]\nAbstract ..."
    assert _extract_year(text) == 2023


def test_extract_year_from_copyright_context():
    """A 4-digit year in a copyright/published context is preferred."""
    text = "Some Title\nPublished in 2019 at a venue.\n© 2019 The Authors."
    assert _extract_year(text) == 2019


def test_extract_year_returns_none_when_absent():
    """No plausible year present -> None (no hardcoded fallback)."""
    text = "A Title With No Dates\nAuthors and affiliations only.\nAbstract."
    assert _extract_year(text) is None


def test_extract_year_from_pdf_metadata():
    """Falls back to fitz metadata creationDate when text has no year."""
    text = "A Title With No Dates In Body"
    metadata = {"creationDate": "D:20210512093000"}
    assert _extract_year(text, metadata) == 2021


def test_extract_year_prefers_most_recent_standalone():
    """Standalone years: the most recent plausible year is chosen."""
    text = "Building on work from 1998 and 2005, we present results."
    assert _extract_year(text) == 2005


def test_parsed_document_year_defaults_to_none():
    """ParsedDocument.year is Optional[int] defaulting to None, not 2024."""
    doc = ParsedDocument(doc_id="x", title="Title")
    assert doc.year is None
