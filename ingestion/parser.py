"""
PDF Parser for Project Aletheia using PyMuPDF (fitz).

Extracts academic literature structure:
- Title, authors, publication year
- Section hierarchy (Abstract, Introduction, Method, Experiments, Results, References)
- Page mappings and clean text stream
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import fitz
from pydantic import BaseModel, Field

from core.exceptions import AletheiaError

logger = logging.getLogger(__name__)

# Standard academic section patterns
SECTION_HEADER_PATTERN = re.compile(
    r"^(?:(?:\d+\.?\s+)?(?:Abstract|Introduction|Related\s+Work|Background|Methodology|Method|Architecture|Framework|Approach|Experiments|Experimental\s+Setup|Results|Discussion|Conclusion|Conclusions|References|Appendix))\b",
    re.IGNORECASE,
)


class ParsedSection(BaseModel):
    """Represents a coherent section within an academic paper."""
    section_name: str
    content: str
    page_number: int


class ParsedDocument(BaseModel):
    """Represents the complete parsed structure of an academic PDF."""
    doc_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: int = 0
    abstract: str = ""
    sections: List[ParsedSection] = Field(default_factory=list)
    total_pages: int = 0
    file_path: str = ""

    @property
    def page_count(self) -> int:
        return self.total_pages


def _clean_text(text: str) -> str:
    """Fix hyphenation across line breaks and normalize whitespace."""
    # Join hyphenated words broken across linebreaks (e.g. "capabil-\nities" -> "capabilities")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace non-breaking spaces or strange characters
    text = text.replace("\xa0", " ")
    # Normalize multiple newlines and spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_year(text: str) -> Optional[int]:
    """Best-effort publication-year extraction from page-1 text.

    Looks for a 4-digit year in a plausible range (1900..current+1), preferring
    the most recent plausible year found (handles copyright lines, arXiv dates).
    Returns None when nothing plausible is present, so callers can mark it unknown
    rather than inventing a value (Issue 34).
    """
    from datetime import datetime as _dt
    current = _dt.now().year
    candidates = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    plausible = [y for y in candidates if 1900 <= y <= current + 1]
    return max(plausible) if plausible else None


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Parse an academic PDF document into structured sections.

    Args:
        file_path: Path to the PDF file

    Returns:
        ParsedDocument containing metadata, sections, and text
    """
    path = Path(file_path)
    if not path.is_file():
        raise AletheiaError(f"PDF file not found: {file_path}")

    doc_id = path.stem.replace(" ", "_")
    logger.info("Parsing PDF: %s (doc_id: %s)", path.name, doc_id)

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise AletheiaError(f"Failed to open PDF with PyMuPDF: {exc}") from exc

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise AletheiaError(f"PDF is empty: {file_path}")

    # ── 1. Metadata Extraction from Page 1 ──────────────────────────────
    page_1 = doc[0]
    p1_text = page_1.get_text("text")
    p1_lines = [line.strip() for line in p1_text.splitlines() if line.strip()]

    # Title (Feature 6.2): prefer the PDF's embedded metadata title when it looks
    # real (publishers often set it); otherwise fall back to the first prominent
    # line of page 1.
    meta_title = ""
    try:
        meta_title = (doc.metadata or {}).get("title", "") or ""
        meta_title = re.sub(r"\s+", " ", meta_title).strip()
    except Exception:
        meta_title = ""

    def _looks_like_real_title(t: str) -> bool:
        # Reject empty, too-short, filename-ish, or boilerplate metadata titles.
        if not t or len(t) < 8:
            return False
        low = t.lower()
        if low.endswith(".pdf") or low in ("untitled", "microsoft word", "document"):
            return False
        return True

    if _looks_like_real_title(meta_title):
        title = meta_title
    else:
        # First prominent line is typically title
        title = p1_lines[0] if p1_lines else path.stem
        if len(title) < 5 and len(p1_lines) > 1:
            title = f"{title} {p1_lines[1]}"

    # Extract authors (lines between title and Abstract)
    authors: List[str] = []
    abstract_idx = -1
    for i, line in enumerate(p1_lines):
        if line.lower() == "abstract" or line.lower().startswith("abstract:"):
            abstract_idx = i
            break

    if abstract_idx > 1:
        for line in p1_lines[1:abstract_idx]:
            # Filter emails and affiliations
            if "@" not in line and "university" not in line.lower() and "institute" not in line.lower() and "laboratory" not in line.lower():
                # Split comma-separated author lists
                for auth in re.split(r"[,;*†‡0-9]", line):
                    cleaned_auth = auth.strip()
                    if len(cleaned_auth) > 2 and len(cleaned_auth.split()) <= 4:
                        authors.append(cleaned_auth)

    # ── 2. Sequential Section Extraction ────────────────────────────────
    sections: List[ParsedSection] = []
    current_section_name = "Header"
    current_section_lines: List[str] = []
    current_section_page = 1
    abstract_text = ""
    all_page_texts: List[str] = []  # full document text, for the no-section fallback (Issue 36)

    for page_num in range(1, total_pages + 1):
        page = doc[page_num - 1]
        text = page.get_text("text")
        all_page_texts.append(text)
        lines = text.splitlines()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line is a section header
            if SECTION_HEADER_PATTERN.match(line_str) and len(line_str) < 80:
                # Flush previous section
                if current_section_lines:
                    content_str = _clean_text("\n".join(current_section_lines))
                    if content_str:
                        sec = ParsedSection(
                            section_name=current_section_name,
                            content=content_str,
                            page_number=current_section_page,
                        )
                        sections.append(sec)
                        if current_section_name.lower() == "abstract":
                            abstract_text = content_str

                # Start new section
                current_section_name = re.sub(r"^\d+\.?\s*", "", line_str).strip()
                current_section_lines = []
                current_section_page = page_num
            else:
                current_section_lines.append(line_str)

    # Flush final section
    if current_section_lines:
        content_str = _clean_text("\n".join(current_section_lines))
        if content_str:
            sec = ParsedSection(
                section_name=current_section_name,
                content=content_str,
                page_number=current_section_page,
            )
            sections.append(sec)
            if current_section_name.lower() == "abstract":
                abstract_text = content_str

    doc.close()

    # Fallback if no sections were identified (e.g. single body)
    if not sections:
        full_text = _clean_text("\n".join(all_page_texts))
        sections.append(
            ParsedSection(
                section_name="Body",
                content=full_text,
                page_number=1,
            )
        )

    logger.info(
        "Successfully parsed %s: %d sections, %d total pages",
        title[:50],
        len(sections),
        total_pages,
    )

    pub_year = _extract_year(p1_text)

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        authors=authors[:5] if authors else ["Research Team"],
        year=pub_year if pub_year is not None else 0,
        abstract=abstract_text,
        sections=sections,
        total_pages=total_pages,
        file_path=str(path),
    )
