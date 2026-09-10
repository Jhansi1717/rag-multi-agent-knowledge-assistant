"""ingestion/pdf_extractor.py

Phase 3.2 — PDF Extraction via PyMuPDF

Provides :class:`PdfExtractor` — the single, focused implementation of
PDF-to-text extraction for Milestone 1.

Design decisions
----------------
* Uses ``fitz`` (PyMuPDF) exclusively.  No OCR, no image extraction.
* Opens the PDF with a ``with`` statement so the file handle is always closed,
  even when an exception is raised mid-extraction.
* Iterates every page in order, never skipping.
* Empty pages (zero-length text after stripping) are recorded in metadata
  rather than silently dropped or filled with invented text.
* If the entire PDF yields no extractable text, the returned text is an
  empty string and ``has_text`` is ``False`` — the caller decides how to
  surface that to the user.
* Milestone 1 scope: text-based PDFs only.  Scanned/image PDFs are
  out of scope; they will be detected (``has_text = False``) but not
  processed further.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, NamedTuple

import fitz  # PyMuPDF — already in requirements.txt


# ---------------------------------------------------------------------------
# Internal page-level result
# ---------------------------------------------------------------------------

class _PageResult(NamedTuple):
    page_number: int   # 1-based
    char_count: int
    is_empty: bool
    text: str


# ---------------------------------------------------------------------------
# PdfExtractor
# ---------------------------------------------------------------------------

class PdfExtractor:
    """Extract text and metadata from a PDF file using PyMuPDF.

    Usage
    -----
    ::

        extractor = PdfExtractor()
        result = extractor.extract("data/hr/leave_policy.pdf")
        print(result["text"])
        print(result["metadata"]["page_count"])
    """

    def extract(self, file_path: str) -> Dict[str, Any]:
        """Open *file_path*, extract text from every page, return a result dict.

        Parameters
        ----------
        file_path:
            Path to a ``.pdf`` file.  Must exist and be a regular file.

        Returns
        -------
        dict with keys:

        ``text`` : str
            Full text of the document (pages joined by ``"\\n\\n"``).
            Empty string if no extractable text was found.

        ``metadata`` : dict with keys:

            ``source_location`` : str
                The input *file_path*.
            ``document_name`` : str
                Basename of the file.
            ``file_size_bytes`` : int
                Size of the file on disk.
            ``format`` : str
                Always ``"pdf"``.
            ``page_count`` : int
                Total number of pages in the PDF.
            ``empty_page_count`` : int
                Number of pages that returned no extractable text.
            ``empty_page_numbers`` : list[int]
                1-based page numbers that were empty.
            ``pdf_producer`` : str
                Value of the PDF ``Producer`` metadata field, or ``""``.
            ``pdf_creator`` : str
                Value of the PDF ``Creator`` metadata field, or ``""``.
            ``pdf_creation_date`` : str
                Value of the PDF ``creationDate`` field, or ``""``.
            ``has_text`` : bool
                ``True`` if at least one page contained extractable text.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        RuntimeError
            If PyMuPDF cannot open the file (corrupted or not a real PDF).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path!r}")

        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        page_results: List[_PageResult] = []

        # ------------------------------------------------------------------
        # Open PDF — guaranteed close via context manager (Requirement 8).
        # ------------------------------------------------------------------
        try:
            pdf_doc = fitz.open(file_path)
        except Exception as exc:
            raise RuntimeError(
                f"PyMuPDF could not open {file_path!r}: {exc}"
            ) from exc

        with pdf_doc:
            page_count: int = pdf_doc.page_count  # Requirement 5

            # Collect document-level metadata from the PDF header.
            raw_meta: Dict[str, str] = pdf_doc.metadata or {}
            pdf_producer: str = raw_meta.get("producer", "") or ""
            pdf_creator: str = raw_meta.get("creator", "") or ""
            pdf_creation_date: str = raw_meta.get("creationDate", "") or ""

            # ------------------------------------------------------------------
            # Extract text page-by-page in order (Requirements 4 & 6).
            # ------------------------------------------------------------------
            for page_index in range(page_count):
                page = pdf_doc[page_index]
                raw_text: str = page.get_text()   # returns "" for empty pages

                stripped: str = raw_text.strip()
                is_empty: bool = len(stripped) == 0  # Requirement 7

                page_results.append(
                    _PageResult(
                        page_number=page_index + 1,   # 1-based
                        char_count=len(stripped),
                        is_empty=is_empty,
                        text=raw_text,                # preserve original spacing
                    )
                )
        # PDF file handle is now closed (Requirement 8 guaranteed by `with`)

        # ------------------------------------------------------------------
        # Assemble full text from page results (page order guaranteed).
        # ------------------------------------------------------------------
        page_texts: List[str] = [pr.text for pr in page_results if not pr.is_empty]
        full_text: str = "\n\n".join(page_texts)

        empty_pages: List[int] = [pr.page_number for pr in page_results if pr.is_empty]
        has_text: bool = len(full_text.strip()) > 0

        # ------------------------------------------------------------------
        # Build metadata dict (Requirement 5).
        # ------------------------------------------------------------------
        page_texts: List[Dict[str, Any]] = [
            {"page_number": pr.page_number, "text": pr.text}
            for pr in page_results
            if not pr.is_empty
        ]

        metadata: Dict[str, Any] = {
            "source_location": file_path,
            "document_name": filename,
            "file_size_bytes": file_size,
            "format": "pdf",
            "page_count": page_count,
            "page_texts": page_texts,
            "empty_page_count": len(empty_pages),
            "empty_page_numbers": empty_pages,
            "pdf_producer": pdf_producer,
            "pdf_creator": pdf_creator,
            "pdf_creation_date": pdf_creation_date,
            "has_text": has_text,
        }

        return {
            "text": full_text,
            "metadata": metadata,
        }
