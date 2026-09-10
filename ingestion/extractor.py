"""ingestion/extractor.py

Phase 3.1 — Text Extraction Layer

Provides:
  DocumentExtractor   — original class (unchanged), used by app.py and
                        index_samples.py.
  extract_document()  — new unified entry point that validates the path,
                        extracts text, and returns a normalised Document.
"""

import os
import pandas as pd
import fitz  # PyMuPDF
import docx

from ingestion.models import Document, make_document, validate_file_path
from ingestion.pdf_extractor import PdfExtractor
from ingestion.docx_extractor import DocxExtractor
from ingestion.txt_extractor import TxtExtractor
from ingestion.csv_extractor import CsvExtractor

class DocumentExtractor:
    def __init__(self):
        pass

    def extract(self, file_path: str) -> dict:
        """
        Extracts text from a given file path based on its extension.
        Returns a dictionary containing the extracted text and basic metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            "source_location": file_path,
            "document_name": os.path.basename(file_path)
        }

        if ext == '.txt':
            text = self._extract_txt(file_path)
        elif ext == '.pdf':
            text = self._extract_pdf(file_path)
        elif ext == '.docx':
            text = self._extract_docx(file_path)
        elif ext == '.csv':
            text = self._extract_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return {
            "text": text,
            "metadata": metadata
        }

    def _extract_txt(self, file_path: str) -> str:
        """Delegate to TxtExtractor (Phase 3.4)."""
        result = TxtExtractor().extract(file_path)
        return result["text"]

    def _extract_pdf(self, file_path: str) -> str:
        """Delegate to PdfExtractor (Phase 3.2)."""
        result = PdfExtractor().extract(file_path)
        return result["text"]

    def _extract_docx(self, file_path: str) -> str:
        """Delegate to DocxExtractor (Phase 3.3)."""
        result = DocxExtractor().extract(file_path)
        return result["text"]

    def _extract_csv(self, file_path: str) -> str:
        """Delegate to CsvExtractor (Phase 3.4)."""
        result = CsvExtractor().extract(file_path)
        return result["text"]


# ---------------------------------------------------------------------------
# Phase 3.1 — Unified extraction entry point
# ---------------------------------------------------------------------------

def extract_document(file_path: str) -> Document:
    """Validate *file_path*, extract its text, and return a :class:`Document`.

    This is the single public entry point for the ingestion package.
    It delegates format-specific work to the appropriate extractor and
    wraps the result in the normalised :class:`~ingestion.models.Document`
    dataclass.

    For PDF files, extraction is handled by :class:`~ingestion.pdf_extractor.
    PdfExtractor`, which captures page-level metadata and handles empty pages.

    For DOCX files, extraction is handled by :class:`~ingestion.docx_extractor.
    DocxExtractor`, which captures paragraph/table metadata and body order.

    For TXT files, extraction is handled by :class:`~ingestion.txt_extractor.
    TxtExtractor`, which preserves line structure and captures line count.

    For CSV files, extraction is handled by :class:`~ingestion.csv_extractor.
    CsvExtractor`, which turns tabular rows into searchable semantic text.

    Parameters
    ----------
    file_path:
        Absolute or relative path to a PDF, DOCX, TXT, or CSV file.

    Returns
    -------
    Document
        A fully populated :class:`Document` instance.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    IsADirectoryError
        If the path is not a regular file.
    ValueError
        If the file extension is unsupported.
    """
    # 1. Validate before touching any library.
    validate_file_path(file_path)

    ext: str = os.path.splitext(file_path)[1].lower()
    filename: str = os.path.basename(file_path)

    # 2. Dispatch to the format-specific extractor.
    if ext == ".pdf":
        result = PdfExtractor().extract(file_path)
    elif ext == ".docx":
        result = DocxExtractor().extract(file_path)
    elif ext == ".txt":
        result = TxtExtractor().extract(file_path)
    elif ext == ".csv":
        result = CsvExtractor().extract(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    raw_text: str = result["text"]
    extra_metadata: dict = result["metadata"]

    # 3. Return the normalised Document.
    return make_document(
        filename=filename,
        file_type=ext,
        source=file_path,
        text=raw_text,
        metadata=extra_metadata,
    )
