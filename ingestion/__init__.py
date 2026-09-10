"""ingestion package — Phase 3.1 - 3.7 public API.

Exports:

  from ingestion import Document, Chunk, extract_document, validate_file_path
  from ingestion import PdfExtractor   # Phase 3.2
  from ingestion import DocxExtractor  # Phase 3.3
  from ingestion import TxtExtractor   # Phase 3.4
  from ingestion import CsvExtractor   # Phase 3.4
  from ingestion import TextCleaner, clean_text, clean_document  # Phase 3.5
  from ingestion import TextChunker, chunk_text, chunk_document  # Phase 3.6
  from ingestion import validate_chunk, validate_chunks, validate_provenance, ProvenanceError  # Phase 3.7

``DocumentExtractor`` remains importable from ``ingestion.extractor`` for
backwards compatibility with ``app.py`` and ``index_samples.py``.
New code should call ``extract_document()``.
"""

from ingestion.models import (
    SUPPORTED_EXTENSIONS,
    Chunk,
    Document,
    make_chunk,
    make_document,
    validate_file_path,
)
from ingestion.pdf_extractor import PdfExtractor
from ingestion.docx_extractor import DocxExtractor
from ingestion.txt_extractor import TxtExtractor
from ingestion.csv_extractor import CsvExtractor
from ingestion.cleaner import TextCleaner, clean_text, clean_document
from ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunker,
    chunk_document,
    chunk_text,
)
from ingestion.validator import (
    ProvenanceError,
    validate_chunk,
    validate_chunks,
    validate_provenance,
)
from ingestion.extractor import DocumentExtractor, extract_document
from ingestion.pipeline import index_file, index_files, infer_domain, ingest_file

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "Chunk",
    "CsvExtractor",
    "Document",
    "DocumentExtractor",
    "DocxExtractor",
    "PdfExtractor",
    "ProvenanceError",
    "TextChunker",
    "TextCleaner",
    "TxtExtractor",
    "chunk_document",
    "chunk_text",
    "clean_document",
    "clean_text",
    "extract_document",
    "index_file",
    "index_files",
    "infer_domain",
    "ingest_file",
    "make_chunk",
    "make_document",
    "validate_chunk",
    "validate_chunks",
    "validate_file_path",
    "validate_provenance",
]
