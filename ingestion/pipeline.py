"""ingestion/pipeline.py

M1.3 — End-to-end ingestion pipeline:

    file → validation → extraction → cleaning → chunking → (embedding + indexing via VectorStore)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ingestion.models import Chunk, Document, validate_file_path
from ingestion.extractor import extract_document
from ingestion.cleaner import clean_document
from ingestion.chunker import TextChunker
from ingestion.validator import validate_provenance

if TYPE_CHECKING:
    from vector_store.store import VectorStore


def infer_domain(file_path: str) -> str:
    """Infer knowledge domain from conventional data/ subdirectories."""
    normalized = file_path.replace("\\", "/").lower()
    if "/hr/" in normalized or normalized.startswith("data/hr/"):
        return "HR"
    if "/technical/" in normalized or normalized.startswith("data/technical/"):
        return "Software"
    return ""


def ingest_file(
    file_path: str,
    domain: Optional[str] = None,
    chunker: Optional[TextChunker] = None,
) -> tuple[Document, List[Chunk]]:
    """Run validation → extraction → cleaning → chunking for one file."""
    validate_file_path(file_path)

    document = extract_document(file_path)
    resolved_domain = domain if domain is not None else infer_domain(file_path)
    if resolved_domain:
        document.metadata["domain"] = resolved_domain

    cleaned = clean_document(document)
    chunker = chunker or TextChunker()
    chunks = chunker.chunk_document(cleaned, domain=resolved_domain)
    validate_provenance(chunks, cleaned)
    return cleaned, chunks


def index_file(
    file_path: str,
    vector_store: "VectorStore",
    domain: Optional[str] = None,
    chunker: Optional[TextChunker] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Ingest a file and index its chunks, skipping duplicate source paths."""
    document, chunks = ingest_file(file_path, domain=domain, chunker=chunker)

    added = vector_store.index_document(document, chunks)
    if save:
        vector_store.save()

    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "file_type": document.file_type,
        "domain": document.metadata.get("domain", ""),
        "source_location": document.source,
        "chunk_count": len(chunks),
        "chunks_added": added,
        "reindexed": vector_store.was_source_reindexed(document.source),
    }


def index_files(
    file_paths: List[str],
    vector_store: "VectorStore",
    domain: Optional[str] = None,
    save: bool = True,
) -> List[Dict[str, Any]]:
    """Index multiple files through the full pipeline."""
    results = []
    for path in file_paths:
        if not os.path.isfile(path):
            continue
        results.append(index_file(path, vector_store, domain=domain, save=False))
    if save:
        vector_store.save()
    return results
