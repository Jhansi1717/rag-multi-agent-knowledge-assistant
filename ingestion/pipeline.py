"""ingestion/pipeline.py

M1.3 — End-to-end ingestion pipeline:

    file → validation → extraction → cleaning → chunking → (embedding + indexing via VectorStore)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ingestion.models import Chunk, Document, validate_file_path
from ingestion.extractor import extract_document
from ingestion.cleaner import clean_document
from ingestion.chunker import TextChunker
from ingestion.validator import validate_provenance

if TYPE_CHECKING:
    from vector_store.store import VectorStore


def infer_domain(file_path: str, text: str = "") -> str:
    """Infer a canonical domain from a conventional path or extracted text."""
    normalized = file_path.replace("\\", "/").lower()
    if "/hr/" in normalized or normalized.startswith("data/hr/"):
        return "Hospital Administration"
    if "/technical/" in normalized or normalized.startswith("data/technical/"):
        return "Software Engineering"
    if text:
        from agents.query_understanding import QueryUnderstandingAgent

        return QueryUnderstandingAgent().detect_domain(text) or ""
    return ""


def ingest_file(
    file_path: str,
    domain: Optional[str] = None,
    chunker: Optional[TextChunker] = None,
    original_filename: Optional[str] = None,
    source_location: Optional[str] = None,
) -> tuple[Document, List[Chunk]]:
    """Run validation → extraction → cleaning → chunking for one file."""
    validate_file_path(file_path)

    document = extract_document(file_path, original_filename=original_filename)
    if not document.text.strip():
        raise ValueError(
            f"No extractable text found in {document.filename!r}."
        )
    resolved_domain = (
        domain if domain is not None else infer_domain(file_path, document.text)
    )
    identity = source_location or document.source
    stable_identity = (
        identity.casefold()
        if identity.startswith("upload://")
        else os.path.normcase(os.path.abspath(identity))
    )
    metadata = dict(document.metadata)
    metadata["ingestion_path"] = document.source
    metadata["document_name"] = document.filename
    metadata["original_filename"] = document.filename
    metadata["source_location"] = identity
    if resolved_domain:
        metadata["domain"] = resolved_domain
    document = replace(
        document,
        document_id=str(uuid.uuid5(uuid.NAMESPACE_URL, stable_identity)),
        source=identity,
        metadata=metadata,
    )

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
    original_filename: Optional[str] = None,
    source_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest a file and index its chunks, skipping duplicate source paths."""
    document, chunks = ingest_file(
        file_path,
        domain=domain,
        chunker=chunker,
        original_filename=original_filename,
        source_location=source_location,
    )

    previous_state = (
        vector_store.index,
        vector_store.metadata,
        vector_store._indexed_sources.copy(),
        vector_store._last_reindexed_source,
    )
    try:
        added = vector_store.index_document(document, chunks)
        if save:
            vector_store.save()
    except Exception:
        (
            vector_store.index,
            vector_store.metadata,
            vector_store._indexed_sources,
            vector_store._last_reindexed_source,
        ) = previous_state
        raise

    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "file_type": document.file_type,
        "domain": document.metadata.get("domain", ""),
        "source_location": document.source,
        "metadata": {
            key: value
            for key, value in document.metadata.items()
            if key != "page_texts"
        },
        "chunk_count": len(chunks),
        "chunks_added": added,
        "reindexed": vector_store.was_source_reindexed(document.source),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "filename": chunk.document_name,
                "source_location": chunk.source_location,
                "domain": chunk.metadata.get("domain", ""),
                "metadata": {
                    key: value
                    for key, value in chunk.metadata.items()
                    if key != "page_texts"
                },
            }
            for chunk in chunks
        ],
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
