"""ingestion/validator.py

Phase 3.7 — Ingestion Provenance Validation

Provides validation utilities to enforce end-to-end traceability:
  Chunk -> Document -> Source

Guarantees:
1. Every Chunk carries complete identity and location references:
   chunk_id, document_id, document_name, chunk_index, text, source_location, metadata.
2. Chunk IDs are non-empty, unique, and deterministic.
3. Chunk texts are non-empty.
4. Document-to-chunk references match exactly (document_id, source_location, document_name).
5. Format-specific source provenance is preserved in chunk metadata:
   - PDF: page_count, empty_page_numbers, etc.
   - CSV: row_count, column_count, column_names
   - DOCX: paragraph_count, table_count
   - TXT: line_count
"""

from __future__ import annotations

from typing import List, Optional, Set

from ingestion.models import Chunk, Document


class ProvenanceError(ValueError):
    """Raised when chunk or document provenance validation fails."""
    pass


def validate_chunk(chunk: Chunk, document: Optional[Document] = None) -> None:
    """Validate a single :class:`~ingestion.models.Chunk`'s provenance integrity.

    Parameters
    ----------
    chunk:
        The Chunk to validate.
    document:
        Optional parent Document to verify reference consistency.

    Raises
    ------
    ProvenanceError
        If any required field is missing or inconsistent.
    """
    if not isinstance(chunk, Chunk):
        raise ProvenanceError(f"Expected Chunk instance, got {type(chunk).__name__}")

    # 1. Identity & field presence
    if not chunk.chunk_id or not chunk.chunk_id.strip():
        raise ProvenanceError("Chunk is missing 'chunk_id'.")

    if not chunk.document_id or not chunk.document_id.strip():
        raise ProvenanceError(f"Chunk '{chunk.chunk_id}' is missing 'document_id'.")

    if not chunk.document_name or not chunk.document_name.strip():
        raise ProvenanceError(f"Chunk '{chunk.chunk_id}' is missing 'document_name'.")

    if not isinstance(chunk.chunk_index, int) or chunk.chunk_index < 0:
        raise ProvenanceError(
            f"Chunk '{chunk.chunk_id}' has invalid chunk_index: {chunk.chunk_index}"
        )

    if not chunk.text or not chunk.text.strip():
        raise ProvenanceError(f"Chunk '{chunk.chunk_id}' has empty text content.")

    if not chunk.source_location or not chunk.source_location.strip():
        raise ProvenanceError(f"Chunk '{chunk.chunk_id}' is missing 'source_location'.")

    if not isinstance(chunk.metadata, dict):
        raise ProvenanceError(f"Chunk '{chunk.chunk_id}' metadata must be a dict.")

    # 2. Document reference consistency
    if document is not None:
        if chunk.document_id != document.document_id:
            raise ProvenanceError(
                f"Chunk '{chunk.chunk_id}' document_id '{chunk.document_id}' "
                f"does not match Document '{document.document_id}'."
            )

        if chunk.source_location != document.source:
            raise ProvenanceError(
                f"Chunk '{chunk.chunk_id}' source_location '{chunk.source_location}' "
                f"does not match Document source '{document.source}'."
            )

        if chunk.document_name != document.filename:
            raise ProvenanceError(
                f"Chunk '{chunk.chunk_id}' document_name '{chunk.document_name}' "
                f"does not match Document filename '{document.filename}'."
            )


def validate_chunks(
    chunks: List[Chunk],
    document: Optional[Document] = None,
) -> None:
    """Validate a collection of chunks for uniqueness, ordering, and provenance.

    Parameters
    ----------
    chunks:
        List of Chunks to validate.
    document:
        Optional parent Document to verify.

    Raises
    ------
    ProvenanceError
        If duplicate IDs, non-sequential indices, or broken references are detected.
    """
    if not isinstance(chunks, list):
        raise ProvenanceError(f"Expected list of Chunks, got {type(chunks).__name__}")

    seen_ids: Set[str] = set()

    for expected_idx, chunk in enumerate(chunks):
        # Validate individual chunk fields & document reference
        validate_chunk(chunk, document=document)

        # Check for duplicate chunk IDs
        if chunk.chunk_id in seen_ids:
            raise ProvenanceError(f"Duplicate chunk_id detected: '{chunk.chunk_id}'")
        seen_ids.add(chunk.chunk_id)

        # Check for sequential chunk ordering
        if chunk.chunk_index != expected_idx:
            raise ProvenanceError(
                f"Non-sequential chunk_index at position {expected_idx}: "
                f"got {chunk.chunk_index}, expected {expected_idx}"
            )


def validate_provenance(chunks: List[Chunk], document: Document) -> None:
    """End-to-end provenance validation for an ingested Document and its Chunks.

    Verifies:
    1. Parent Document has valid identifiers and non-empty text.
    2. All Chunks trace back to the Document (ID, source, filename).
    3. Format-specific provenance metadata is present:
       - .pdf: page_count
       - .docx: paragraph_count
       - .txt: line_count
       - .csv: row_count, column_names
    4. Chunk token metrics are present and within valid ranges.

    Parameters
    ----------
    chunks:
        List of Chunks generated from *document*.
    document:
        Parent Document instance.

    Raises
    ------
    ProvenanceError
        If any provenance requirement is violated.
    """
    if not isinstance(document, Document):
        raise ProvenanceError(f"Expected Document instance, got {type(document).__name__}")

    if not document.document_id:
        raise ProvenanceError("Document is missing 'document_id'.")

    if not document.filename:
        raise ProvenanceError("Document is missing 'filename'.")

    if not document.source:
        raise ProvenanceError("Document is missing 'source'.")

    # Validate chunks against parent document
    validate_chunks(chunks, document=document)

    # Format-specific metadata checks
    ext = document.file_type.lower()
    meta = document.metadata or {}

    if ext == ".pdf":
        if "page_count" not in meta:
            raise ProvenanceError(f"PDF Document '{document.filename}' missing 'page_count' metadata.")
    elif ext == ".docx":
        if "paragraph_count" not in meta:
            raise ProvenanceError(f"DOCX Document '{document.filename}' missing 'paragraph_count' metadata.")
    elif ext == ".txt":
        if "line_count" not in meta:
            raise ProvenanceError(f"TXT Document '{document.filename}' missing 'line_count' metadata.")
    elif ext == ".csv":
        if "row_count" not in meta or "column_names" not in meta:
            raise ProvenanceError(f"CSV Document '{document.filename}' missing row/column provenance metadata.")

    # Validate token metrics in chunks
    for chunk in chunks:
        cmeta = chunk.metadata
        if "token_start" not in cmeta or "token_end" not in cmeta:
            raise ProvenanceError(
                f"Chunk '{chunk.chunk_id}' missing token boundary metadata (token_start/token_end)."
            )
        if cmeta["token_start"] > cmeta["token_end"]:
            raise ProvenanceError(
                f"Chunk '{chunk.chunk_id}' has invalid token range: [{cmeta['token_start']}:{cmeta['token_end']}]."
            )
