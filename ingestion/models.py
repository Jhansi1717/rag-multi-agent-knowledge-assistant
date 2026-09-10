"""ingestion/models.py

Phase 3.1 — Common Ingestion Data Model and Interfaces

Defines the normalised Document representation that every format extractor
(PDF, DOCX, TXT, CSV) must produce, plus the shared validation guard that
sits in front of all extraction calls.

Nothing in this module imports from the rest of the project.
No embeddings, no vector-store, no external libraries beyond stdlib.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Supported extensions (single source of truth for the ingestion package)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset = frozenset({".pdf", ".docx", ".txt", ".csv"})


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Normalised representation of a single ingested document.

    Fields
    ------
    document_id : str
        Unique identifier for this extraction run (UUID4).
        Unique per call — suitable for local tracing through the pipeline.
    filename : str
        Original filename as provided to the extractor, including extension.
    file_type : str
        Lower-case file extension, e.g. ``".pdf"``, ``".csv"``.
    source : str
        Original file path or source identifier passed to the extractor.
    created_at : str
        ISO 8601 UTC timestamp of the extraction call.
    text : str
        Full raw textual content extracted from the document.
    metadata : dict
        Format-specific supplementary information (file size, page count, …).
        The exact keys depend on the extractor; callers must not assume any
        particular key is present.
    """

    document_id: str
    filename: str
    file_type: str
    source: str
    created_at: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Guard against obviously wrong construction.
        if not self.document_id:
            raise ValueError("document_id must not be empty.")
        if not self.filename:
            raise ValueError("filename must not be empty.")
        if not self.file_type.startswith("."):
            raise ValueError(
                f"file_type must be a dotted extension (e.g. '.pdf'), "
                f"got {self.file_type!r}."
            )

    # Convenience accessors ------------------------------------------------

    @property
    def extension(self) -> str:
        """Alias for ``file_type`` — lower-case dotted extension."""
        return self.file_type

    @property
    def word_count(self) -> int:
        """Approximate word count of the extracted text."""
        return len(self.text.split()) if self.text else 0

    @property
    def char_count(self) -> int:
        """Total character count of the extracted text."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialisation."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "source": self.source,
            "created_at": self.created_at,
            "text": self.text,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Normalised representation of an atomic segment of a Document.

    Fields
    ------
    chunk_id : str
        Deterministic composite identifier: ``"{document_id}_{chunk_index}"``.
    document_id : str
        Identifier of the parent Document.
    document_name : str
        Human-readable filename of the parent Document.
    chunk_index : int
        0-based sequential index of this chunk within the parent Document.
    text : str
        The textual content of this chunk.
    source_location : str
        Original file path or URI of the parent Document.
    metadata : dict
        Combined chunk-level and document-level metadata (token offsets, counts, etc.).
    """

    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    text: str
    source_location: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty.")
        if not self.document_id:
            raise ValueError("document_id must not be empty.")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be >= 0, got {self.chunk_index}")

    @property
    def token_count(self) -> int:
        """Token count from chunk metadata if available."""
        return self.metadata.get("token_count", 0)

    @property
    def char_count(self) -> int:
        """Total character count of chunk text."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation matching vector store schema."""
        meta = self.metadata or {}
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "filename": self.document_name,
            "document_name": self.document_name,
            "file_type": meta.get("file_type", ""),
            "domain": meta.get("domain", ""),
            "text": self.text,
            "source_location": self.source_location,
            "metadata": self.metadata,
        }


def make_chunk(
    *,
    document_id: str,
    document_name: str,
    chunk_index: int,
    text: str,
    source_location: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_id: Optional[str] = None,
) -> Chunk:
    """Create a :class:`Chunk` with standard deterministic ID."""
    return Chunk(
        chunk_id=chunk_id or f"{document_id}_{chunk_index}",
        document_id=document_id,
        document_name=document_name,
        chunk_index=chunk_index,
        text=text,
        source_location=source_location,
        metadata=metadata if metadata is not None else {},
    )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_document(
    *,
    filename: str,
    file_type: str,
    source: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Document:
    """Create a :class:`Document` with generated fields filled in automatically.

    Parameters
    ----------
    filename:
        Original filename (basename), e.g. ``"leave_policy.pdf"``.
    file_type:
        File extension.  Will be lower-cased automatically.
    source:
        Path or identifier of the source file.
    text:
        Full extracted text.
    metadata:
        Optional format-specific metadata dict.  Defaults to ``{}``.
    document_id:
        Override the auto-generated UUID4.  Useful for deterministic testing.
    created_at:
        Override the auto-generated ISO 8601 timestamp.  Useful for testing.

    Returns
    -------
    Document
    """
    return Document(
        document_id=document_id or str(uuid.uuid4()),
        filename=filename,
        file_type=file_type.lower(),
        source=source,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        text=text,
        metadata=metadata if metadata is not None else {},
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_file_path(file_path: str) -> None:
    """Assert that *file_path* can be ingested.

    Performs three checks in order:

    1. The path exists on the filesystem.
    2. The path points to a regular file (not a directory or other special
       filesystem entry).
    3. The file extension is one of :data:`SUPPORTED_EXTENSIONS`.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the candidate file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    IsADirectoryError
        If the path exists but is not a regular file.
    ValueError
        If the file extension is not in :data:`SUPPORTED_EXTENSIONS`.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path!r}"
        )

    if not os.path.isfile(file_path):
        raise IsADirectoryError(
            f"Path is not a regular file: {file_path!r}"
        )

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension {ext!r}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )
