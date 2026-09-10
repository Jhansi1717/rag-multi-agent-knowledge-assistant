"""ingestion/cleaner.py

Phase 3.5 — Text Cleaning and Normalization

Provides:
  clean_text(text: str) -> str
  clean_document(document: Document) -> Document
  TextCleaner — class wrapper for cleaner operations

Cleaning rules applied:
1. Normalizes CRLF (\\r\\n) and CR (\\r) line endings to LF (\\n).
2. Normalizes Unicode representations (NFKC).
3. Strips unprintable control characters and zero-width artifacts while
   preserving valid formatting.
4. Strips trailing whitespace from each line.
5. Normalizes repeated spaces/tabs to a single space per run while preserving line breaks.
6. Collapses excessive consecutive blank lines (3+ newlines reduced to 2 newlines / \\n\\n).
7. Strips leading and trailing blank space from the overall text.
8. Preserves headings, numbering, punctuation, table markup (|), symbols,
   and sentence boundaries.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Any, Dict

from ingestion.models import Document

# Regex pattern for non-printable control characters excluding \t and \n
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]")

# Regex pattern for horizontal whitespace (spaces, tabs, non-breaking spaces)
_HORIZONTAL_WS_RE = re.compile(r"[^\S\n\r]+")

# Regex pattern for 3 or more consecutive newlines
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize and clean extracted raw text without altering semantic meaning.

    Parameters
    ----------
    text:
        Raw textual content extracted from any document format.

    Returns
    -------
    str
        Cleaned, normalized text.
    """
    if not text:
        return ""

    # 1. Normalize line breaks to standard Unix \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalize Unicode representations (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 3. Strip non-printable control chars and zero-width artifacts
    text = _CONTROL_CHAR_RE.sub("", text)

    # 4. Clean line-by-line: collapse horizontal whitespace, strip line ends
    cleaned_lines = []
    for line in text.split("\n"):
        # Collapse multiple horizontal whitespace within the line
        norm_line = _HORIZONTAL_WS_RE.sub(" ", line).strip()
        cleaned_lines.append(norm_line)

    text = "\n".join(cleaned_lines)

    # 5. Collapse excessive blank lines (3 or more \n to 2 \n)
    text = _EXCESSIVE_NEWLINES_RE.sub("\n\n", text)

    # 6. Final strip of outer whitespace
    return text.strip()


def clean_document(document: Document) -> Document:
    """Return a new :class:`~ingestion.models.Document` with cleaned text.

    Preserves document identity, source, file type, and existing metadata
    while updating text with normalized content and adding cleaning metadata.

    Parameters
    ----------
    document:
        Input Document instance.

    Returns
    -------
    Document
        Document with normalized text and updated metadata.
    """
    cleaned_content = clean_text(document.text)

    # Create updated metadata preserving existing keys
    new_metadata: Dict[str, Any] = dict(document.metadata)
    new_metadata["cleaned"] = True
    new_metadata["cleaned_char_count"] = len(cleaned_content)

    return replace(
        document,
        text=cleaned_content,
        metadata=new_metadata,
    )


class TextCleaner:
    """Text cleaner helper class."""

    def clean(self, text: str) -> str:
        """Clean raw text string."""
        return clean_text(text)

    def clean_document(self, document: Document) -> Document:
        """Clean Document instance."""
        return clean_document(document)
