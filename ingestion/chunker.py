"""ingestion/chunker.py

M1.3 — Paragraph-aware token chunking via tiktoken.

Provides:
  chunk_text(text, ...) -> list[dict]
  chunk_document(document, ...) -> list[Chunk]
  TextChunker — class wrapper

Baseline (M1.3):
  DEFAULT_CHUNK_SIZE = 650 tokens  (target within 500–800)
  DEFAULT_CHUNK_OVERLAP = 75 tokens  (target within 50–100)
  DEFAULT_MAX_CHUNK_SIZE = 800 tokens
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import tiktoken

from ingestion.models import Chunk, Document, make_chunk

DEFAULT_CHUNK_SIZE: int = 650
DEFAULT_CHUNK_OVERLAP: int = 75
DEFAULT_MAX_CHUNK_SIZE: int = 800
DEFAULT_ENCODING: str = "cl100k_base"

_ROW_PREFIX_RE = re.compile(r"^Row (\d+):")


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


def _get_encoder(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def _count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text)) if text else 0


def _document_segments(document: Document) -> List[Dict[str, Any]]:
    """Split document text into segments preferring paragraph / page / row boundaries."""
    meta = document.metadata or {}
    ext = document.file_type.lower()

    if ext == ".pdf" and meta.get("page_texts"):
        return [
            {"text": p["text"].strip(), "page_number": p["page_number"]}
            for p in meta["page_texts"]
            if p.get("text", "").strip()
        ]

    if ext == ".csv":
        segments: List[Dict[str, Any]] = []
        for line in document.text.split("\n"):
            line = line.strip()
            if not line:
                continue
            seg: Dict[str, Any] = {"text": line}
            match = _ROW_PREFIX_RE.match(line)
            if match:
                seg["row_number"] = int(match.group(1))
            segments.append(seg)
        return segments

    paragraphs = re.split(r"\n\s*\n", document.text)
    return [{"text": p.strip()} for p in paragraphs if p.strip()]


def _segment_token_count(segment: Dict[str, Any], encoder: tiktoken.Encoding) -> int:
    return _count_tokens(segment["text"], encoder)


def _merge_segment_meta(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive page/row span metadata from grouped segments."""
    extra: Dict[str, Any] = {}
    pages = [s["page_number"] for s in segments if "page_number" in s]
    rows = [s["row_number"] for s in segments if "row_number" in s]
    if pages:
        extra["page_start"] = min(pages)
        extra["page_end"] = max(pages)
        if len(set(pages)) == 1:
            extra["page_number"] = pages[0]
    if rows:
        extra["row_start"] = min(rows)
        extra["row_end"] = max(rows)
        if len(set(rows)) == 1:
            extra["row_number"] = rows[0]
    return extra


def _token_slice_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str,
    start_chunk_index: int = 0,
) -> List[Dict[str, Any]]:
    """Fallback token sliding window for oversized segments."""
    encoder = _get_encoder(encoding_name)
    tokens: List[int] = encoder.encode(text)
    if not tokens:
        return []

    total_tokens = len(tokens)
    step = chunk_size - chunk_overlap
    chunks: List[Dict[str, Any]] = []
    chunk_index = start_chunk_index
    start_idx = 0

    while start_idx < total_tokens:
        end_idx = min(start_idx + chunk_size, total_tokens)
        chunk_token_slice = tokens[start_idx:end_idx]
        chunk_str = encoder.decode(chunk_token_slice)
        chunks.append({
            "text": chunk_str,
            "chunk_index": chunk_index,
            "token_start": start_idx,
            "token_end": end_idx,
            "token_count": len(chunk_token_slice),
            "char_count": len(chunk_str),
        })
        chunk_index += 1
        if end_idx == total_tokens:
            break
        start_idx += step

    return chunks


def _overlap_segments(
    segments: List[Dict[str, Any]],
    chunk_overlap: int,
    encoder: tiktoken.Encoding,
) -> List[Dict[str, Any]]:
    """Keep trailing segments whose combined tokens fit within overlap budget."""
    kept: List[Dict[str, Any]] = []
    token_budget = 0
    for seg in reversed(segments):
        seg_tokens = _segment_token_count(seg, encoder)
        if kept and token_budget + seg_tokens > chunk_overlap:
            break
        kept.insert(0, seg)
        token_budget += seg_tokens
    return kept


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    """Split *text* into paragraph-aware overlapping token segments."""
    _validate_chunk_params(chunk_size, chunk_overlap)

    if not text or not text.strip():
        return []

    pseudo_doc = Document(
        document_id="pseudo",
        filename="pseudo.txt",
        file_type=".txt",
        source="pseudo",
        created_at="1970-01-01T00:00:00+00:00",
        text=text,
        metadata={},
    )
    segments = _document_segments(pseudo_doc)
    return _chunk_segments(
        segments,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
        max_chunk_size=max_chunk_size,
    )


def _chunk_segments(
    segments: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str,
    max_chunk_size: int,
) -> List[Dict[str, Any]]:
    encoder = _get_encoder(encoding_name)
    if not segments:
        return []

    raw_chunks: List[Dict[str, Any]] = []
    current_segments: List[Dict[str, Any]] = []
    current_tokens = 0
    token_offset = 0
    chunk_index = 0

    def flush(current: List[Dict[str, Any]]) -> None:
        nonlocal chunk_index, token_offset
        if not current:
            return
        combined = "\n\n".join(s["text"] for s in current)
        token_count = _count_tokens(combined, encoder)
        span_meta = _merge_segment_meta(current)
        raw_chunks.append({
            "text": combined,
            "chunk_index": chunk_index,
            "token_start": token_offset,
            "token_end": token_offset + token_count,
            "token_count": token_count,
            "char_count": len(combined),
            **span_meta,
        })
        token_offset += max(token_count - chunk_overlap, 0)
        chunk_index += 1

    for segment in segments:
        seg_text = segment["text"]
        seg_tokens = _segment_token_count(segment, encoder)

        if seg_tokens > chunk_size or seg_tokens > max_chunk_size:
            if current_segments:
                flush(current_segments)
                current_segments = []
                current_tokens = 0
            for sub in _token_slice_chunks(
                seg_text, chunk_size, chunk_overlap, encoding_name, chunk_index
            ):
                sub.update(_merge_segment_meta([segment]))
                raw_chunks.append(sub)
                chunk_index = sub["chunk_index"] + 1
            continue

        if current_tokens + seg_tokens > chunk_size and current_segments:
            flush(current_segments)
            current_segments = _overlap_segments(current_segments, chunk_overlap, encoder)
            current_tokens = sum(_segment_token_count(s, encoder) for s in current_segments)

        current_segments.append(segment)
        current_tokens += seg_tokens

    if current_segments:
        flush(current_segments)

    for idx, raw in enumerate(raw_chunks):
        raw["chunk_index"] = idx

    return raw_chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    domain: Optional[str] = None,
) -> List[Chunk]:
    """Chunk a Document into Chunk models with full M1.3 metadata."""
    _validate_chunk_params(chunk_size, chunk_overlap)

    segments = _document_segments(document)
    raw_chunks = _chunk_segments(
        segments,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
        max_chunk_size=max_chunk_size,
    )

    resolved_domain = domain or document.metadata.get("domain", "")
    doc_chunks: List[Chunk] = []

    for raw in raw_chunks:
        chunk_meta: Dict[str, Any] = dict(document.metadata)
        chunk_meta.update({
            "filename": document.filename,
            "file_type": document.file_type,
            "domain": resolved_domain,
            "token_start": raw["token_start"],
            "token_end": raw["token_end"],
            "token_count": raw["token_count"],
            "char_count": raw["char_count"],
            "chunk_size_target": chunk_size,
            "chunk_overlap_target": chunk_overlap,
        })
        for key in ("page_number", "page_start", "page_end", "row_number", "row_start", "row_end"):
            if key in raw:
                chunk_meta[key] = raw[key]

        doc_chunks.append(
            make_chunk(
                document_id=document.document_id,
                document_name=document.filename,
                chunk_index=raw["chunk_index"],
                text=raw["text"],
                source_location=document.source,
                metadata=chunk_meta,
            )
        )

    return doc_chunks


class TextChunker:
    """Paragraph-aware token chunker."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        encoding_name: str = DEFAULT_ENCODING,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> None:
        _validate_chunk_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        return chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name,
            max_chunk_size=self.max_chunk_size,
        )

    def chunk_document(self, document: Document, domain: Optional[str] = None) -> List[Chunk]:
        return chunk_document(
            document,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name,
            max_chunk_size=self.max_chunk_size,
            domain=domain,
        )
