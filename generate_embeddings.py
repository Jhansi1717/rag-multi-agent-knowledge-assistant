"""generate_embeddings.py

Phase 4.2 — Generate embeddings for the Milestone 1 knowledge base.

Pipeline:
    Discover files → Extract → Clean → Chunk → Validate → Embed

Produces an in-memory list of indexing-ready records:
    {
        "chunk_id": "...",
        "document_id": "...",
        "document_name": "...",
        "chunk_index": ...,
        "text": "...",
        "source_location": "...",
        "metadata": {...},
        "embedding": [...]       # list of floats, len == dimension
    }

Does NOT build a FAISS index.
"""

import glob
import os
from typing import Any, Dict, List

import numpy as np

from ingestion import extract_document, clean_document, TextChunker, validate_provenance
from vector_store.embeddings import EmbeddingService


def discover_files() -> List[str]:
    """Return all supported files under data/hr/ and data/technical/."""
    files = []
    files.extend(glob.glob("data/hr/*.*"))
    files.extend(glob.glob("data/technical/*.*"))
    return [f for f in files if os.path.isfile(f) and not f.endswith(".gitkeep")]


def generate_embedding_records(
    embedding_service: EmbeddingService | None = None,
) -> List[Dict[str, Any]]:
    """Run the full pipeline and return indexing-ready records.

    Parameters
    ----------
    embedding_service : EmbeddingService, optional
        If None a new service is created (loads the model).

    Returns
    -------
    list[dict]
        One record per chunk, each containing an ``"embedding"`` key.
    """
    if embedding_service is None:
        embedding_service = EmbeddingService()

    chunker = TextChunker()
    files = discover_files()

    all_chunks = []
    file_results = []

    for file_path in files:
        doc = extract_document(file_path)
        cleaned_doc = clean_document(doc)
        chunks = chunker.chunk_document(cleaned_doc)
        validate_provenance(chunks, cleaned_doc)
        all_chunks.extend(chunks)
        file_results.append({
            "file": file_path,
            "filename": cleaned_doc.filename,
            "type": cleaned_doc.file_type,
            "chunk_count": len(chunks),
        })

    # Generate embeddings in a single batch
    if all_chunks:
        embeddings = embedding_service.encode_chunks(all_chunks)
    else:
        embeddings = np.empty((0, embedding_service.dimension), dtype=np.float32)

    # Build indexing-ready records
    records = []
    for i, chunk in enumerate(all_chunks):
        records.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source_location": chunk.source_location,
            "metadata": chunk.metadata,
            "embedding": embeddings[i].tolist(),
        })

    return records, file_results, embeddings


def main():
    print("================================================================")
    print("PHASE 4.2: EMBEDDING GENERATION FOR MILESTONE 1 CORPUS")
    print("================================================================")

    svc = EmbeddingService()
    records, file_results, embeddings = generate_embedding_records(svc)

    print(f"\nModel:              {svc.model_name}")
    print(f"Embedding dimension: {svc.dimension}")
    print()

    # Per-file report
    print("--- Per-File Report ---")
    for fr in file_results:
        print(f"  {fr['file']:<45} type={fr['type']:<6} chunks={fr['chunk_count']}")

    total_chunks = len(records)
    total_embeddings = len(embeddings)

    print()
    print("--- Aggregate ---")
    print(f"Files processed:    {len(file_results)}")
    print(f"Total chunks:       {total_chunks}")
    print(f"Total embeddings:   {total_embeddings}")

    # Validation
    errors = []

    if total_chunks != total_embeddings:
        errors.append(f"Count mismatch: {total_chunks} chunks vs {total_embeddings} embeddings")

    if total_embeddings > 0:
        dims = set(len(r["embedding"]) for r in records)
        if len(dims) != 1:
            errors.append(f"Inconsistent dimensions: {dims}")
        elif dims.pop() != svc.dimension:
            errors.append(f"Dimension mismatch: expected {svc.dimension}, got {dims}")

    for r in records:
        if not r["chunk_id"]:
            errors.append(f"Missing chunk_id in record")
        if not r["text"] or not r["text"].strip():
            errors.append(f"Empty text in chunk {r['chunk_id']}")

    print()
    if errors:
        print("Validation Status:  FAILED")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("Validation Status:  PASSED")
        print(f"  - chunk/embedding count aligned: {total_chunks} == {total_embeddings}")
        print(f"  - all embeddings have dimension: {svc.dimension}")
        print(f"  - all chunk_ids present")
        print(f"  - all texts non-empty")


if __name__ == "__main__":
    main()
