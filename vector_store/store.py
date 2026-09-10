import json
import os
from typing import Dict, List, Optional, Union

import faiss
import numpy as np

from ingestion.models import Chunk, Document
from vector_store.embeddings import EmbeddingProvider, DEFAULT_MODEL_NAME


class VectorStore:
    """FAISS-backed vector store with JSON metadata and duplicate-source guard."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        index_path: str = "data/index.faiss",
        meta_path: str = "data/metadata.json",
        embedder: Optional[EmbeddingProvider] = None,
    ):
        self.embedder = embedder or EmbeddingProvider(model_name)
        self.model_name = self.embedder.model_name
        self.index_path = index_path
        self.meta_path = meta_path
        self.dimension = self.embedder.dimension

        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata: Dict[str, dict] = {}
        self._indexed_sources: Dict[str, str] = {}
        self._last_reindexed_source: Optional[str] = None

    def _normalize_chunk(self, chunk: Union[Chunk, dict]) -> Chunk:
        if isinstance(chunk, Chunk):
            return chunk
        if isinstance(chunk, dict):
            return Chunk(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                document_name=chunk.get("document_name") or chunk.get("filename", ""),
                chunk_index=int(chunk["chunk_index"]),
                text=chunk["text"],
                source_location=chunk["source_location"],
                metadata=chunk.get("metadata", {}),
            )
        raise TypeError(f"Expected Chunk or dict, got {type(chunk).__name__}")

    def _meta_entries(self) -> Dict[str, dict]:
        return {k: v for k, v in self.metadata.items() if not k.startswith("_")}

    def _rebuild_index_from_metadata(self) -> None:
        entries = sorted(self._meta_entries().items(), key=lambda kv: int(kv[0]))
        self.index = faiss.IndexFlatL2(self.dimension)
        if not entries:
            return
        texts = [entry["text"] for _, entry in entries]
        embeddings = self.embedder.encode_texts(texts)
        self.index.add(embeddings)

    def _remove_source(self, source_location: str) -> bool:
        to_remove = [
            idx
            for idx, meta in self._meta_entries().items()
            if meta.get("source_location") == source_location
        ]
        if not to_remove:
            return False
        for idx in to_remove:
            del self.metadata[idx]
        self._indexed_sources.pop(source_location, None)
        self._rebuild_index_from_metadata()
        return True

    def is_source_indexed(self, source_location: str) -> bool:
        return source_location in self._indexed_sources

    def was_source_reindexed(self, source_location: str) -> bool:
        return self._last_reindexed_source == source_location

    def index_document(self, document: Document, chunks: List[Chunk]) -> int:
        """Index chunks for a document, replacing any prior index for the same source."""
        if not isinstance(document, Document):
            raise TypeError("document must be a Document instance")

        self._last_reindexed_source = None
        if self.is_source_indexed(document.source):
            self._remove_source(document.source)
            self._last_reindexed_source = document.source

        if not chunks:
            self._indexed_sources[document.source] = document.document_id
            return 0

        self.add_chunks(chunks)
        self._indexed_sources[document.source] = document.document_id
        return len(chunks)

    def add_chunks(self, chunks: List[Union[Chunk, dict]]):
        if not chunks:
            return

        normalized = [self._normalize_chunk(c) for c in chunks]
        texts = [chunk.text for chunk in normalized]
        embeddings = self.embedder.encode_chunks(normalized)

        start_id = len(self._meta_entries())
        self.index.add(embeddings)

        for i, chunk in enumerate(normalized):
            row_id = str(start_id + i)
            record = chunk.to_dict()
            record["embedding_model"] = self.model_name
            record["embedding_dim"] = self.dimension
            self.metadata[row_id] = record

    def save(self):
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_path) or ".", exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        payload = {
            "_indexed_sources": self._indexed_sources,
            **self.metadata,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Index or metadata file not found.")

        self.index = faiss.read_index(self.index_path)

        with open(self.meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._indexed_sources = raw.pop("_indexed_sources", {})
        self.metadata = {k: v for k, v in raw.items() if not k.startswith("_")}

        if not self._indexed_sources:
            for meta in self.metadata.values():
                src = meta.get("source_location")
                doc_id = meta.get("document_id")
                if src and doc_id:
                    self._indexed_sources[src] = doc_id

    def search(self, query: str, top_k: int = 3):
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedder.encode_query(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        entries = self._meta_entries()
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = entries[str(idx)].copy()
            # FAISS IndexFlatL2 returns a distance: lower values are better.
            # Keep the legacy key for existing M1 callers, but expose the
            # unambiguous name for M2 retrieval normalization.
            meta["distance_score"] = float(dist)
            meta["similarity_score"] = meta["distance_score"]
            results.append(meta)

        return results
