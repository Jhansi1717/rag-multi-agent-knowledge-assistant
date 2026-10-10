import json
import os
import shutil
import tempfile
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
        self.metadata = {
            str(new_id): entry
            for new_id, (_, entry) in enumerate(entries)
        }
        if not entries:
            return
        texts = [entry["text"] for _, entry in entries]
        embeddings = self.embedder.encode_texts(texts)
        self.index.add(embeddings)

    def _remove_source(self, source_location: str) -> bool:
        entries = self._meta_entries()
        removed_document_ids = {
            meta.get("document_id")
            for meta in entries.values()
            if meta.get("source_location") == source_location
        }
        remove_rows = {
            int(row_id)
            for row_id, meta in entries.items()
            if meta.get("source_location") == source_location
        }
        if not remove_rows:
            return False
        retained = [
            entry for row_id, entry in sorted(entries.items(), key=lambda item: int(item[0]))
            if int(row_id) not in remove_rows
        ]
        candidate_index = faiss.IndexFlatL2(self.dimension)
        if retained:
            candidate_index.add(np.vstack([
                self.index.reconstruct(int(row_id))
                for row_id, entry in sorted(entries.items(), key=lambda item: int(item[0]))
                if int(row_id) not in remove_rows
            ]).astype(np.float32, copy=False))
        candidate_metadata = {
            str(row_id): entry.copy() for row_id, entry in enumerate(retained)
        }
        candidate_sources = {
            source: document_id
            for source, document_id in self._indexed_sources.items()
            if source != source_location and document_id not in removed_document_ids
        }
        self._assert_alignment(candidate_index, candidate_metadata)
        self.index = candidate_index
        self.metadata = candidate_metadata
        self._indexed_sources = candidate_sources
        return True

    def is_source_indexed(self, source_location: str) -> bool:
        return source_location in self._indexed_sources

    def was_source_reindexed(self, source_location: str) -> bool:
        return self._last_reindexed_source == source_location

    def index_document(self, document: Document, chunks: List[Chunk]) -> int:
        """Atomically index a document, replacing its prior source or identity."""
        if not isinstance(document, Document):
            raise TypeError("document must be a Document instance")

        self._last_reindexed_source = None
        normalized = [self._normalize_chunk(chunk) for chunk in chunks]
        chunk_ids = set()
        for chunk in normalized:
            if chunk.document_id != document.document_id:
                raise ValueError("Chunk document_id does not match document identity")
            if chunk.source_location != document.source:
                raise ValueError("Chunk source_location does not match document source")
            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk_id in document: {chunk.chunk_id}")
            chunk_ids.add(chunk.chunk_id)

        embeddings = self._encode_chunks(normalized) if normalized else None
        current_entries = self._meta_entries()
        self._assert_alignment(self.index, current_entries)
        replaced_rows = {
            int(row_id)
            for row_id, entry in current_entries.items()
            if entry.get("source_location") == document.source
            or entry.get("document_id") == document.document_id
        }
        was_reindexed = bool(
            replaced_rows
            or document.source in self._indexed_sources
            or document.document_id in self._indexed_sources.values()
        )

        retained = [
            entry for row_id, entry in sorted(current_entries.items(), key=lambda item: int(item[0]))
            if int(row_id) not in replaced_rows
        ]
        retained_chunk_ids = {entry.get("chunk_id") for entry in retained}
        if retained_chunk_ids.intersection(chunk_ids):
            raise ValueError("Chunk ID already exists in another indexed document")
        candidate_index = faiss.IndexFlatL2(self.dimension)
        if retained:
            retained_vectors = np.vstack([
                self.index.reconstruct(int(row_id))
                for row_id, entry in sorted(current_entries.items(), key=lambda item: int(item[0]))
                if int(row_id) not in replaced_rows
            ]).astype(np.float32, copy=False)
            candidate_index.add(retained_vectors)
        candidate_metadata = {
            str(row_id): entry.copy() for row_id, entry in enumerate(retained)
        }
        if normalized:
            candidate_index.add(embeddings)
            for offset, chunk in enumerate(normalized):
                row_id = str(len(retained) + offset)
                record = chunk.to_dict()
                record["embedding_model"] = self.model_name
                record["embedding_dim"] = self.dimension
                candidate_metadata[row_id] = record

        candidate_sources = {
            source: document_id
            for source, document_id in self._indexed_sources.items()
            if source != document.source and document_id != document.document_id
        }
        candidate_sources[document.source] = document.document_id
        self._assert_alignment(candidate_index, candidate_metadata)

        self.index = candidate_index
        self.metadata = candidate_metadata
        self._indexed_sources = candidate_sources
        if was_reindexed:
            self._last_reindexed_source = document.source
        return len(normalized)

    def _encode_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        embeddings = np.asarray(self.embedder.encode_chunks(chunks), dtype=np.float32)
        expected_shape = (len(chunks), self.dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Embedder returned shape {embeddings.shape}, expected {expected_shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedder returned non-finite vector values")
        return np.ascontiguousarray(embeddings)

    @staticmethod
    def _assert_alignment(index, metadata: Dict[str, dict]) -> None:
        expected_ids = [str(row_id) for row_id in range(index.ntotal)]
        try:
            row_ids = sorted(metadata, key=int)
        except (TypeError, ValueError) as exc:
            raise ValueError("Metadata contains a non-numeric FAISS row ID.") from exc
        if row_ids != expected_ids:
            raise ValueError(
                "FAISS index and metadata row IDs do not align: "
                f"{index.ntotal} vectors, metadata rows {row_ids}."
            )
        if any(not isinstance(metadata[row_id], dict) for row_id in expected_ids):
            raise ValueError("Every FAISS row must map to one metadata object.")

    def add_chunks(self, chunks: List[Union[Chunk, dict]]):
        if not chunks:
            return

        normalized = [self._normalize_chunk(c) for c in chunks]
        current_entries = self._meta_entries()
        self._assert_alignment(self.index, current_entries)
        existing_chunk_ids = {
            entry.get("chunk_id") for entry in current_entries.values()
        }
        incoming_chunk_ids = [chunk.chunk_id for chunk in normalized]
        if len(set(incoming_chunk_ids)) != len(incoming_chunk_ids):
            raise ValueError("Duplicate chunk_id in added chunks")
        if existing_chunk_ids.intersection(incoming_chunk_ids):
            raise ValueError("Chunk ID already exists in the vector store")
        embeddings = self._encode_chunks(normalized)
        candidate_index = faiss.IndexFlatL2(self.dimension)
        if self.index.ntotal:
            candidate_index.add(np.vstack([
                self.index.reconstruct(row_id)
                for row_id in range(self.index.ntotal)
            ]).astype(np.float32, copy=False))
        candidate_index.add(embeddings)
        candidate_metadata = {
            row_id: entry.copy() for row_id, entry in current_entries.items()
        }
        start_id = len(candidate_metadata)
        for i, chunk in enumerate(normalized):
            row_id = str(start_id + i)
            record = chunk.to_dict()
            record["embedding_model"] = self.model_name
            record["embedding_dim"] = self.dimension
            candidate_metadata[row_id] = record
        self._assert_alignment(candidate_index, candidate_metadata)
        self.index = candidate_index
        self.metadata = candidate_metadata

    def save(self):
        self._assert_alignment(self.index, self._meta_entries())
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_path) or ".", exist_ok=True)

        payload = {
            "_indexed_sources": self._indexed_sources,
            **self.metadata,
        }
        index_fd, staged_index = tempfile.mkstemp(
            prefix="index-", suffix=".faiss.tmp",
            dir=os.path.dirname(self.index_path) or ".",
        )
        meta_fd, staged_meta = tempfile.mkstemp(
            prefix="metadata-", suffix=".json.tmp",
            dir=os.path.dirname(self.meta_path) or ".",
        )
        backup_index = None
        backup_meta = None
        index_existed = os.path.exists(self.index_path)
        meta_existed = os.path.exists(self.meta_path)
        try:
            os.close(index_fd)
            os.close(meta_fd)
            faiss.write_index(self.index, staged_index)
            with open(staged_meta, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if index_existed:
                backup_fd, backup_index = tempfile.mkstemp(
                    prefix="index-backup-", suffix=".tmp",
                    dir=os.path.dirname(self.index_path) or ".",
                )
                os.close(backup_fd)
                shutil.copy2(self.index_path, backup_index)
            if meta_existed:
                backup_fd, backup_meta = tempfile.mkstemp(
                    prefix="metadata-backup-", suffix=".tmp",
                    dir=os.path.dirname(self.meta_path) or ".",
                )
                os.close(backup_fd)
                shutil.copy2(self.meta_path, backup_meta)
            try:
                shutil.copyfile(staged_index, self.index_path)
                shutil.copyfile(staged_meta, self.meta_path)
            except Exception:
                if backup_index:
                    shutil.copyfile(backup_index, self.index_path)
                elif not index_existed and os.path.exists(self.index_path):
                    os.remove(self.index_path)
                if backup_meta:
                    shutil.copyfile(backup_meta, self.meta_path)
                elif not meta_existed and os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
                raise
        finally:
            for path in (staged_index, staged_meta, backup_index, backup_meta):
                if path and os.path.exists(path):
                    os.remove(path)

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Index or metadata file not found.")

        loaded_index = faiss.read_index(self.index_path)
        if loaded_index.d != self.dimension:
            raise ValueError(
                f"Loaded FAISS dimension {loaded_index.d} does not match embedder "
                f"dimension {self.dimension}."
            )

        with open(self.meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        indexed_sources = raw.pop("_indexed_sources", {})
        metadata = {k: v for k, v in raw.items() if not k.startswith("_")}
        self._assert_alignment(loaded_index, metadata)

        if not indexed_sources:
            for meta in metadata.values():
                src = meta.get("source_location")
                doc_id = meta.get("document_id")
                if src and doc_id:
                    indexed_sources[src] = doc_id
        self.index = loaded_index
        self.metadata = metadata
        self._indexed_sources = indexed_sources

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
            provenance = dict(meta.get("metadata") or {})
            provenance.setdefault("source_location", meta.get("source_location", ""))
            provenance.setdefault("domain", meta.get("domain", ""))
            meta["metadata"] = provenance
            # FAISS IndexFlatL2 returns a distance: lower values are better.
            # Keep the legacy key for existing M1 callers, but expose the
            # unambiguous name for M2 retrieval normalization.
            meta["distance_score"] = float(dist)
            meta["similarity_score"] = meta["distance_score"]
            results.append(meta)

        return results
