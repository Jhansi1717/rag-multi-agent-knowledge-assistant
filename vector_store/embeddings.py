"""vector_store/embeddings.py

M1.3 — EmbeddingProvider abstraction over sentence-transformers.

Baseline model: all-MiniLM-L6-v2 (384-dimensional embeddings)
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.models import Chunk

DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"


class EmbeddingProvider:
    """Thin wrapper around a SentenceTransformer model.

    The model is loaded exactly once during ``__init__`` and reused for every
    encode call.  All returned vectors are ``np.ndarray`` with dtype float32
    and consistent dimensionality equal to ``self.dimension``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        # sentence-transformers ≥ 3.x renamed the method; keep both for compat.
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension: int = self._model.get_embedding_dimension()
        else:
            self._dimension: int = self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("encode_text requires non-empty input text.")
        vec = np.asarray(self._model.encode(text, convert_to_numpy=True), dtype=np.float32)
        if vec.shape != (self.dimension,):
            raise ValueError(
                f"Model returned vector shape {vec.shape}, expected ({self.dimension},)."
            )
        if not np.isfinite(vec).all():
            raise ValueError("Model returned non-finite embedding values.")
        return vec

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("encode_texts requires a non-empty list of texts.")
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace-only.")
        vecs = np.asarray(self._model.encode(texts, convert_to_numpy=True), dtype=np.float32)
        expected_shape = (len(texts), self.dimension)
        if vecs.shape != expected_shape:
            raise ValueError(
                f"Model returned embedding shape {vecs.shape}, expected {expected_shape}."
            )
        if not np.isfinite(vecs).all():
            raise ValueError("Model returned non-finite embedding values.")
        return np.ascontiguousarray(vecs)

    def encode_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        if not chunks:
            raise ValueError("encode_chunks requires a non-empty list of Chunk objects.")
        texts = [chunk.text for chunk in chunks]
        return self.encode_texts(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_text(text)


# Backwards-compatible alias used by earlier milestones.
EmbeddingService = EmbeddingProvider
