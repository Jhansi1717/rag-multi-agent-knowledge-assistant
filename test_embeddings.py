"""test_embeddings.py

Phase 4.1 — Tests for the modular EmbeddingService.

Tests:
  - single text encoding
  - batch text encoding
  - chunk encoding
  - query encoding
  - empty input rejection
  - dimension consistency across calls
  - model name and dimension exposure
"""

import numpy as np
import pytest

from ingestion.models import make_chunk
from vector_store.embeddings import EmbeddingService


# ---------------------------------------------------------------------------
# Fixture: share a single model load across all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def svc():
    return EmbeddingService()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_model_name(svc):
    assert svc.model_name == "all-MiniLM-L6-v2"


def test_dimension_is_positive_int(svc):
    assert isinstance(svc.dimension, int)
    assert svc.dimension > 0


# ---------------------------------------------------------------------------
# encode_text — single string
# ---------------------------------------------------------------------------

def test_encode_text_returns_ndarray(svc):
    vec = svc.encode_text("Hello world")
    assert isinstance(vec, np.ndarray)


def test_encode_text_shape(svc):
    vec = svc.encode_text("Hello world")
    assert vec.shape == (svc.dimension,)


def test_encode_text_dtype(svc):
    vec = svc.encode_text("Hello world")
    assert vec.dtype == np.float32


def test_encode_text_empty_raises(svc):
    with pytest.raises(ValueError):
        svc.encode_text("")


def test_encode_text_whitespace_raises(svc):
    with pytest.raises(ValueError):
        svc.encode_text("   ")


# ---------------------------------------------------------------------------
# encode_texts — batch of strings
# ---------------------------------------------------------------------------

def test_encode_texts_shape(svc):
    texts = ["First text", "Second text", "Third text"]
    vecs = svc.encode_texts(texts)
    assert vecs.shape == (3, svc.dimension)


def test_encode_texts_dtype(svc):
    vecs = svc.encode_texts(["A", "B"])
    assert vecs.dtype == np.float32


def test_encode_texts_empty_list_raises(svc):
    with pytest.raises(ValueError):
        svc.encode_texts([])


def test_encode_texts_empty_element_raises(svc):
    with pytest.raises(ValueError):
        svc.encode_texts(["valid", ""])


# ---------------------------------------------------------------------------
# encode_chunks — batch of Chunk objects
# ---------------------------------------------------------------------------

def test_encode_chunks_shape(svc):
    chunks = [
        make_chunk(
            document_id="doc1",
            document_name="test.txt",
            chunk_index=i,
            text=f"Chunk number {i} with some content.",
            source_location="test.txt",
        )
        for i in range(4)
    ]
    vecs = svc.encode_chunks(chunks)
    assert vecs.shape == (4, svc.dimension)


def test_encode_chunks_empty_raises(svc):
    with pytest.raises(ValueError):
        svc.encode_chunks([])


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------

def test_encode_query_shape(svc):
    vec = svc.encode_query("What is the leave policy?")
    assert vec.shape == (svc.dimension,)


def test_encode_query_empty_raises(svc):
    with pytest.raises(ValueError):
        svc.encode_query("")


# ---------------------------------------------------------------------------
# Dimension consistency
# ---------------------------------------------------------------------------

def test_dimension_consistency_across_calls(svc):
    """All encode methods must return vectors with the same dimension."""
    v1 = svc.encode_text("first")
    v2 = svc.encode_query("second")
    v3 = svc.encode_texts(["third", "fourth"])
    chunk = make_chunk(
        document_id="d1", document_name="x.txt",
        chunk_index=0, text="fifth", source_location="x.txt",
    )
    v4 = svc.encode_chunks([chunk])

    assert v1.shape[-1] == svc.dimension
    assert v2.shape[-1] == svc.dimension
    assert v3.shape[-1] == svc.dimension
    assert v4.shape[-1] == svc.dimension
