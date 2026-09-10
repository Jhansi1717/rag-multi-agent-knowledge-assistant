"""M1.4 — Focused multi-agent retrieval tests."""

import os

import pytest

from agents import Orchestrator
from agents.query_understanding import QueryUnderstandingAgent
from ingestion import index_file
from ingestion.models import make_chunk
from vector_store.embeddings import EmbeddingProvider
from vector_store.store import VectorStore


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingProvider()


@pytest.fixture(scope="module")
def eval_orchestrator(embedder):
    index_path = "data/evaluation/index.faiss"
    meta_path = "data/evaluation/metadata.json"
    if not os.path.exists(index_path):
        pytest.skip("Evaluation index missing — run index_evaluation_corpus.py first.")
    store = VectorStore(
        index_path=index_path,
        meta_path=meta_path,
        embedder=embedder,
    )
    store.load()
    return Orchestrator(store, top_k=3)


@pytest.fixture
def mini_orchestrator(tmp_path, embedder):
    """Small in-memory index for controlled low-relevance tests."""
    txt = tmp_path / "notes.txt"
    txt.write_text(
        "Widget assembly requires three bolts and two washers. "
        "Torque each bolt to 25 Newton-meters.",
        encoding="utf-8",
    )
    store = VectorStore(
        index_path=str(tmp_path / "t.faiss"),
        meta_path=str(tmp_path / "t.json"),
        embedder=embedder,
    )
    index_file(str(txt), store, domain="Software Engineering", save=True)
    store.load()
    return Orchestrator(store, top_k=3, confidence_threshold=0.5)


class TestQueryUnderstanding:
    def test_factual_intent(self):
        parsed = QueryUnderstandingAgent().analyze(
            "How long is a typical agile sprint?"
        )
        assert parsed.intent == "factual"

    def test_procedural_intent(self):
        parsed = QueryUnderstandingAgent().analyze(
            "What are the steps to create a feature branch?"
        )
        assert parsed.intent == "procedural"

    def test_comparative_intent(self):
        parsed = QueryUnderstandingAgent().analyze(
            "How do microservices differ from a monolithic architecture?"
        )
        assert parsed.intent == "comparative"

    def test_unavailable_intent(self):
        parsed = QueryUnderstandingAgent().analyze(
            "What is the annual revenue of TechCorp International?"
        )
        assert parsed.intent == "unavailable"


class TestOrchestrator:
    def test_factual_query(self, eval_orchestrator):
        resp = eval_orchestrator.handle(
            "How long is a typical agile sprint in the engineering guide?"
        )
        assert resp.status == "answered"
        assert resp.intent == "factual"
        assert resp.retrieval_hits
        assert "sprint" in resp.answer.lower() or "two-week" in resp.answer.lower()
        assert resp.retrieval_hits[0].filename == "microservices_architecture.pdf"

    def test_procedural_query(self, eval_orchestrator):
        resp = eval_orchestrator.handle(
            "What are the steps to create a feature branch in git?"
        )
        assert resp.status == "answered"
        assert resp.intent == "procedural"
        assert "git_workflow" in resp.retrieval_hits[0].filename

    def test_comparative_query(self, eval_orchestrator):
        resp = eval_orchestrator.handle(
            "How do microservices differ from a monolithic architecture?"
        )
        assert resp.status == "answered"
        assert resp.intent == "comparative"
        assert resp.retrieval_hits

    def test_unavailable_query(self, eval_orchestrator):
        resp = eval_orchestrator.handle(
            "What is the annual revenue of TechCorp International?"
        )
        assert resp.status == "unavailable"
        assert resp.intent == "unavailable"
        assert "not available" in resp.answer.lower()

    def test_low_relevance(self, mini_orchestrator):
        resp = mini_orchestrator.handle(
            "Explain quantum chromodynamics in particle physics laboratories."
        )
        assert resp.status in ("clarification_needed", "unavailable")

    def test_citation_preservation(self, eval_orchestrator):
        resp = eval_orchestrator.handle(
            "What naming convention should classes use in the coding standards?"
        )
        assert resp.status == "answered"
        assert resp.citations
        cite = resp.citations[0]
        assert cite.chunk_id
        assert cite.filename == "coding_standards.docx"
        assert cite.excerpt
        assert "[source:" in resp.answer

    def test_retrieval_hit_fields(self, eval_orchestrator):
        resp = eval_orchestrator.handle(
            "What is the standard oral dose of Acetaminophen?",
            top_k=2,
        )
        hit = resp.retrieval_hits[0]
        assert hit.rank == 1
        assert hit.score >= 0
        assert hit.text
        assert hit.document_id
        assert hit.filename
        assert hit.chunk_id
