"""M1.4 — Focused multi-agent retrieval tests."""

import os

import pytest

from agents import Orchestrator
from agents.models import RetrievalResult
from agents.query_understanding import QueryUnderstandingAgent
from agents.retrieval_agent import RetrievalAgent
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
    @pytest.mark.parametrize(
        "query,domain",
        [
            ("What is a Git branch?", "Software Engineering"),
            ("What is the standard oral dose of acetaminophen?", "Hospital Administration"),
            ("How long is a typical agile sprint?", "Software Engineering"),
            ("What is the purpose of hand hygiene?", "Hospital Administration"),
        ],
    )
    def test_factual_categories(self, query, domain):
        result = QueryUnderstandingAgent().analyze(query)
        assert result.query_type == "factual"
        assert result.routing == "RETRIEVAL"
        assert result.classification_confidence > 0
        assert result.domain == domain

    @pytest.mark.parametrize(
        "query,domain",
        [
            ("How do I create a feature branch?", "Software Engineering"),
            ("What are the steps for a pull request?", "Software Engineering"),
            ("What is the procedure for patient admission?", "Hospital Administration"),
            ("What protocol should nurses follow for hand hygiene?", "Hospital Administration"),
        ],
    )
    def test_procedural_categories(self, query, domain):
        result = QueryUnderstandingAgent().analyze(query)
        assert result.query_type == "procedural"
        assert result.routing == "RETRIEVAL"
        assert result.domain == domain

    @pytest.mark.parametrize(
        "query,domain",
        [
            ("How do Git merge and rebase differ?", "Software Engineering"),
            ("Compare REST versus SOAP APIs.", "Software Engineering"),
            ("What is the difference between ICU and emergency care?", "Hospital Administration"),
            ("Contrast oral and intravenous medication administration.", "Hospital Administration"),
        ],
    )
    def test_comparative_categories(self, query, domain):
        result = QueryUnderstandingAgent().analyze(query)
        assert result.query_type == "comparative"
        assert result.routing == "RETRIEVAL"
        assert result.domain == domain

    @pytest.mark.parametrize(
        "query",
        ["Git", "What about it?", "This?", "Can you explain that?"],
    )
    def test_ambiguous_categories(self, query):
        result = QueryUnderstandingAgent().analyze(query)
        assert result.query_type == "ambiguous"
        assert result.routing == "CLARIFICATION"
        assert result.classification_confidence > 0

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

    def test_unknown_fact_remains_factual(self):
        parsed = QueryUnderstandingAgent().analyze(
            "What is the annual revenue of TechCorp International?"
        )
        assert parsed.query_type == "factual"
        assert parsed.routing == "RETRIEVAL"
        assert parsed.classification_confidence > 0
        assert parsed.reason

    @pytest.mark.parametrize(
        "query,expected_type,domain",
        [
            (
                "What workflow should staff follow during patient admission?",
                "procedural",
                "Hospital Administration",
            ),
            (
                "What are the similarities and differences between REST and SOAP?",
                "comparative",
                "Software Engineering",
            ),
        ],
    )
    def test_explicit_m2_cues(self, query, expected_type, domain):
        result = QueryUnderstandingAgent().analyze(query)
        assert result.query_type == expected_type
        assert result.routing == "RETRIEVAL"
        assert result.domain == domain


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

    def test_unknown_fact_is_evaluated_after_retrieval(self, eval_orchestrator):
        resp = eval_orchestrator.handle(
            "What is the annual revenue of TechCorp International?"
        )
        assert resp.intent == "factual"
        assert resp.status in ("answered", "clarification_needed")

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


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.requested_top_k = None

    def retrieve(self, query, top_k=3):
        self.requested_top_k = top_k
        return self.results


class TestRetrievalAgent:
    def test_ranks_by_l2_distance_and_exposes_relevance(self):
        agent = RetrievalAgent(
            FakeRetriever(
                [
                    {
                        "similarity_score": 3.0,
                        "text": "far",
                        "document_id": "d2",
                        "filename": "far.txt",
                        "chunk_id": "c2",
                    },
                    {
                        "similarity_score": 1.0,
                        "text": "near",
                        "document_id": "d1",
                        "filename": "near.txt",
                        "chunk_id": "c1",
                    },
                ]
            ),
            min_relevance=0.0,
        )
        result = agent.retrieve("query", query_type="factual", top_k=2)
        assert isinstance(result, RetrievalResult)
        assert [hit.filename for hit in result.results] == ["near.txt", "far.txt"]
        assert result.results[0].distance_score == 1.0
        assert result.results[0].relevance_score == 0.5
        assert result.results[0].relevance_score > result.results[1].relevance_score

    def test_filters_below_threshold_and_preserves_metadata(self):
        result = RetrievalAgent(
            FakeRetriever(
                [
                    {
                        "similarity_score": 1.0,
                        "text": "supported",
                        "document_id": "d1",
                        "filename": "guide.txt",
                        "chunk_id": "c1",
                        "metadata": {"page": 2, "domain": "Software Engineering"},
                    },
                    {
                        "similarity_score": 9.0,
                        "text": "weak",
                        "document_id": "d2",
                        "filename": "weak.txt",
                        "chunk_id": "c2",
                    },
                ]
            ),
            min_relevance=0.4,
        ).retrieve("query", query_type="factual", domain="Software Engineering")
        assert len(result.results) == 1
        assert result.filtered_count == 1
        assert result.results[0].metadata["page"] == 2
        assert result.sufficient_evidence is True

    @pytest.mark.parametrize(
        "domain,query",
        [
            ("Software Engineering", "What is a Git branch?"),
            ("Hospital Administration", "What is patient admission?"),
        ],
    )
    def test_domain_filtering_does_not_fallback(self, domain, query):
        result = RetrievalAgent(
            FakeRetriever(
                [
                    {
                        "similarity_score": 0.1,
                        "text": "other domain",
                        "document_id": "d1",
                        "filename": "other.txt",
                        "chunk_id": "c1",
                        "metadata": {"domain": "Other"},
                    }
                ]
            )
        ).retrieve(query, query_type="factual", domain=domain)
        assert result.results == []
        assert result.no_relevant_information is True
        assert result.filtered_count == 1

    def test_empty_results_and_invalid_top_k(self):
        agent = RetrievalAgent(FakeRetriever([]))
        result = agent.retrieve("annual revenue", query_type="factual", top_k=3)
        assert result.results == []
        assert result.sufficient_evidence is False
        assert result.no_relevant_information is True
        with pytest.raises(ValueError):
            agent.retrieve("query", top_k=0)
