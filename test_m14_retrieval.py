"""M1.4 — Focused multi-agent retrieval tests."""

import os
from types import SimpleNamespace

import pytest

from agents import Orchestrator
from agents.models import RetrievalResult
from agents.query_understanding import QueryUnderstandingAgent
from agents.retrieval_agent import RetrievalAgent
from ingestion import index_file
from ingestion.models import make_chunk
from vector_store.embeddings import EmbeddingProvider
from vector_store.store import VectorStore


class FakeLLM:
    def __init__(self, answer=None):
        self.answer = answer
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **request):
        answer = self.answer
        if answer is None:
            context = request["messages"][1]["content"]
            source_marker = "source_filename='"
            source_start = context.find(source_marker)
            source = ""
            evidence = "The retrieved evidence supports this result."
            if source_start >= 0:
                source_start += len(source_marker)
                source = context[source_start:].split("'", 1)[0]
                metadata_end = context.find("\n", context.find("metadata=", source_start))
                if metadata_end >= 0:
                    evidence = context[metadata_end + 1:].split("\n\n[", 1)[0].strip()
            answer = f"{evidence} [source: {source}]"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
        )


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
    from agents.response_generation import ResponseGenerationAgent

    return Orchestrator(
        store,
        top_k=3,
        response_gen=ResponseGenerationAgent(llm_client=FakeLLM()),
    )


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
        assert resp.status in ("answered", "unavailable", "clarification_needed")

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
                        "metadata": {
                            "page": 2,
                            "section": "Deployment",
                            "row_start": 4,
                            "row_end": 6,
                            "domain": "Software Engineering",
                        },
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
        assert result.results[0].metadata["section"] == "Deployment"
        assert result.results[0].metadata["row_start"] == 4
        assert result.results[0].metadata["row_end"] == 6
        assert result.sufficient_evidence is True

    def test_comparison_reserves_each_target_with_small_top_k(self):
        class TargetRetriever:
            def retrieve(self, query, top_k=3):
                if query == "Compare release trains versus identity verification.":
                    return [
                        {
                            "distance_score": 0.49,
                            "text": "Identity verification evidence.",
                            "document_id": "identity-doc",
                            "filename": "identity.docx",
                            "chunk_id": "identity-1",
                        },
                        {
                            "distance_score": 1.058,
                            "text": "More identity verification evidence.",
                            "document_id": "identity-pdf",
                            "filename": "identity.pdf",
                            "chunk_id": "identity-2",
                        },
                        {
                            "distance_score": 1.0671,
                            "text": "Hospital identity verification evidence.",
                            "document_id": "identity-hospital",
                            "filename": "hospital_identity.txt",
                            "chunk_id": "identity-3",
                        },
                    ][:top_k]
                if query == "release trains":
                    return [{
                        "distance_score": 1.1454,
                        "text": "Release trains coordinate software delivery.",
                        "document_id": "release-doc",
                        "filename": "release_trains.txt",
                        "chunk_id": "release-1",
                    }]
                if query == "identity verification":
                    return [{
                        "distance_score": 0.49,
                        "text": "Identity verification evidence.",
                        "document_id": "identity-doc",
                        "filename": "identity.docx",
                        "chunk_id": "identity-1",
                    }]
                return []

        result = RetrievalAgent(TargetRetriever(), min_relevance=0.45).retrieve(
            "Compare release trains versus identity verification.",
            query_type="comparative",
            top_k=3,
        )

        assert len(result.results) == 3
        assert {hit.filename for hit in result.results} >= {
            "release_trains.txt",
            "identity.docx",
        }
        assert [hit.rank for hit in result.results] == [1, 2, 3]

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

    def test_domain_filter_rejects_conflicting_domain_metadata(self):
        result = RetrievalAgent(
            FakeRetriever([
                {
                    "distance_score": 0.1,
                    "text": "A passage with contradictory domain labels.",
                    "document_id": "d1",
                    "filename": "conflict.txt",
                    "chunk_id": "c1",
                    "domain": "Software Engineering",
                    "metadata": {"domain": "Hospital Administration"},
                }
            ]),
            min_relevance=0.0,
        ).retrieve(
            "A query for software engineering.",
            query_type="factual",
            domain="Software Engineering",
        )

        assert result.results == []
        assert result.filtered_count == 1
        assert result.no_relevant_information is True

    def test_missing_distance_is_not_treated_as_perfect_relevance(self):
        agent = RetrievalAgent(
            FakeRetriever([{
                "text": "A result without score metadata.",
                "document_id": "d1",
                "filename": "missing-score.txt",
                "chunk_id": "c1",
            }]),
            min_relevance=0.0,
        )

        with pytest.raises(ValueError, match="no L2 distance score"):
            agent.retrieve("query", query_type="factual", top_k=1)

    @pytest.mark.parametrize("distance", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_distance_is_rejected(self, distance):
        agent = RetrievalAgent(
            FakeRetriever([{
                "distance_score": distance,
                "text": "Invalid numeric score.",
                "document_id": "d1",
                "filename": "invalid-score.txt",
                "chunk_id": "c1",
            }]),
            min_relevance=0.0,
        )

        with pytest.raises(ValueError, match="non-finite"):
            agent.retrieve("query", query_type="factual", top_k=1)

    def test_top_k_caps_results_even_if_backend_overreturns(self):
        class OverreturningRetriever:
            requested_top_k = None

            def retrieve(self, query, top_k=3):
                self.requested_top_k = top_k
                return [
                    {
                        "distance_score": distance,
                        "text": f"Candidate {rank}",
                        "document_id": f"d{rank}",
                        "filename": f"candidate-{rank}.txt",
                        "chunk_id": f"c{rank}",
                    }
                    for rank, distance in enumerate((0.1, 0.2, 0.3, 0.4), start=1)
                ]

        backend = OverreturningRetriever()
        result = RetrievalAgent(backend, min_relevance=0.0).retrieve(
            "query", query_type="factual", top_k=2
        )

        assert backend.requested_top_k == 2
        assert len(result.results) == result.top_k == 2
        assert [hit.rank for hit in result.results] == [1, 2]

    def test_missing_requested_domain_rejects_unlabeled_candidate(self):
        result = RetrievalAgent(
            FakeRetriever([
                {
                    "distance_score": 0.1,
                    "text": "Git feature branch workflow steps.",
                    "document_id": "doc-1",
                    "filename": "guide.txt",
                    "chunk_id": "chunk-1",
                    "domain": "",
                    "metadata": {"domain": ""},
                }
            ]),
            min_relevance=0.0,
        ).retrieve(
            "How do I create a Git feature branch?",
            query_type="procedural",
            domain="Software Engineering",
        )

        assert result.results == []
        assert result.filtered_count == 1
        assert result.no_relevant_information is True
        assert result.sufficient_evidence is False

    def test_domain_filter_rejects_unlabeled_upload_hits(self):
        result = RetrievalAgent(
            FakeRetriever(
                [
                    {
                        "similarity_score": 0.1,
                        "text": "Git feature branch workflow steps.",
                        "document_id": "doc-1",
                        "filename": "guide.txt",
                        "chunk_id": "chunk-1",
                        "domain": "",
                        "metadata": {"domain": ""},
                    }
                ]
            )
        ).retrieve(
            "How do I create a Git feature branch?",
            query_type="procedural",
            domain="Software Engineering",
        )

        assert result.results == []
        assert result.filtered_count == 1
        assert result.no_relevant_information is True
        assert result.sufficient_evidence is False

    def test_comparative_retrieval_searches_each_target(self):
        class TargetRetriever:
            def retrieve(self, query, top_k=3):
                if "git feature branch workflow" in query.lower():
                    return [{
                        "distance_score": 0.1,
                        "text": "Git feature branch workflow evidence.",
                        "document_id": "software-doc",
                        "filename": "software.txt",
                        "chunk_id": "software-1",
                    }]
                if "patient admission paperwork" in query.lower():
                    return [{
                        "distance_score": 0.2,
                        "text": "Patient admission paperwork evidence.",
                        "document_id": "hospital-doc",
                        "filename": "hospital.txt",
                        "chunk_id": "hospital-1",
                    }]
                return [{
                    "distance_score": 0.15,
                    "text": "Git feature branch workflow evidence.",
                    "document_id": "software-doc",
                    "filename": "software.txt",
                    "chunk_id": "software-1",
                }]

        result = RetrievalAgent(TargetRetriever(), min_relevance=0.0).retrieve(
            "Compare the Git feature branch workflow with patient admission paperwork.",
            query_type="comparative",
            top_k=3,
        )

        assert {hit.filename for hit in result.results} == {
            "software.txt",
            "hospital.txt",
        }
        assert len(result.results) == 2

    def test_unavailable_queries_require_clear_signal(self):
        agent = RetrievalAgent(
            FakeRetriever([
                {
                    "similarity_score": 0.55,
                    "text": "Hospital admission policy: patient needs ID and insurance.",
                    "document_id": "doc-admission",
                    "filename": "admission_policy.pdf",
                    "chunk_id": "admission-1",
                },
                {
                    "similarity_score": 0.52,
                    "text": "Hospital admission process and check-in steps.",
                    "document_id": "doc-admission-2",
                    "filename": "admission_process.pdf",
                    "chunk_id": "admission-2",
                },
            ]),
            min_relevance=0.45,
        )
        for query in [
            "How much revenue did the hospital generate?",
            "What was the hospital's annual revenue?",
        ]:
            result = agent.retrieve(query, query_type="factual", top_k=2)
            assert result.results == []
            assert result.no_relevant_information is True
            assert result.sufficient_evidence is False

    def test_named_hospital_revenue_rejects_admission_policy_neighbor(self):
        agent = RetrievalAgent(
            FakeRetriever([
                {
                    "distance_score": 0.8734,
                    "text": (
                        "Riverside General Hospital patient admission policy. "
                        "Patients must present a photo ID and insurance card and "
                        "understand their financial responsibility."
                    ),
                    "document_id": "admission-policy",
                    "filename": "patient_admission_policy.pdf",
                    "chunk_id": "admission-1",
                },
                {
                    "distance_score": 1.4182,
                    "text": "Riverside General Hospital emergency response protocols.",
                    "document_id": "emergency-protocols",
                    "filename": "emergency_protocols.txt",
                    "chunk_id": "emergency-1",
                },
            ]),
            min_relevance=0.45,
        )

        result = agent.retrieve(
            "What was Riverside General Hospital's total revenue last fiscal year?",
            query_type="factual",
            top_k=2,
        )

        assert result.results == []
        assert result.no_relevant_information is True
        assert result.sufficient_evidence is False

    def test_valid_close_semantic_match_is_not_filtered(self):
        agent = RetrievalAgent(
            FakeRetriever([
                {
                    "similarity_score": 0.20,
                    "text": "Release trains enable frequent, manageable deployments and safer releases.",
                    "document_id": "doc-release",
                    "filename": "release_trains.txt",
                    "chunk_id": "release-1",
                },
                {
                    "similarity_score": 0.36,
                    "text": "Identity verification confirms a user's authenticated identity.",
                    "document_id": "doc-identity",
                    "filename": "identity_verification.txt",
                    "chunk_id": "identity-1",
                },
            ]),
            min_relevance=0.45,
        )
        result = agent.retrieve(
            "What is the main advantage of release trains?",
            query_type="factual",
            top_k=2,
        )
        assert [hit.filename for hit in result.results] == ["release_trains.txt"]
        assert result.sufficient_evidence is True

    def test_same_document_cluster_keeps_best_hit_without_false_no_result(self):
        agent = RetrievalAgent(
            FakeRetriever([
                {
                    "similarity_score": 0.18,
                    "text": "Release trains enable small well-scoped deployments and safer rollouts.",
                    "document_id": "doc-release",
                    "filename": "release_trains.txt",
                    "chunk_id": "release-1",
                },
                {
                    "similarity_score": 0.22,
                    "text": "Release trains also reduce risk by pacing changes across teams.",
                    "document_id": "doc-release",
                    "filename": "release_trains.txt",
                    "chunk_id": "release-2",
                },
            ]),
            min_relevance=0.45,
        )
        result = agent.retrieve(
            "What is the main advantage of release trains?",
            query_type="factual",
            top_k=2,
        )
        assert [hit.filename for hit in result.results] == ["release_trains.txt"]
        assert result.sufficient_evidence is True

    def test_directly_supported_close_identity_hits_keep_best_evidence(self):
        agent = RetrievalAgent(
            FakeRetriever([
                {
                    "distance_score": 0.6056,
                    "text": (
                        "At hospital admission, staff verify patient identity. "
                        "The main advantage of identity verification is preventing "
                        "patient misidentification and supporting safe care."
                    ),
                    "document_id": "hospital-identity",
                    "filename": "hospital_identity.txt",
                    "chunk_id": "identity-1",
                },
                {
                    "distance_score": 0.6317,
                    "text": (
                        "Identity verification confirms that a person is who they "
                        "claim to be before granting access."
                    ),
                    "document_id": "identity-reference",
                    "filename": "identity_reference.docx",
                    "chunk_id": "identity-2",
                },
            ]),
            min_relevance=0.45,
        )

        result = agent.retrieve(
            "What is the main advantage of identity verification?",
            query_type="factual",
            top_k=2,
        )

        assert [hit.filename for hit in result.results] == ["hospital_identity.txt"]
        assert result.sufficient_evidence is True

    def test_valid_comparison_keeps_both_targets(self):
        agent = RetrievalAgent(
            FakeRetriever([
                {
                    "similarity_score": 0.18,
                    "text": "Release trains allow frequent, low-risk deployment across teams.",
                    "document_id": "doc-release",
                    "filename": "release_trains.txt",
                    "chunk_id": "release-1",
                },
                {
                    "similarity_score": 0.24,
                    "text": "Identity verification confirms who the user is before access is granted.",
                    "document_id": "doc-identity",
                    "filename": "identity_verification.txt",
                    "chunk_id": "identity-1",
                },
            ]),
            min_relevance=0.45,
        )
        result = agent.retrieve(
            "Compare release trains versus identity verification.",
            query_type="comparative",
            top_k=2,
        )
        assert {hit.filename for hit in result.results} == {
            "release_trains.txt",
            "identity_verification.txt",
        }
        assert result.sufficient_evidence is True

    def test_empty_results_and_invalid_top_k(self):
        agent = RetrievalAgent(FakeRetriever([]))
        result = agent.retrieve("annual revenue", query_type="factual", top_k=3)
        assert result.results == []
        assert result.sufficient_evidence is False
        assert result.no_relevant_information is True
        with pytest.raises(ValueError):
            agent.retrieve("query", top_k=0)


def test_retrieval_evaluation_marks_ambiguous_queries_not_applicable():
    from evaluation.evaluate_retrieval import build_report, evaluate_query

    class RetrievalMustNotRun:
        def retrieve(self, query, top_k=5):
            raise AssertionError("Ambiguous queries are not retrieval cases")

    row = evaluate_query(
        RetrievalMustNotRun(),
        {
            "query_id": "ambiguous-1",
            "domain": "Software Engineering",
            "category": "ambiguous",
            "query": "What about that?",
            "expected_document": "NONE",
        },
    )
    report = build_report([row])

    assert row["not_applicable"] is True
    assert row["failed"] is False
    assert report["not_applicable_queries"] == 1
    assert report["failed_queries"] == []


def test_retrieval_evaluation_separates_unavailable_no_result_from_false_positive():
    from evaluation.evaluate_retrieval import build_report, evaluate_query

    spec = {
        "query_id": "unavailable-1",
        "domain": "Hospital Administration",
        "category": "unavailable-information",
        "query": "What was the hospital revenue?",
        "expected_document": "NONE",
    }
    no_hit = evaluate_query(RetrievalAgent(FakeRetriever([])), spec)
    false_positive = evaluate_query(
        RetrievalAgent(
            FakeRetriever(
                [{
                    "similarity_score": 0.8,
                    "text": "Patient admission policy.",
                    "document_id": "doc-1",
                    "filename": "admission.pdf",
                    "chunk_id": "chunk-1",
                }]
            ),
            min_relevance=0.0,
        ),
        spec,
    )
    report = build_report([no_hit, false_positive])

    assert no_hit["failed"] is False
    assert no_hit["unavailable_no_result"] is True
    assert false_positive["unavailable_false_positive"] is True
    assert report["unavailable_no_result_queries"] == 1
    assert report["unavailable_false_positive_queries"] == 1
    assert report["failed_queries"] == []


class TestComparisonTargetExtraction:
    """Tests for the new comparison target extraction functionality."""

    def test_extract_comparison_targets_lowercase_multiword(self):
        """Test extraction of lowercase multi-word comparison targets."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare release trains versus identity verification")
        assert len(targets) == 2
        assert "release trains" in targets
        assert "identity verification" in targets

    def test_extract_comparison_targets_uppercase(self):
        """Test extraction of uppercase comparison targets."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare MICROSERVICES versus MONOLITHIC ARCHITECTURE")
        assert len(targets) == 2
        assert "microservices" in targets
        assert "monolithic architecture" in targets

    def test_extract_comparison_targets_mixed_case(self):
        """Test extraction of mixed-case comparison targets."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare Semantic Versioning and Continuous Delivery")
        assert len(targets) == 2
        assert "semantic versioning" in targets
        assert "continuous delivery" in targets

    def test_extract_comparison_targets_versus_pattern(self):
        """Test 'A versus B' pattern extraction."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare release trains versus identity verification")
        assert len(targets) == 2

    def test_extract_comparison_targets_vs_pattern(self):
        """Test 'A vs B' pattern extraction."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare git vs svn")
        assert len(targets) == 2
        assert "git" in targets
        assert "svn" in targets

    def test_extract_comparison_targets_and_pattern(self):
        """Test 'Compare A and B' pattern extraction."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare semantic versioning and continuous delivery")
        assert len(targets) == 2

    def test_extract_comparison_targets_with_pattern(self):
        """Test 'Compare A with B' pattern extraction."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare release trains with identity verification")
        assert len(targets) == 2

    def test_extract_comparison_targets_difference_between(self):
        """Test 'What is the difference between A and B' pattern extraction."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("What is the difference between JWT authentication and session authentication")
        assert len(targets) == 2
        assert "jwt authentication" in targets
        assert "session authentication" in targets

    def test_extract_comparison_targets_differ_from(self):
        """Test 'How does A differ from B' pattern extraction."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("How does JWT authentication differ from session authentication")
        assert len(targets) == 2

    def test_extract_comparison_targets_standalone_vs(self):
        """Test standalone 'A vs B' pattern without 'compare' keyword."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("git vs svn")
        assert len(targets) == 2

    def test_extract_comparison_targets_pronoun_only(self):
        """Test that pronoun-only targets are filtered out."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare them")
        assert len(targets) == 0

    def test_extract_comparison_targets_single_pronoun(self):
        """Test that single pronoun targets are filtered out."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare it with that")
        assert len(targets) == 0

    def test_extract_comparison_targets_no_comparative_pattern(self):
        """Test that non-comparative queries return empty list."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("What is a Git branch?")
        assert len(targets) == 0

    def test_extract_comparison_targets_with_advantages(self):
        """Test extraction with 'advantages of' modifier."""
        agent = QueryUnderstandingAgent()
        targets = agent.extract_comparison_targets("Compare the advantages of semantic versioning and continuous delivery")
        assert len(targets) == 2
        # The filtering removes "advantages of" prefix
        assert "semantic versioning" in targets
        assert "continuous delivery" in targets
