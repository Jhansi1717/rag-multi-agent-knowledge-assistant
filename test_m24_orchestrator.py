from agents.models import (
    QueryUnderstandingResult,
    RetrievalHit,
    ResponseResult,
    RetrievalResult,
)
from agents.orchestrator import Orchestrator


class FakeUnderstanding:
    def __init__(self, query_type="factual", routing="RETRIEVAL"):
        self.result = QueryUnderstandingResult(
            query="input",
            normalized_query="input",
            query_type=query_type,
            classification_confidence=0.9,
            routing=routing,
            domain="Software Engineering",
            reason="test",
        )

    def analyze(self, query):
        return self.result


class FakeRetrieval:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else RetrievalResult(
            query="input",
            query_type="factual",
            top_k=3,
            results=[
                RetrievalHit(
                    rank=1,
                    relevance_score=0.8,
                    distance_score=0.25,
                    text="Supported evidence.",
                    document_id="doc",
                    filename="guide.txt",
                    chunk_id="chunk",
                )
            ],
            retrieval_confidence=0.8,
            sufficient_evidence=True,
        )
        self.error = error
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return self.results


class FakeResponse:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def generate(self, query, hits, **kwargs):
        self.calls.append((query, hits, kwargs))
        if self.error:
            raise self.error
        if not hits:
            return ResponseResult(
                answer="No information found.",
                confidence=0.0,
                confidence_level="LOW",
                grounded=False,
                no_information_found=True,
                query_type=kwargs["query_type"],
            )
        return ResponseResult(
            answer="Grounded test answer.",
            confidence=kwargs["confidence"],
            confidence_level="HIGH",
            grounded=True,
            query_type=kwargs["query_type"],
        )


def orchestrator(query_type="factual", routing="RETRIEVAL", retrieval=None, response=None):
    return Orchestrator(
        vector_store=object(),
        understanding=FakeUnderstanding(query_type, routing),
        retrieval=retrieval or FakeRetrieval(),
        response_gen=response or FakeResponse(),
    )


def test_factual_procedural_and_comparative_follow_sequential_flow():
    for query_type in ("factual", "procedural", "comparative"):
        retrieval = FakeRetrieval()
        response = FakeResponse()
        result = orchestrator(query_type, retrieval=retrieval, response=response).handle("input")
        assert result.status == "answered"
        assert result.query_type == query_type
        assert result.grounded is True
        assert result.request_id
        assert result.retrieval is retrieval.results
        assert retrieval.calls
        assert response.calls


def test_ambiguous_returns_structured_clarification_without_retrieval():
    retrieval = FakeRetrieval()
    response = FakeResponse()
    result = orchestrator(
        "ambiguous",
        "CLARIFICATION",
        retrieval=retrieval,
        response=response,
    ).handle("this")
    assert result.status == "clarification_needed"
    assert result.query_type == "ambiguous"
    assert result.request_id
    assert retrieval.calls == []
    assert response.calls == []


def test_unavailable_is_determined_by_empty_retrieval_evidence():
    retrieval = FakeRetrieval(
        RetrievalResult(
            query="unknown",
            query_type="factual",
            top_k=3,
            retrieval_confidence=0.0,
            sufficient_evidence=False,
            no_relevant_information=True,
        )
    )
    response = FakeResponse()
    result = orchestrator(retrieval=retrieval, response=response).handle("unknown")
    assert result.status == "unavailable"
    assert result.no_information_found is True
    assert result.grounded is False
    assert result.retrieval.no_relevant_information is True
    assert response.calls


def test_retrieval_failure_returns_structured_error():
    result = orchestrator(
        retrieval=FakeRetrieval(error=RuntimeError("search failed"))
    ).handle("input", request_id="req-retrieval")
    assert result.status == "error"
    assert result.request_id == "req-retrieval"
    assert result.error.agent == "retrieval"
    assert result.error.code == "RETRIEVAL_ERROR"


def test_response_generation_failure_returns_structured_error():
    result = orchestrator(
        response=FakeResponse(error=RuntimeError("llm failed"))
    ).handle("input")
    assert result.status == "error"
    assert result.error.agent == "response_generation"
    assert result.error.code == "RESPONSE_GENERATION_ERROR"


def test_empty_or_all_filtered_retrieval_is_handed_to_response_agent():
    retrieval_result = RetrievalResult(
        query="input",
        query_type="factual",
        top_k=3,
        results=[],
        filtered_count=3,
        retrieval_confidence=0.0,
        sufficient_evidence=False,
        no_relevant_information=True,
    )
    retrieval = FakeRetrieval(retrieval_result)
    response = FakeResponse()

    result = orchestrator(retrieval=retrieval, response=response).handle("input")

    assert response.calls
    assert response.calls[0][1] == []
    assert response.calls[0][2]["confidence"] == 0.0
    assert result.retrieval.filtered_count == 3
    assert result.retrieval.no_relevant_information is True


def test_response_generation_receives_original_query_and_structured_retrieval():
    retrieval = FakeRetrieval()
    response = FakeResponse()

    orchestrator(retrieval=retrieval, response=response).handle("  Original query  ")

    query, hits, kwargs = response.calls[0]
    assert query == "  Original query  "
    assert hits is retrieval.results.results
    assert kwargs["query_type"] == "factual"
    assert kwargs["confidence"] == retrieval.results.retrieval_confidence
    assert kwargs["request_id"]
    assert retrieval.calls[0][1]["request_id"] == kwargs["request_id"]


def test_malformed_retrieval_output_returns_structured_error():
    class MalformedRetrieval:
        def retrieve(self, query, **kwargs):
            return {"results": []}

    result = orchestrator(
        retrieval=MalformedRetrieval(),
        response=FakeResponse(),
    ).handle("input", request_id="req-malformed-retrieval")

    assert result.status == "error"
    assert result.request_id == "req-malformed-retrieval"
    assert result.error.code == "RETRIEVAL_ERROR"


def test_malformed_response_output_returns_structured_error():
    class MalformedResponse:
        def generate(self, query, hits, **kwargs):
            return {"answer": "not a ResponseResult"}

    result = orchestrator(
        retrieval=FakeRetrieval(),
        response=MalformedResponse(),
    ).handle("input", request_id="req-malformed-response")

    assert result.status == "error"
    assert result.request_id == "req-malformed-response"
    assert result.error.code == "RESPONSE_GENERATION_ERROR"


def test_invalid_understanding_output_returns_structured_error():
    class InvalidUnderstanding:
        def analyze(self, query):
            return QueryUnderstandingResult(
                query=query,
                normalized_query=query,
                query_type="factual",
                classification_confidence=0.9,
                routing="INVALID",
            )

    result = Orchestrator(
        vector_store=object(),
        understanding=InvalidUnderstanding(),
        retrieval=FakeRetrieval(),
        response_gen=FakeResponse(),
    ).handle("input", request_id="req-invalid-routing")

    assert result.status == "error"
    assert result.request_id == "req-invalid-routing"
    assert result.error.code == "CLASSIFICATION_ERROR"


def test_classification_failure_returns_structured_error():
    class BrokenUnderstanding:
        def analyze(self, query):
            raise RuntimeError("classifier failed")

    result = Orchestrator(
        vector_store=object(),
        understanding=BrokenUnderstanding(),
        retrieval=FakeRetrieval(),
        response_gen=FakeResponse(),
    ).handle("input", request_id="req-classification")
    assert result.status == "error"
    assert result.request_id == "req-classification"
    assert result.error.agent == "query_understanding"
    assert result.error.code == "CLASSIFICATION_ERROR"
