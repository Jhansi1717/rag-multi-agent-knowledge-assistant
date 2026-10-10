from agents.models import (
    QueryUnderstandingResult,
    RetrievalHit,
    ResponseResult,
    RetrievalResult,
    ClarificationResult,
    ClarificationState,
    ContextResolutionResult,
    ConversationTurn,
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

    def extract_conversational_entities(self, text):
        return []

    def extract_entities(self, text):
        return []

    def extract_comparison_targets(self, text):
        return []

    def detect_domain(self, text):
        return None


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
            query="input",
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


def test_response_generation_receives_same_normalized_query_as_retrieval():
    retrieval = FakeRetrieval()
    response = FakeResponse()

    orchestrator(retrieval=retrieval, response=response).handle("  Original query  ")

    query, hits, kwargs = response.calls[0]
    assert query == "input"
    assert query == retrieval.calls[0][0]
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


def test_retrieval_hit_legacy_score_alias_is_distance():
    hit = RetrievalHit(
        rank=1,
        score=1.0,
        text="Evidence.",
        document_id="doc",
        filename="guide.txt",
        chunk_id="chunk",
    )

    assert hit.score == 1.0
    assert hit.distance_score == 1.0
    assert hit.relevance_score == 0.5


class RecordingUnderstanding:
    def __init__(self, query_type="factual"):
        self.query_type = query_type
        self.queries = []

    def analyze(self, query):
        self.queries.append(query)
        normalized = " ".join(query.split())
        return QueryUnderstandingResult(
            query=query,
            normalized_query=normalized,
            query_type=self.query_type,
            classification_confidence=0.9,
            routing="RETRIEVAL",
            domain="Software Engineering",
        )

    def extract_conversational_entities(self, text):
        return []


class EchoRetrieval:
    def __init__(self, hits=None, unavailable=False):
        self.hits = hits if hits is not None else [
            RetrievalHit(
                rank=1,
                relevance_score=0.8,
                distance_score=0.25,
                text="Supported evidence.",
                document_id="doc",
                filename="guide.txt",
                chunk_id="chunk",
            )
        ]
        self.unavailable = unavailable
        self.calls = []
        self.results = None

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        self.results = RetrievalResult(
            query=query,
            query_type=kwargs["query_type"],
            top_k=kwargs["top_k"],
            results=[] if self.unavailable else self.hits,
            retrieval_confidence=0.0 if self.unavailable else 0.8,
            sufficient_evidence=not self.unavailable,
            no_relevant_information=self.unavailable,
        )
        return self.results


class FixedContextResolution:
    def __init__(self, resolved_query, context_used=False):
        self.resolved_query = resolved_query
        self.context_used = context_used

    def resolve(self, query, context, **kwargs):
        return ContextResolutionResult(
            original_query=query,
            resolved_query=self.resolved_query,
            context_used=self.context_used,
            resolution_reason="Resolved test context." if self.context_used else "Standalone query.",
        )


class NoClarification:
    def analyze(self, query):
        return ClarificationResult(original_query=query, clarification_needed=False)


class AlwaysSuccessResponse(FakeResponse):
    def generate(self, query, hits, **kwargs):
        self.calls.append((query, hits, kwargs))
        return ResponseResult(
            answer="Unsupported success.",
            confidence=0.9,
            confidence_level="HIGH",
            grounded=True,
            no_information_found=False,
            query_type=kwargs["query_type"],
        )


def pipeline_orchestrator(query_type="factual", context_resolution=None, retrieval=None, response=None, clarification=None):
    understanding = RecordingUnderstanding(query_type=query_type)
    retrieval = retrieval or EchoRetrieval()
    response = response or FakeResponse()
    instance = Orchestrator(
        vector_store=object(),
        understanding=understanding,
        retrieval=retrieval,
        response_gen=response,
        clarification=clarification or NoClarification(),
    )
    if context_resolution is not None:
        instance.context_resolution = context_resolution
    return instance, understanding, retrieval, response


def test_clarification_answer_resumes_with_refined_query_at_every_stage():
    instance, understanding, retrieval, response = pipeline_orchestrator()
    original_query = "How do I create a branch?"
    session_id = "clarified-session"
    instance.memory.set_pending_clarification(
        session_id,
        ClarificationState(
            original_query=original_query,
            needed=True,
            clarification_id="clarification-1",
            question="Which kind of branch?",
        ),
    )

    result = instance.handle(
        "for Git",
        session_id=session_id,
        clarification_id="clarification-1",
    )

    expected_query = "How do I create a branch? for Git"
    assert result.status == "answered"
    assert understanding.queries == [expected_query]
    assert retrieval.calls[0][0] == expected_query
    assert response.calls[0][0] == expected_query
    assert response.calls[0][1] is retrieval.results.results


def test_context_follow_up_uses_resolved_query_and_marks_memory_used():
    context_resolution = FixedContextResolution(
        "What is the Git feature-branch workflow?", context_used=True
    )
    instance, understanding, retrieval, response = pipeline_orchestrator(
        context_resolution=context_resolution
    )
    instance.memory.append_turn(
        "follow-up-session",
        ConversationTurn(
            session_id="follow-up-session",
            turn_index=0,
            query="Tell me about Git feature branches.",
            answer="They are used for isolated changes.",
        ),
    )

    result = instance.handle("What are its steps?", session_id="follow-up-session")

    expected_query = "What is the Git feature-branch workflow?"
    assert result.status == "answered"
    assert understanding.queries == [expected_query]
    assert retrieval.calls[0][0] == expected_query
    assert response.calls[0][0] == expected_query
    assert result.m3.memory.used_context is True


def test_standalone_query_does_not_claim_irrelevant_memory_was_used():
    instance, _, _, _ = pipeline_orchestrator(
        context_resolution=FixedContextResolution(
            "What is the weather in Paris?", context_used=False
        )
    )
    instance.memory.append_turn(
        "standalone-session",
        ConversationTurn(
            session_id="standalone-session",
            turn_index=0,
            query="Tell me about Git branches.",
            answer="They isolate changes.",
        ),
    )

    result = instance.handle(
        "What is the weather in Paris?", session_id="standalone-session"
    )

    assert result.m3.memory.used_context is False
    assert result.m3.memory.relevant_context == ""


def test_unavailable_retrieval_cannot_become_a_successful_generated_answer():
    retrieval = EchoRetrieval(unavailable=True)
    response = AlwaysSuccessResponse()
    instance, _, _, _ = pipeline_orchestrator(
        retrieval=retrieval,
        response=response,
    )

    result = instance.handle("Unknown fact")

    assert response.calls[0][1] == []
    assert response.calls[0][2]["sufficient_evidence"] is False
    assert response.calls[0][2]["no_relevant_information"] is True
    assert result.status == "unavailable"
    assert result.no_information_found is True
    assert result.grounded is False
    assert result.citations == []


def test_comparative_retrieval_results_reach_generation_intact():
    query_type = "comparative"
    hits = [
        RetrievalHit(
            rank=1, relevance_score=0.8, distance_score=0.25,
            text="Evidence for target A.", document_id="doc-a",
            filename="a.txt", chunk_id="chunk-a",
        ),
        RetrievalHit(
            rank=2, relevance_score=0.7, distance_score=0.43,
            text="Evidence for target B.", document_id="doc-b",
            filename="b.txt", chunk_id="chunk-b",
        ),
    ]
    retrieval = EchoRetrieval(hits=hits)
    instance, _, _, response = pipeline_orchestrator(
        query_type=query_type,
        retrieval=retrieval,
    )

    result = instance.handle("Compare target A and target B.")

    assert result.status == "answered"
    assert response.calls[0][1] is retrieval.results.results
    assert [hit.chunk_id for hit in response.calls[0][1]] == ["chunk-a", "chunk-b"]


def test_context_resolution_failure_identifies_failed_stage():
    class BrokenContextResolution:
        def resolve(self, query, context, **kwargs):
            raise RuntimeError("context store failed")

    instance, _, retrieval, response = pipeline_orchestrator(
        context_resolution=BrokenContextResolution()
    )

    result = instance.handle("What about that?", request_id="context-failure")

    assert result.status == "error"
    assert result.request_id == "context-failure"
    assert result.error.agent == "context_resolution"
    assert result.error.code == "CONTEXT_RESOLUTION_ERROR"
    assert retrieval.calls == []
    assert response.calls == []


def test_memory_failure_identifies_memory_stage():
    instance, _, retrieval, response = pipeline_orchestrator()

    def fail_create_session(session_id):
        raise RuntimeError("session store failed")

    instance.memory.create_session = fail_create_session
    result = instance.handle("A normal query", request_id="memory-failure")

    assert result.status == "error"
    assert result.request_id == "memory-failure"
    assert result.error.agent == "memory"
    assert result.error.code == "MEMORY_ERROR"
    assert retrieval.calls == []
    assert response.calls == []


def test_clarification_failure_identifies_clarification_stage():
    class BrokenClarification:
        def analyze(self, query):
            raise RuntimeError("clarification failed")

    instance, _, retrieval, response = pipeline_orchestrator(
        clarification=BrokenClarification(),
        context_resolution=FixedContextResolution(
            "What is the weather in Paris?", context_used=False
        ),
    )
    result = instance.handle("What is the weather in Paris?", request_id="clar-error")

    assert result.status == "error"
    assert result.error.agent == "clarification"
    assert result.error.code == "CLARIFICATION_ERROR"
    assert retrieval.calls == []
    assert response.calls == []
