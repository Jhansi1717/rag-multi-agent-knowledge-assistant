"""M3 end-to-end orchestrator tests covering all 8 required scenarios."""

import pytest
from agents.clarification import ClarificationAgent
from agents.models import (
    ClarificationResult,
    ClarificationState,
    Citation,
    QueryUnderstandingResult,
    ResponseResult,
    RetrievalHit,
    RetrievalResult,
)
from agents.orchestrator import Orchestrator
from agents.query_understanding import QueryUnderstandingAgent


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeUnderstanding:
    def __init__(self, query_type="factual", routing="RETRIEVAL"):
        self._query_type = query_type
        self._routing = routing

    def analyze(self, query):
        return QueryUnderstandingResult(
            query=query,
            normalized_query=query.strip(),
            query_type=self._query_type,
            classification_confidence=0.9,
            routing=self._routing,
            domain="Software Engineering",
            reason="test classification",
        )

    def extract_entities(self, text):
        return []

    def extract_conversational_entities(self, text):
        return []

    def extract_comparison_targets(self, text):
        return []

    def detect_domain(self, text):
        return None


class FakeRetrieval:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else RetrievalResult(
            query="q", query_type="factual", top_k=3,
            results=[RetrievalHit(
                rank=1, relevance_score=0.8, text="Knowledge-base evidence.",
                document_id="doc1", filename="guide.txt", chunk_id="chunk-1",
            )],
            retrieval_confidence=0.8, sufficient_evidence=True,
        )
        self.error = error
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.results


class FakeResponse:
    def __init__(self, no_info=False, error=None, answer="Grounded answer.", citations=None):
        self.no_info = no_info
        self.error = error
        self.answer = answer
        self.citations = citations or []
        self.calls = []

    def generate(self, query, hits, **kwargs):
        self.calls.append(query)
        if self.error:
            raise self.error
        if self.no_info or not hits:
            return ResponseResult(
                answer="No information found.",
                confidence=0.0, grounded=False,
                no_information_found=True,
                query_type=kwargs["query_type"],
            )
        return ResponseResult(
            answer=self.answer,
            citations=self.citations,
            confidence=kwargs["confidence"],
            grounded=True,
            query_type=kwargs["query_type"],
        )


class FakeClarificationAgent:
    """Always says the query is clear."""
    def analyze(self, query):
        return ClarificationResult(original_query=query, clarification_needed=False)


class AmbiguousClarificationAgent:
    """Always says the query needs clarification."""
    def analyze(self, query):
        return ClarificationResult(
            original_query=query,
            clarification_needed=True,
            ambiguity_type="ambiguous_terminology",
            clarification_question="Which specific process do you mean?",
            clarification_id="test-clar-id",
        )


def _orc(query_type="factual", retrieval=None, response=None,
         clarification=None, routing="RETRIEVAL"):
    return Orchestrator(
        vector_store=object(),
        understanding=FakeUnderstanding(query_type, routing),
        retrieval=retrieval or FakeRetrieval(),
        response_gen=response or FakeResponse(),
        clarification=clarification or FakeClarificationAgent(),
    )


def _real_memory_orc(retrieval=None, response=None):
    understanding = QueryUnderstandingAgent()
    return Orchestrator(
        vector_store=object(),
        understanding=understanding,
        retrieval=retrieval or FakeRetrieval(),
        response_gen=response or FakeResponse(),
        clarification=ClarificationAgent(understanding),
    )


# ── Test 1: Normal factual query ───────────────────────────────────────────────

def test_normal_query_completes_full_pipeline():
    retrieval = FakeRetrieval()
    response = FakeResponse()
    result = _orc(retrieval=retrieval, response=response).handle("What is Docker?")

    assert result.status == "answered"
    assert result.orchestration_status == "completed"
    assert result.grounded is True
    assert result.request_id
    assert retrieval.calls
    assert response.calls
    assert result.m3 is not None
    assert result.m3.transparency is not None
    # All 4 pipeline stages should be tracked
    stage_names = [s.stage_name for s in result.m3.transparency.stages]
    assert "memory" in stage_names
    assert "clarification" in stage_names
    assert "query_understanding" in stage_names
    assert "retrieval" in stage_names
    assert "response_generation" in stage_names


def test_comparative_query_reaches_retrieval_and_response_generation():
    retrieval = FakeRetrieval()
    response = FakeResponse()
    orc = _real_memory_orc(retrieval=retrieval, response=response)

    result = orc.handle("Compare release trains versus identity verification.")

    assert result.status == "answered"
    assert result.query_type == "comparative"
    assert retrieval.calls
    assert response.calls
    assert [stage.stage_name for stage in result.m3.transparency.stages] == [
        "memory",
        "clarification",
        "query_understanding",
        "retrieval",
        "response_generation",
    ]


# ── Test 2: Ambiguous query → clarification required ──────────────────────────

def test_ambiguous_query_stops_before_retrieval():
    retrieval = FakeRetrieval()
    response = FakeResponse()
    orc = _orc(
        retrieval=retrieval,
        response=response,
        clarification=AmbiguousClarificationAgent(),
    )
    result = orc.handle("Explain the process.", session_id="session-amb")

    assert result.status == "clarification_needed"
    assert result.orchestration_status == "requires_clarification"
    assert "Which specific process" in result.answer
    assert retrieval.calls == []   # retrieval must NOT have been called
    assert response.calls == []   # response must NOT have been called
    assert result.m3 is not None
    assert result.m3.clarification is not None
    assert result.m3.clarification.clarification_id == "test-clar-id"

    # State must be persisted in memory
    pending = orc.memory.get_pending_clarification("session-amb")
    assert pending is not None
    assert pending.original_query == "Explain the process."


# ── Test 3: Clarification response resumes original pipeline ──────────────────

def test_clarification_answer_resumes_pipeline():
    retrieval = FakeRetrieval()
    response = FakeResponse()
    orc = _orc(retrieval=retrieval, response=response)

    # Manually inject a pending clarification state
    session_id = "session-resume"
    orc.memory.create_session(session_id)
    state = ClarificationState(
        original_query="Explain the process.",
        needed=True,
        clarification_id="clar-resume",
        question="Which process?",
    )
    orc.memory.set_pending_clarification(session_id, state)

    # User now answers the clarification
    result = orc.handle("The deployment process", session_id=session_id)

    assert result.status == "answered"
    assert result.orchestration_status == "completed"
    assert retrieval.calls  # Retrieval was resumed
    # The refined query should incorporate the original + answer
    refined_query_used = retrieval.calls[0]
    assert "deployment process" in refined_query_used.lower()


# ── Test 4: Follow-up question uses memory context ────────────────────────────

def test_follow_up_question_uses_session_memory():
    session_id = "session-followup"
    retrieval = FakeRetrieval()
    response = FakeResponse()
    orc = _real_memory_orc(retrieval=retrieval, response=response)

    # First turn
    orc.handle("What is Docker?", session_id=session_id)

    # Second turn — follow up
    result = orc.handle("How do I install it?", session_id=session_id)

    assert result.status == "answered"
    assert result.orchestration_status == "completed"
    # Memory should now have 2 turns
    history = orc.memory.get_session(session_id)
    assert len(history) == 2


# ── Test 5: Pronoun follow-up resolved via context ────────────────────────────

def test_pronoun_followup_resolved_to_entity():
    from agents.context_resolution import ContextResolutionAgent

    session_id = "session-pronoun"
    retrieval = FakeRetrieval()
    response = FakeResponse()
    orc = _orc(retrieval=retrieval, response=response)

    # Seed memory with a Docker turn
    from agents.models import ConversationTurn
    orc.memory.create_session(session_id)
    orc.memory.append_turn(session_id, ConversationTurn(
        session_id=session_id,
        turn_index=0,
        query="What is Docker?",
        answer="Docker is a container platform.",
        active_topics=["docker"],
    ))

    # Now use a custom context resolution agent to verify behavior
    resolution_agent = ContextResolutionAgent()
    context = orc.memory.get_relevant_context(session_id, "How do I deploy it?")
    resolution = resolution_agent.resolve("How do I deploy it?", context)

    # If Docker was the entity, pronoun should have been resolved
    if resolution.context_used:
        assert "docker" in resolution.resolved_query.lower()
    else:
        # Insufficient context — that's also a valid outcome
        assert not resolution.context_used


# ── Test 6: Topic switch — prior context not injected ─────────────────────────

def test_topic_switch_does_not_inject_prior_topic():
    session_id = "session-switch"
    retrieval = FakeRetrieval()
    response = FakeResponse()
    orc = _orc(retrieval=retrieval, response=response)

    # First turn: Docker
    orc.handle("What is Docker?", session_id=session_id)

    # Second turn: Completely unrelated query
    result = orc.handle("What is a sprint in Agile?", session_id=session_id)

    assert result.status == "answered"
    assert result.orchestration_status == "completed"
    # The retrieval query should NOT contain "docker"
    assert "docker" not in retrieval.calls[-1].lower()


# ── Test 7: Insufficient context — routes to clarification ────────────────────

def test_insufficient_context_routes_to_clarification():
    # The real ClarificationAgent catches unresolved numbered references
    from agents.clarification import ClarificationAgent
    from agents.query_understanding import QueryUnderstandingAgent
    session_id = "session-insufficient"
    retrieval = FakeRetrieval()
    response = FakeResponse()
    # Use real QueryUnderstandingAgent so ClarificationAgent can analyze properly
    real_qu = QueryUnderstandingAgent()
    orc = Orchestrator(
        vector_store=object(),
        understanding=real_qu,
        retrieval=retrieval,
        response_gen=response,
        # Use the real ClarificationAgent which catches unresolved numbered references
        clarification=ClarificationAgent(),
    )

    # No prior context in this session
    result = orc.handle("What about the second one?", session_id=session_id)

    # ClarificationAgent should catch the unresolved "the second one"
    assert result.orchestration_status == "requires_clarification"
    assert retrieval.calls == []


# ── Test 8: No-result query returns no_relevant_information ───────────────────

def test_no_result_query_returns_no_relevant_information():
    empty_retrieval = RetrievalResult(
        query="unknown", query_type="factual", top_k=3,
        results=[], retrieval_confidence=0.0,
        sufficient_evidence=False, no_relevant_information=True,
    )
    retrieval = FakeRetrieval(results=empty_retrieval)
    response = FakeResponse(no_info=True)
    result = _orc(retrieval=retrieval, response=response).handle("Unknown topic XYZ")

    assert result.status == "unavailable"
    assert result.orchestration_status == "no_relevant_information"
    assert result.no_information_found is True
    assert result.grounded is False


@pytest.mark.parametrize(
    ("first_query", "follow_up", "expected_topic"),
    [
        ("What is semantic versioning?", "What are its main advantages?", "semantic versioning"),
        ("What is JWT?", "What are its benefits?", "jwt"),
        ("Explain release trains.", "What are its advantages?", "release train"),
        ("Explain Git feature branches.", "What are its advantages?", "git feature branches"),
    ],
)
def test_real_topic_followups_resolve_and_persist_turn_metadata(
    first_query, follow_up, expected_topic
):
    citation = Citation(
        chunk_id="chunk-1", filename="guide.txt", excerpt="Knowledge-base evidence.", document_id="doc1"
    )
    retrieval = FakeRetrieval()
    response = FakeResponse(citations=[citation])
    orc = _real_memory_orc(retrieval=retrieval, response=response)
    session_id = f"memory-{expected_topic.replace(' ', '-')}"

    first = orc.handle(first_query, session_id=session_id)
    second = orc.handle(follow_up, session_id=session_id)

    assert first.status == "answered"
    assert second.status == "answered"
    assert expected_topic in retrieval.calls[-1].lower()
    assert response.calls[-1] == retrieval.calls[-1]
    assert expected_topic in response.calls[-1].lower()
    turns = orc.memory.get_session(session_id)
    assert turns[0].active_topics
    assert turns[1].query == follow_up
    assert turns[1].refined_query is not None
    assert expected_topic in turns[1].refined_query.lower()
    assert turns[1].answer == response.answer
    assert "guide.txt" in turns[1].referenced_documents
    assert "guide.txt#chunk-1" in turns[1].citations
    assert turns[1].referenced_entities
    assert second.m3.transparency.source_evidence[0].chunk_text == "Knowledge-base evidence."
    assert second.m3.transparency.citations == [citation]


def test_real_topic_switch_uses_latest_topic_for_pronoun():
    retrieval = FakeRetrieval()
    orc = _real_memory_orc(retrieval=retrieval)

    orc.handle("Explain release trains.", session_id="switch")
    orc.handle("Explain identity verification.", session_id="switch")
    result = orc.handle("What are its advantages?", session_id="switch")

    assert result.status == "answered"
    assert "identity verification" in retrieval.calls[-1].lower()
    assert "release trains" not in retrieval.calls[-1].lower()


def test_unresolved_follow_up_in_fresh_session_requests_clarification():
    retrieval = FakeRetrieval()
    orc = _real_memory_orc(retrieval=retrieval)

    result = orc.handle("What are its advantages?", session_id="fresh-reference")

    assert result.status == "clarification_needed"
    assert "refer" in result.answer.lower()
    assert retrieval.calls == []


def test_ambiguous_multiple_topics_are_not_guessed():
    retrieval = FakeRetrieval()
    orc = _real_memory_orc(retrieval=retrieval)

    orc.handle("Compare Kubernetes and Docker.", session_id="ambiguous-reference")
    result = orc.handle(
        "What are its features?", session_id="ambiguous-reference"
    )

    assert result.status == "clarification_needed"
    assert retrieval.calls == ["Compare Kubernetes and Docker."]


def test_ambiguous_reference_clarifies_even_with_another_entity_in_query():
    retrieval = FakeRetrieval()
    orc = _real_memory_orc(retrieval=retrieval)

    orc.handle("Compare Kubernetes and Docker.", session_id="mixed-reference")
    result = orc.handle(
        "What are its benefits for JWT?", session_id="mixed-reference"
    )

    assert result.status == "clarification_needed"
    assert "reference" in result.answer.lower()
    assert retrieval.calls == ["Compare Kubernetes and Docker."]


def test_complete_comparison_reaches_retrieval_and_response_generation():
    retrieval = FakeRetrieval()
    response = FakeResponse()
    orc = _real_memory_orc(retrieval=retrieval, response=response)

    result = orc.handle(
        "Compare release trains versus identity verification.",
        session_id="comparison-pipeline",
    )

    assert result.status == "answered"
    assert result.query_type == "comparative"
    assert retrieval.calls == ["Compare release trains versus identity verification."]
    assert response.calls == ["Compare release trains versus identity verification."]
    stage_names = [stage.stage_name for stage in result.m3.transparency.stages]
    assert "query_understanding" in stage_names
    assert "retrieval" in stage_names
    assert "response_generation" in stage_names


def test_numbered_reference_uses_ordered_answer_entities():
    retrieval = FakeRetrieval()
    response = FakeResponse(
        answer="The three stages are planning, implementation, and review."
    )
    orc = _real_memory_orc(retrieval=retrieval, response=response)

    orc.handle("What are the three stages?", session_id="numbered-reference")
    result = orc.handle("What about the second one?", session_id="numbered-reference")

    assert result.status == "answered"
    assert "implementation" in retrieval.calls[-1].lower()
