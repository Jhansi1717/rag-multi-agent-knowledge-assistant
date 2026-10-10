"""Tests for M3 Bounded Conversation Memory Service."""

import pytest
from agents.memory import ConversationMemoryAgent
from agents.models import ClarificationState, ConversationTurn

def create_turn(session_id: str, index: int, query: str = "q", answer: str = "a", topics=None, docs=None):
    return ConversationTurn(
        session_id=session_id,
        turn_index=index,
        query=query,
        answer=answer,
        active_topics=topics or [],
        referenced_documents=docs or []
    )

def test_session_isolation():
    mem = ConversationMemoryAgent()
    mem.append_turn("s1", create_turn("s1", 0, "q1", "a1"))
    mem.append_turn("s2", create_turn("s2", 0, "q2", "a2"))

    assert len(mem.get_session("s1")) == 1
    assert mem.get_session("s1")[0].query == "q1"
    assert len(mem.get_session("s2")) == 1
    assert mem.get_session("s2")[0].query == "q2"

def test_bounded_history_turns():
    mem = ConversationMemoryAgent(max_turns=3)
    for i in range(5):
        mem.append_turn("s1", create_turn("s1", 0, f"q{i}", f"a{i}"))

    history = mem.get_session("s1")
    assert len(history) == 3
    assert history[0].query == "q2"
    assert history[2].query == "q4"
    assert history[0].turn_index == 0
    assert history[2].turn_index == 2

def test_bounded_history_characters():
    mem = ConversationMemoryAgent(max_characters=25) # small limit
    mem.append_turn("s1", create_turn("s1", 0, "short", "answer"))
    mem.append_turn("s1", create_turn("s1", 1, "a very long query here", "and a very long answer here"))

    context = mem.get_recent_context("s1", limit=5)
    assert len(context) <= 25
    # Should truncate the start of the string to fit 25 chars
    assert context == "d a very long answer here"

def test_follow_up_context():
    mem = ConversationMemoryAgent()
    mem.append_turn(
        "s1", create_turn("s1", 0, "How to deploy?", "Use docker", topics=["docker"])
    )
    mem.append_turn(
        "s1", create_turn("s1", 1, "Next step?", "Use k8s", topics=["k8s"])
    )

    context = mem.get_relevant_context("s1", "What about docker?")
    assert "How to deploy?" in context
    assert "Use docker" in context
    assert "Use k8s" not in context


def test_unrelated_standalone_query_gets_no_prior_context():
    mem = ConversationMemoryAgent()
    mem.append_turn(
        "s1",
        create_turn(
            "s1",
            0,
            "What is semantic versioning?",
            "Semantic versioning labels releases.",
            topics=["semantic versioning"],
        ),
    )

    context = mem.get_relevant_context("s1", "How do hospital admissions work?")

    assert context == ""

def test_topic_switching_relevance():
    mem = ConversationMemoryAgent(max_characters=150) # Small limit to force eviction

    # Old relevant turn
    mem.append_turn("s1", create_turn("s1", 0, "Tell me about Docker", "Docker is a container.", topics=["docker"]))

    # Irrelevant turns
    for i in range(3):
        mem.append_turn("s1", create_turn("s1", i+1, f"filler {i}", "filler", topics=["filler"]))

    # Recent turns
    mem.append_turn("s1", create_turn("s1", 4, "Recent query", "Recent answer"))

    # Query matching old turn
    context = mem.get_relevant_context("s1", "How do I secure Docker?")

    # Include the matching Docker turn without adding unrelated recent history.
    assert "Docker is a container" in context
    assert "Recent query" not in context
    assert "filler 0" not in context

def test_clear_session():
    mem = ConversationMemoryAgent()
    mem.append_turn("s1", create_turn("s1", 0))
    mem.set_pending_clarification("s1", ClarificationState(clarification_id="c1"))

    mem.clear_session("s1")

    with pytest.raises(ValueError):
        mem.get_session("s1")
    assert mem.get_pending_clarification("s1") is None

def test_missing_session():
    mem = ConversationMemoryAgent()
    with pytest.raises(ValueError, match="Unknown session"):
        mem.get_session("missing")

def test_clarification_lifecycle():
    mem = ConversationMemoryAgent()
    mem.set_pending_clarification("s1", ClarificationState(clarification_id="c1", original_query="test"))

    refined = mem.resolve_clarification("s1", "c1", "answer")
    assert "test answer" in refined
    state = mem.get_pending_clarification("s1")
    assert state.clarification_answer == "answer"

    # Adding turn should clear it
    mem.append_turn("s1", create_turn("s1", 0))
    assert mem.get_pending_clarification("s1") is None


def test_clarification_limit_survives_storing_clarification_turns():
    mem = ConversationMemoryAgent(max_clarifications=2)
    for index in range(2):
        state = ClarificationState(
            clarification_id=f"clarification-{index}",
            original_query=f"Explain process {index}.",
            question="Which process?",
        )
        mem.set_pending_clarification("clarification-session", state)
        mem.append_turn(
            "clarification-session",
            create_turn(
                "clarification-session",
                index,
                state.original_query,
                "Which process?",
            ),
        )

    with pytest.raises(ValueError, match="Maximum clarification loops"):
        mem.set_pending_clarification(
            "clarification-session",
            ClarificationState(clarification_id="clarification-overflow"),
        )

def test_active_topics_populated():
    """Test that active_topics are extracted and stored in ConversationTurn."""
    from agents.query_understanding import QueryUnderstandingAgent
    mem = ConversationMemoryAgent()

    turn = create_turn("s1", 0, "What is semantic versioning?", "Semantic versioning is a versioning scheme.")
    # Manually extract topics as the orchestrator would
    qu = QueryUnderstandingAgent()
    query_topics = qu.extract_conversational_entities("What is semantic versioning?")
    answer_topics = qu.extract_conversational_entities("Semantic versioning is a versioning scheme.")
    turn.active_topics = list(set(query_topics + answer_topics))

    mem.append_turn("s1", turn)

    retrieved = mem.get_session("s1")[0]
    assert len(retrieved.active_topics) > 0
    assert any("semantic versioning" in topic.lower() for topic in retrieved.active_topics)

def test_lowercase_topic_pronoun_resolution():
    """Test scenario A: lowercase topic with pronoun 'its'."""
    from agents.query_understanding import QueryUnderstandingAgent
    mem = ConversationMemoryAgent()
    qu = QueryUnderstandingAgent()

    # First turn: lowercase topic
    turn1 = create_turn("s1", 0, "What is semantic versioning?", "Semantic versioning is a versioning scheme.")
    query_topics = qu.extract_conversational_entities("What is semantic versioning?")
    answer_topics = qu.extract_conversational_entities("Semantic versioning is a versioning scheme.")
    turn1.active_topics = list(set(query_topics + answer_topics))
    mem.append_turn("s1", turn1)

    # Second turn: pronoun reference
    context = mem.get_relevant_context("s1", "What are its main advantages?")
    assert "semantic versioning" in context.lower() or "versioning" in context.lower()

def test_uppercase_topic_pronoun_resolution():
    """Test scenario B: uppercase topic with pronoun 'its'."""
    from agents.query_understanding import QueryUnderstandingAgent
    mem = ConversationMemoryAgent()
    qu = QueryUnderstandingAgent()

    # First turn: uppercase topic
    turn1 = create_turn("s1", 0, "What is JWT?", "JWT is a JSON web token.")
    query_topics = qu.extract_conversational_entities("What is JWT?")
    answer_topics = qu.extract_conversational_entities("JWT is a JSON web token.")
    turn1.active_topics = list(set(query_topics + answer_topics))
    mem.append_turn("s1", turn1)

    # Second turn: pronoun reference
    context = mem.get_relevant_context("s1", "What are its benefits?")
    assert "jwt" in context.lower()

def test_topic_continuation():
    """Test scenario C: topic continuation with pronoun."""
    from agents.query_understanding import QueryUnderstandingAgent
    mem = ConversationMemoryAgent()
    qu = QueryUnderstandingAgent()

    # First turn
    turn1 = create_turn("s1", 0, "Explain release trains.", "Release trains are...")
    query_topics = qu.extract_conversational_entities("Explain release trains.")
    answer_topics = qu.extract_conversational_entities("Release trains are...")
    turn1.active_topics = list(set(query_topics + answer_topics))
    mem.append_turn("s1", turn1)

    # Second turn: continuation
    context = mem.get_relevant_context("s1", "What are its advantages?")
    assert "release train" in context.lower()

def test_topic_switching():
    """Test scenario D: topic switching - pronoun should refer to most recent topic."""
    from agents.query_understanding import QueryUnderstandingAgent
    mem = ConversationMemoryAgent()
    qu = QueryUnderstandingAgent()

    # First turn: release trains
    turn1 = create_turn("s1", 0, "Explain release trains.", "Release trains are...")
    query_topics1 = qu.extract_conversational_entities("Explain release trains.")
    answer_topics1 = qu.extract_conversational_entities("Release trains are...")
    turn1.active_topics = list(set(query_topics1 + answer_topics1))
    mem.append_turn("s1", turn1)

    # Second turn: identity verification (topic switch)
    turn2 = create_turn("s1", 1, "Explain identity verification.", "Identity verification is...")
    query_topics2 = qu.extract_conversational_entities("Explain identity verification.")
    answer_topics2 = qu.extract_conversational_entities("Identity verification is...")
    turn2.active_topics = list(set(query_topics2 + answer_topics2))
    mem.append_turn("s1", turn2)

    # Third turn: pronoun should refer to identity verification (most recent)
    context = mem.get_relevant_context("s1", "What are its advantages?")
    # The most recent topic should be prioritized
    assert "identity verification" in context.lower()

def test_numbered_reference():
    """Test scenario E: numbered reference 'the second one'."""
    from agents.query_understanding import QueryUnderstandingAgent
    mem = ConversationMemoryAgent()
    qu = QueryUnderstandingAgent()

    # First turn: multiple entities
    turn1 = create_turn("s1", 0, "What are the three approaches?", "They are A, B, and C.")
    query_topics = qu.extract_conversational_entities("What are the three approaches?")
    answer_topics = qu.extract_conversational_entities("They are A, B, and C.")
    turn1.active_topics = list(set(query_topics + answer_topics))
    mem.append_turn("s1", turn1)

    # Second turn: numbered reference
    context = mem.get_relevant_context("s1", "What about the second one?")
    # Context should include the first turn with the three approaches
    assert "three approaches" in context.lower() or "A, B, and C" in context.lower()

def test_unresolved_reference_fresh_session():
    """Test scenario F: unresolved reference in fresh session."""
    mem = ConversationMemoryAgent()

    # No prior context
    context = mem.get_relevant_context("s1", "What are its advantages?")
    assert context == ""

def test_fallback_entity_extraction_without_active_topics():
    """Test that context extraction works even when active_topics is empty."""
    from agents.query_understanding import QueryUnderstandingAgent
    mem = ConversationMemoryAgent()
    qu = QueryUnderstandingAgent()

    # Turn without active_topics (old data)
    turn1 = create_turn("s1", 0, "What is semantic versioning?", "Semantic versioning is a versioning scheme.", topics=[])
    mem.append_turn("s1", turn1)

    # Should still extract entities from query and answer
    context = mem.get_relevant_context("s1", "What are its main advantages?")
    assert "semantic versioning" in context.lower() or "versioning" in context.lower()


def test_extracts_unlisted_lowercase_document_concepts():
    from agents.query_understanding import QueryUnderstandingAgent

    topics = QueryUnderstandingAgent().extract_conversational_entities(
        "What is database sharding?"
    )
    assert "database sharding" in [topic.lower() for topic in topics]


def test_relevant_context_matches_referenced_entities():
    mem = ConversationMemoryAgent(max_characters=200)
    turn = create_turn(
        "s1", 0, "What is database sharding?", "It partitions data across nodes."
    )
    turn.referenced_entities = ["database sharding"]
    mem.append_turn("s1", turn)

    context = mem.get_relevant_context("s1", "Tell me about database sharding")
    assert "database sharding" in context.lower()


class RecordingUnavailableRetrieval:
    """Capture query handoffs without fabricating model-generated answers."""

    def __init__(self):
        self.queries = []

    def retrieve(self, query, query_type="factual", top_k=3, **kwargs):
        from agents.models import RetrievalResult

        self.queries.append(query)
        return RetrievalResult(
            query=query,
            query_type=query_type,
            top_k=top_k,
            results=[],
            retrieval_confidence=0.0,
            sufficient_evidence=False,
            no_relevant_information=True,
        )


def create_real_no_provider_orchestrator():
    from agents.orchestrator import Orchestrator
    from agents.query_understanding import QueryUnderstandingAgent
    from agents.response_generation import ResponseGenerationAgent

    retrieval = RecordingUnavailableRetrieval()
    orchestrator = Orchestrator(
        vector_store=object(),
        understanding=QueryUnderstandingAgent(),
        retrieval=retrieval,
        response_gen=ResponseGenerationAgent(api_key="", llm_client=None),
    )
    return orchestrator, retrieval


def test_end_to_end_follow_up_a_semantic_versioning_uses_required_context():
    orchestrator, retrieval = create_real_no_provider_orchestrator()
    session_id = "memory-scenario-a"

    orchestrator.handle("What is semantic versioning?", session_id=session_id)
    result = orchestrator.handle(
        "What are its main advantages?", session_id=session_id
    )

    assert "semantic versioning" in retrieval.queries[-1].lower()
    assert result.m3.memory.used_context is True
    assert "semantic versioning" in result.m3.memory.relevant_context.lower()
    assert result.status == "unavailable"  # No provider/evidence; state contract only.


def test_end_to_end_topic_switch_then_follow_up_uses_topic_b():
    orchestrator, retrieval = create_real_no_provider_orchestrator()
    session_id = "memory-scenario-b"

    orchestrator.handle("Tell me about semantic versioning.", session_id=session_id)
    orchestrator.handle("Now tell me about JWT authentication.", session_id=session_id)
    result = orchestrator.handle("What are its benefits?", session_id=session_id)

    assert "jwt" in retrieval.queries[-1].lower()
    assert "semantic versioning" not in retrieval.queries[-1].lower()
    assert result.m3.memory.used_context is True
    assert result.status == "unavailable"


def test_end_to_end_return_to_explicit_earlier_topic_after_switch():
    orchestrator, retrieval = create_real_no_provider_orchestrator()
    session_id = "memory-scenario-c"

    orchestrator.handle("What is semantic versioning?", session_id=session_id)
    orchestrator.handle("What is JWT authentication?", session_id=session_id)
    result = orchestrator.handle(
        "What about semantic versioning?", session_id=session_id
    )

    assert "semantic versioning" in retrieval.queries[-1].lower()
    assert "jwt authentication" not in retrieval.queries[-1].lower()
    assert result.m3.memory.used_context is False


def test_new_session_does_not_inherit_history_or_document_references():
    orchestrator, retrieval = create_real_no_provider_orchestrator()
    orchestrator.handle("What is semantic versioning?", session_id="shared-client-a")
    orchestrator.memory.get_session("shared-client-a")[-1].referenced_documents = [
        "private-source.pdf"
    ]

    result = orchestrator.handle(
        "What about the second one?", session_id="shared-client-b"
    )

    assert result.status == "clarification_needed"
    assert retrieval.queries == ["What is semantic versioning?"]
    assert result.m3.memory.relevant_context == ""
    assert result.m3.memory.used_context is False
    assert "private-source.pdf" not in orchestrator.memory.get_relevant_context(
        "shared-client-b", "What about the second one?"
    )


def test_ambiguous_follow_up_without_sufficient_context_clarifies():
    orchestrator, retrieval = create_real_no_provider_orchestrator()

    result = orchestrator.handle(
        "What about the second one?", session_id="empty-context-session"
    )

    assert result.status == "clarification_needed"
    assert result.clarification_question
    assert retrieval.queries == []


def test_clarification_answer_resumes_original_query_and_clears_state():
    orchestrator, retrieval = create_real_no_provider_orchestrator()
    session_id = "clarification-resume-session"
    initial = orchestrator.handle("Explain the process.", session_id=session_id)

    assert initial.status == "clarification_needed"
    clarification_id = initial.m3.clarification.clarification_id
    resumed = orchestrator.handle(
        "I mean deployment workflow",
        session_id=session_id,
        clarification_id=clarification_id,
    )

    assert "deployment workflow" in retrieval.queries[-1].lower()
    assert "explain" in retrieval.queries[-1].lower()
    assert resumed.status == "unavailable"
    assert orchestrator.memory.get_pending_clarification(session_id) is None
    turns = orchestrator.memory.get_session(session_id)
    assert any(turn.clarification_answer == "I mean deployment workflow" for turn in turns)
    assert any(
        turn.refined_query and "deployment workflow" in turn.refined_query.lower()
        for turn in turns
    )
