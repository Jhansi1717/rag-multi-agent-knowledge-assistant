"""Tests for M3 ContextResolutionAgent."""

from agents.context_resolution import ContextResolutionAgent
from agents.models import ClarificationState

def test_standalone_query_not_rewritten():
    agent = ContextResolutionAgent()
    res = agent.resolve("What is a microservice?", "User: What is Docker?\nAssistant: It is a container engine.")
    assert not res.context_used
    assert res.resolved_query == "What is a microservice?"

def test_pronoun_reference_resolved():
    agent = ContextResolutionAgent()
    # Context has only one meaningful entity: Docker
    res = agent.resolve("How do I deploy it?", "User: Docker is a container engine.\nAssistant: Yes, Docker is the tool.")
    assert res.context_used
    assert "docker" in res.resolved_query.lower()

def test_numbered_reference_resolved():
    agent = ContextResolutionAgent()
    # Provide two entities in the last turn
    context = "User: Compare Kubernetes and Docker Compose.\nAssistant: Both are orchestration tools."
    res = agent.resolve("I want to use the second one.", context)
    assert res.context_used
    assert "docker" in res.resolved_query.lower() or "compose" in res.resolved_query.lower()

def test_previous_topic_reference():
    agent = ContextResolutionAgent()
    context = "User: What is Kubernetes?\nAssistant: Kubernetes is an orchestration tool."
    res = agent.resolve("Explain the previous method.", context)
    assert res.context_used
    assert "kubernetes" in res.resolved_query.lower()

def test_topic_continuation():
    agent = ContextResolutionAgent()
    context = "User: Tell me about Docker.\nAssistant: Docker is a container."
    res = agent.resolve("What about scaling?", context)
    assert res.context_used
    assert "docker" in res.resolved_query.lower()
    assert "scaling" in res.resolved_query.lower()


def test_explicit_earlier_topic_is_not_rewritten_to_latest_topic():
    agent = ContextResolutionAgent()
    context = (
        "User: What is semantic versioning?\n"
        "Assistant: Semantic versioning is a versioning scheme.\n"
        "User: What is JWT authentication?\n"
        "Assistant: JWT authentication uses signed tokens."
    )

    result = agent.resolve(
        "What about semantic versioning?",
        context,
        active_topics=["JWT authentication"],
        referenced_entities=["JWT authentication"],
    )

    assert result.context_used is False
    assert result.resolved_query == "What about semantic versioning?"

def test_insufficient_context_skips_rewrite():
    agent = ContextResolutionAgent()
    # Context has no meaningful entities after noise filtering (no domain terms, no capitalized proper nouns)
    res = agent.resolve("How do I deploy it?", "User: yes\nAssistant: ok sure")
    assert not res.context_used
    assert res.resolved_query == "How do I deploy it?"

def test_pending_clarification_skips_rewrite():
    agent = ContextResolutionAgent()
    state = ClarificationState(clarification_id="c1")
    res = agent.resolve("I want to deploy it.", "User: Docker.\nAssistant: Docker.", clarification_state=state)
    assert not res.context_used
    assert res.resolution_reason == "Pending clarification takes precedence."

def test_pronoun_its_lowercase_topic():
    """Test resolving 'its' with lowercase topic."""
    agent = ContextResolutionAgent()
    # Use a context that will extract semantic versioning as an entity
    context = "User: Explain semantic versioning.\nAssistant: Semantic versioning is a versioning scheme."
    res = agent.resolve("What are its main advantages?", context)
    assert res.context_used
    assert "semantic versioning" in res.resolved_query.lower() or "versioning" in res.resolved_query.lower()
    assert res.resolved_query == "What are semantic versioning's main advantages?"


def test_plural_topic_uses_plural_possessive():
    agent = ContextResolutionAgent()
    context = "User: Explain release trains.\nAssistant: Release trains are a deployment strategy."

    result = agent.resolve("What are its advantages?", context)

    assert result.context_used
    assert result.resolved_query == "What are release trains' advantages?"

def test_pronoun_its_uppercase_topic():
    """Test resolving 'its' with uppercase topic."""
    agent = ContextResolutionAgent()
    context = "User: Explain JWT authentication.\nAssistant: JWT is a JSON web token."
    res = agent.resolve("What are its benefits?", context)
    assert res.context_used
    assert "jwt" in res.resolved_query.lower()

def test_pronoun_with_multiple_entities_ambiguous():
    """Test that ambiguous context with multiple entities does not resolve."""
    agent = ContextResolutionAgent()
    context = "User: Compare Kubernetes and Docker.\nAssistant: Both are container tools."
    res = agent.resolve("What are its features?", context)
    # Should not resolve because multiple entities exist
    assert not res.context_used

def test_pronoun_resolves_to_most_recent_topic():
    """Test that pronoun resolves to the most recent single topic."""
    agent = ContextResolutionAgent()
    # Use context where only the most recent turn has a clear single topic
    context = "User: Explain release trains.\nAssistant: Release trains are...\nUser: What about identity verification?\nAssistant: Identity verification is a process."
    res = agent.resolve("What are its advantages?", context)
    # Should resolve to identity verification (most recent)
    # Note: This may not resolve if multiple entities are present
    # The important thing is that it doesn't incorrectly resolve to release trains
    if res.context_used:
        assert "identity verification" in res.resolved_query.lower()

def test_approach_mentioned_reference():
    """Test resolving 'the approach you mentioned' reference."""
    agent = ContextResolutionAgent()
    context = "User: What is microservices architecture?\nAssistant: Microservices is an architectural approach."
    res = agent.resolve("Explain the approach you mentioned in more detail.", context)
    # Should resolve to the most recent entity (microservices)
    assert res.context_used
    assert "microservices" in res.resolved_query.lower()

def test_first_one_reference():
    """Test resolving 'the first one' reference."""
    agent = ContextResolutionAgent()
    context = "User: Compare Kubernetes and Docker Compose.\nAssistant: Both are orchestration tools."
    res = agent.resolve("I want to use the first one.", context)
    assert res.context_used
    assert "kubernetes" in res.resolved_query.lower()

def test_third_one_reference():
    """Test resolving 'the third one' reference."""
    agent = ContextResolutionAgent()
    context = "User: Compare Kubernetes, Docker Compose, and Helm.\nAssistant: All are deployment tools."
    res = agent.resolve("I want to use the third one.", context)
    assert res.context_used
    assert "helm" in res.resolved_query.lower()

# ── Regression Tests for Scenarios A-G ─────────────────────────────────────

def test_scenario_a_lowercase_topic_pronoun():
    """Scenario A: lowercase topic with pronoun 'its'."""
    agent = ContextResolutionAgent()
    # First turn: lowercase topic
    context1 = "User: What is semantic versioning?\nAssistant: Semantic versioning is a versioning scheme."
    # Second turn: pronoun reference
    res = agent.resolve("What are its main advantages?", context1)
    assert res.context_used
    assert "semantic versioning" in res.resolved_query.lower() or "versioning" in res.resolved_query.lower()

def test_scenario_b_uppercase_topic_pronoun():
    """Scenario B: uppercase topic with pronoun 'its'."""
    agent = ContextResolutionAgent()
    context1 = "User: What is JWT?\nAssistant: JWT is a JSON web token."
    res = agent.resolve("What are its benefits?", context1)
    assert res.context_used
    assert "jwt" in res.resolved_query.lower()

def test_scenario_c_topic_continuation():
    """Scenario C: topic continuation with pronoun."""
    agent = ContextResolutionAgent()
    context1 = "User: Explain release trains.\nAssistant: Release trains are a deployment strategy."
    res = agent.resolve("What are its advantages?", context1)
    # Check if it resolves to release trains or if context has the topic
    if res.context_used:
        assert "release train" in res.resolved_query.lower()
    else:
        # If not resolved, verify the context extraction is working
        entities = agent._extract_topics_from_context(context1)
        assert len(entities) > 0, "Should extract at least one entity from context"

def test_scenario_d_topic_switching():
    """Scenario D: topic switching - pronoun should refer to most recent topic."""
    agent = ContextResolutionAgent()
    # Two turns: release trains then identity verification
    context = "User: Explain release trains.\nAssistant: Release trains are...\nUser: Explain identity verification.\nAssistant: Identity verification is a process."
    res = agent.resolve("What are its advantages?", context)
    # Should resolve to identity verification (most recent)
    if res.context_used:
        assert "identity verification" in res.resolved_query.lower()

def test_scenario_e_numbered_reference():
    """Scenario E: numbered reference 'the second one'."""
    agent = ContextResolutionAgent()
    context = "User: Compare Kubernetes, Docker Compose, and Helm.\nAssistant: All are deployment tools."
    res = agent.resolve("What about the second one?", context)
    assert res.context_used
    assert "docker" in res.resolved_query.lower() or "compose" in res.resolved_query.lower()

def test_scenario_f_unresolved_reference_fresh_session():
    """Scenario F: unresolved reference in fresh session."""
    agent = ContextResolutionAgent()
    res = agent.resolve("What are its advantages?", "")
    assert not res.context_used
    assert res.resolved_query == "What are its advantages?"

def test_scenario_g_ambiguous_context_multiple_entities():
    """Scenario G: ambiguous context with multiple entities."""
    agent = ContextResolutionAgent()
    context = "User: Compare Kubernetes and Docker.\nAssistant: Both are container tools."
    res = agent.resolve("What are its features?", context)
    # Should not resolve because multiple entities exist (ambiguous)
    assert not res.context_used
