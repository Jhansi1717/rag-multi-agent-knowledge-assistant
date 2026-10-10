"""Tests for the M3 ClarificationAgent logic."""

from agents.clarification import ClarificationAgent

def test_clear_query_does_not_need_clarification():
    agent = ClarificationAgent()
    result = agent.analyze("What is a Git branch?")
    assert not result.clarification_needed
    assert result.original_query == "What is a Git branch?"
    assert result.clarification_question is None
    assert result.clarification_id is None

def test_empty_query_returns_incomplete():
    agent = ClarificationAgent()
    result = agent.analyze("")
    assert result.clarification_needed
    assert result.ambiguity_type == "incomplete"
    assert "What would you like to know" in result.clarification_question
    assert result.clarification_id is not None

def test_whitespace_query_returns_incomplete():
    agent = ClarificationAgent()
    result = agent.analyze("   ")
    assert result.clarification_needed
    assert result.ambiguity_type == "incomplete"

def test_multi_part_request():
    agent = ClarificationAgent()
    result = agent.analyze("Explain the git process and compare it to SVN")
    assert result.clarification_needed
    assert result.ambiguity_type == "multi_part"
    assert "multiple topics" in result.clarification_question

def test_context_dependent_request():
    agent = ClarificationAgent()
    result = agent.analyze("Explain the next step.")
    assert result.clarification_needed
    assert result.ambiguity_type == "context_dependent"
    assert "process or previous step" in result.clarification_question

def test_unresolved_pronoun():
    agent = ClarificationAgent()
    result = agent.analyze("What about the second one?")
    assert result.clarification_needed
    assert result.ambiguity_type == "unresolved_reference"
    assert "second one" in result.clarification_question

def test_incomplete_request_detects_missing_entities():
    agent = ClarificationAgent()
    result = agent.analyze("Compare the two.")
    assert result.clarification_needed
    assert result.ambiguity_type == "incomplete"
    assert "compare" in result.clarification_question

def test_ambiguous_terminology_detects_generic_nouns():
    agent = ClarificationAgent()
    result = agent.analyze("Explain the process.")
    assert result.clarification_needed
    assert result.ambiguity_type == "ambiguous_terminology"
    assert "process" in result.clarification_question


def test_ambiguous_terminology_uses_correct_plural_forms():
    agent = ClarificationAgent()
    expected_phrases = {
        "Explain the process.": "multiple processes",
        "What are the requirements?": "multiple requirements",
        "Explain the system.": "multiple systems",
        "Explain the workflow.": "multiple workflows",
    }

    for query, phrase in expected_phrases.items():
        result = agent.analyze(query)
        assert result.clarification_needed
        assert phrase in result.clarification_question.lower()
        assert "processs" not in result.clarification_question.lower()


def test_missing_object_detects_requirements():
    agent = ClarificationAgent()
    result = agent.analyze("What are the requirements?")
    assert result.clarification_needed
    # "the requirements" triggers the ambiguous terminology check
    assert result.ambiguity_type == "ambiguous_terminology"
    assert "requirements" in result.clarification_question

# Comparative query regression tests

def test_comparative_query_with_lowercase_targets():
    """Test that lowercase multi-word comparison targets are recognized."""
    agent = ClarificationAgent()
    query = "Compare release trains versus identity verification."
    result = agent.analyze(query)
    assert not result.clarification_needed
    assert result.original_query == query
    from agents.query_understanding import QueryUnderstandingAgent
    assert QueryUnderstandingAgent().extract_comparison_targets(query) == [
        "release trains", "identity verification"
    ]

def test_comparative_query_with_uppercase_targets():
    """Test that uppercase comparison targets are recognized."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare MICROSERVICES versus MONOLITHIC ARCHITECTURE")
    assert not result.clarification_needed
    assert result.original_query == "Compare MICROSERVICES versus MONOLITHIC ARCHITECTURE"

def test_comparative_query_with_mixed_case_targets():
    """Test that mixed-case comparison targets are recognized."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare Semantic Versioning and Continuous Delivery")
    assert not result.clarification_needed
    assert result.original_query == "Compare Semantic Versioning and Continuous Delivery"

def test_comparative_query_versus_pattern():
    """Test 'A versus B' pattern."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare release trains versus identity verification")
    assert not result.clarification_needed

def test_comparative_query_vs_pattern():
    """Test 'A vs B' pattern."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare git vs svn")
    assert not result.clarification_needed

def test_comparative_query_and_pattern():
    """Test 'Compare A and B' pattern."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare semantic versioning and continuous delivery")
    assert not result.clarification_needed

def test_comparative_query_with_pattern():
    """Test 'Compare A with B' pattern."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare release trains with identity verification")
    assert not result.clarification_needed

def test_comparative_query_difference_between_pattern():
    """Test 'What is the difference between A and B' pattern."""
    agent = ClarificationAgent()
    result = agent.analyze("What is the difference between JWT authentication and session authentication")
    assert not result.clarification_needed

def test_comparative_query_differ_from_pattern():
    """Test 'How does A differ from B' pattern."""
    agent = ClarificationAgent()
    result = agent.analyze("How does JWT authentication differ from session authentication")
    assert not result.clarification_needed

def test_comparative_query_standalone_vs_pattern():
    """Test standalone versus forms without the 'compare' keyword."""
    agent = ClarificationAgent()
    for query in ("git vs svn", "release trains versus identity verification"):
        result = agent.analyze(query)
        assert not result.clarification_needed

def test_comparative_query_missing_targets():
    """Test that comparative queries without targets trigger clarification."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare the two")
    assert result.clarification_needed
    assert result.ambiguity_type == "incomplete"
    assert "compare" in result.clarification_question.lower()


def test_comparative_query_difference_without_targets_triggers_clarification():
    agent = ClarificationAgent()
    result = agent.analyze("What is the difference?")
    assert result.clarification_needed
    assert result.ambiguity_type == "incomplete"

def test_comparative_query_pronoun_only_targets():
    """Test that comparative queries with pronoun-only targets trigger clarification."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare them.")
    assert result.clarification_needed
    # Pronoun-only queries are caught by incomplete check since they have no entities
    assert result.ambiguity_type == "incomplete"


def test_comparative_query_rejects_one_concrete_target_and_one_pronoun():
    agent = ClarificationAgent()
    result = agent.analyze("Compare identity verification and them.")
    assert result.clarification_needed
    assert result.ambiguity_type == "incomplete"

def test_comparative_query_single_pronoun_target():
    """Test that comparative queries with single pronoun target trigger clarification."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare it with that")
    assert result.clarification_needed
    # Pronoun-only queries are caught by incomplete check since they have no entities
    assert result.ambiguity_type == "incomplete"

def test_comparative_query_with_advantages():
    """Test comparative query with 'advantages of' modifier."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare the advantages of semantic versioning and continuous delivery")
    assert not result.clarification_needed

def test_comparative_query_all_uppercase():
    """Test comparative query with all uppercase targets."""
    agent = ClarificationAgent()
    result = agent.analyze("COMPARE RELEASE TRAINS VERSUS IDENTITY VERIFICATION")
    assert not result.clarification_needed

def test_comparative_query_title_case():
    """Test comparative query with title case targets."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare Release Trains Versus Identity Verification")
    assert not result.clarification_needed

def test_comparative_query_with_benefits():
    """Test comparative query with 'benefits of' modifier."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare the benefits of microservices and monolithic architecture")
    assert not result.clarification_needed

def test_comparative_query_with_disadvantages():
    """Test comparative query with 'disadvantages of' modifier."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare the disadvantages of git and svn")
    assert not result.clarification_needed

def test_comparative_query_with_articles():
    """Test comparative query with articles before targets."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare the git versus the svn")
    assert not result.clarification_needed

def test_comparative_query_with_the_advantages():
    """Test comparative query with 'the advantages of' modifier."""
    agent = ClarificationAgent()
    result = agent.analyze("Compare the advantages of the release trains and the identity verification")
    assert not result.clarification_needed
