from types import SimpleNamespace

from agents.models import RetrievalHit
from agents.response_generation import ResponseGenerationAgent


class MockLLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **request):
        self.calls.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer)
                )
            ]
        )


class FailingLLM:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **request):
        raise RuntimeError("mock provider failure")


def retrieval_hit(
    text,
    filename,
    chunk_id,
    document_id,
    domain,
    relevance_score=0.8,
):
    return RetrievalHit(
        rank=1,
        relevance_score=relevance_score,
        distance_score=(1 / relevance_score) - 1,
        text=text,
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        metadata={"domain": domain, "page": 1},
    )


def test_context_supported_factual_query_is_grounded():
    llm = MockLLM(
        "A feature branch is a separate line of development. "
        "[source: git_workflow.txt]"
    )
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is a feature branch?",
        [
            retrieval_hit(
                "A feature branch is a separate line of development.",
                "git_workflow.txt",
                "git-0",
                "git-doc",
                "Software Engineering",
            )
        ],
        intent="factual",
        confidence=0.8,
    )

    assert result.grounded is True
    assert result.no_information_found is False
    assert len(llm.calls) == 1
    assert "metadata=" in llm.calls[0]["messages"][0]["content"]


def test_context_supported_procedural_query_is_grounded():
    llm = MockLLM("1. Run git checkout -b feature/name. [source: git_workflow.txt]")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "How do I create a feature branch?",
        [
            retrieval_hit(
                "Run git checkout -b feature/name.",
                "git_workflow.txt",
                "git-1",
                "git-doc",
                "Software Engineering",
            )
        ],
        intent="procedural",
        confidence=0.8,
    )

    assert result.grounded is True
    assert result.no_information_found is False
    assert result.query_type == "procedural"
    assert "numbered steps" in llm.calls[0]["messages"][0]["content"]


def test_context_supported_comparative_query_is_grounded():
    llm = MockLLM(
        "REST is request-response; SOAP uses a formal protocol. "
        "[source: api_guide.txt]"
    )
    hits = [
        retrieval_hit(
            "REST uses request-response APIs.",
            "api_guide.txt",
            "rest-0",
            "api-doc",
            "Software Engineering",
        ),
        retrieval_hit(
            "SOAP uses a formal protocol.",
            "api_guide.txt",
            "soap-0",
            "api-doc",
            "Software Engineering",
        ),
    ]
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "Compare REST and SOAP.",
        hits,
        intent="comparative",
        confidence=0.8,
    )

    assert result.grounded is True
    assert result.query_type == "comparative"
    assert result.no_information_found is False
    assert "structured comparison" in llm.calls[0]["messages"][0]["content"]


def test_insufficient_evidence_returns_no_information():
    llm = MockLLM("This response must not be used.")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is the hospital's annual revenue?",
        [],
        intent="factual",
        confidence=0.0,
    )

    assert result.no_information_found is True
    assert result.grounded is False
    assert result.citations == []
    assert llm.calls == []


def test_unrelated_context_does_not_allow_unsupported_answer():
    llm = MockLLM(
        ResponseGenerationAgent.NO_INFORMATION_MESSAGE
    )
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is the hospital's annual revenue?",
        [
            retrieval_hit(
                "Git branches support isolated software changes.",
                "git_workflow.txt",
                "git-2",
                "git-doc",
                "Software Engineering",
            )
        ],
        intent="factual",
        confidence=0.8,
    )

    assert result.no_information_found is True
    assert result.grounded is False
    assert result.answer == ResponseGenerationAgent.NO_INFORMATION_MESSAGE


def test_citations_use_actual_document_and_chunk_metadata():
    llm = MockLLM(
        "The documented dose is 650 mg. "
        "[source: medication_dosage_reference.csv]"
    )
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is the standard oral dose?",
        [
            retrieval_hit(
                "The standard oral dose is 650 mg.",
                "medication_dosage_reference.csv",
                "dose-7",
                "medication-doc",
                "Hospital Administration",
            )
        ],
        intent="factual",
        confidence=0.8,
    )

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.filename == "medication_dosage_reference.csv"
    assert citation.chunk_id == "dose-7"
    assert citation.document_id == "medication-doc"
    assert citation.excerpt == "The standard oral dose is 650 mg."


def test_provider_configuration_from_environment(monkeypatch):
    # Test OpenAI default
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
    monkeypatch.setenv("OPENAI_MODEL", "test-model-openai")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    agent_openai = ResponseGenerationAgent()
    assert agent_openai._provider == "openai"
    assert agent_openai._api_key == "test-key-openai"
    assert agent_openai._llm_model == "test-model-openai"

    # Test Gemini
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-gemini")
    monkeypatch.setenv("GEMINI_MODEL", "test-model-gemini")
    agent_gemini = ResponseGenerationAgent()
    assert agent_gemini._provider == "gemini"
    assert agent_gemini._api_key == "test-key-gemini"
    assert agent_gemini._llm_model == "test-model-gemini"

    # The client uses base_url for Gemini
    if agent_gemini._client is not None:
        assert str(agent_gemini._client.base_url).startswith("https://generativelanguage.googleapis.com/")


def test_unsupported_provider_raises_value_error():
    import pytest
    with pytest.raises(ValueError, match="Unsupported LLM provider: unknown"):
        ResponseGenerationAgent(provider="unknown")

def test_llm_failure_returns_controlled_error_response():
    result = ResponseGenerationAgent(llm_client=FailingLLM()).generate(
        "What is supported by the guide?",
        [
            retrieval_hit(
                "The guide supports this documented fact.",
                "guide.txt",
                "guide-0",
                "guide-doc",
                "Software Engineering",
            )
        ],
        intent="factual",
        confidence=0.8,
    )

    assert result.no_information_found is True
    assert result.grounded is False
    assert result.answer == ResponseGenerationAgent.NO_INFORMATION_MESSAGE
    assert result.citations == []


def test_grounding_prompt_and_response_contract():
    llm = MockLLM(
        "The context supports the documented answer. [source: guide.txt]"
    )
    def test_missing_or_unknown_source_reference_is_not_grounded():
        hit = retrieval_hit(
            "Riverside General Hospital patient admission requires photo identification.",
            "patient_admission_policy.pdf",
            "admission-1",
            "hospital-doc",
            "Hospital Administration",
        )
        for answer in (
            "Annual revenue was $20 million.",
            "Annual revenue was $20 million. [source: nonexistent.pdf]",
        ):
            result = ResponseGenerationAgent(
                llm_client=MockLLM(answer)
            ).generate(
                "What was Riverside hospital revenue?",
                [hit],
                intent="factual",
                confidence=0.8,
            )

            assert result.grounded is False
            assert result.no_information_found is True
            assert result.citations == []
            assert result.answer == ResponseGenerationAgent.NO_INFORMATION_MESSAGE
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is documented?",
        [
            retrieval_hit(
                "The documented answer is in the guide.",
                "guide.txt",
                "guide-0",
                "guide-doc",
                "Software Engineering",
            )
        ],
        intent="factual",
        confidence=0.8,
    )

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "ONLY the supplied context" in prompt
    assert "outside knowledge" in prompt
    assert "invent" in prompt
    assert "insufficient" in prompt
    assert result.answer
    assert result.query_type == "factual"
    assert result.confidence == 0.8
    assert result.confidence_level == "HIGH"
    assert result.grounded is True
    assert result.no_information_found is False
    assert result.citations[0].filename == "guide.txt"


def test_empty_retrieved_chunk_is_insufficient_evidence():
    llm = MockLLM("must not be used")
    empty_hit = retrieval_hit("", "empty.txt", "empty-0", "empty-doc", "Software Engineering")

    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is documented?",
        [empty_hit],
        intent="factual",
        confidence=0.8,
    )

    assert result.no_information_found is True
    assert result.grounded is False
    assert result.citations == []
    assert llm.calls == []


def test_low_confidence_does_not_present_context_as_supported():
    llm = MockLLM("must not be used")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is documented?",
        [
            retrieval_hit(
                "A weakly related passage.",
                "weak.txt",
                "weak-0",
                "weak-doc",
                "Software Engineering",
                relevance_score=0.2,
            )
        ],
        intent="factual",
        confidence=0.2,
    )

    assert result.no_information_found is True
    assert result.grounded is False
    assert result.citations == []
    assert llm.calls == []


def test_retrieval_unavailable_flags_prevent_provider_call():
    hit = retrieval_hit(
        "A documented fact from the guide.",
        "guide.txt",
        "guide-0",
        "guide-doc",
        "Software Engineering",
    )
    for evidence_flags in (
        {"sufficient_evidence": False, "no_relevant_information": False},
        {"sufficient_evidence": True, "no_relevant_information": True},
    ):
        llm = MockLLM("A documented fact. [source: guide.txt]")
        result = ResponseGenerationAgent(llm_client=llm).generate(
            "What is documented?",
            [hit],
            confidence=0.8,
            **evidence_flags,
        )

        assert result.no_information_found is True
        assert result.grounded is False
        assert result.citations == []
        assert llm.calls == []


def test_every_citation_must_map_to_supplied_retrieved_evidence():
    llm = MockLLM(
        "The guide documents a feature branch. [source: guide.txt] "
        "Revenue was reported. [source: nonexistent.pdf]"
    )
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "Summarize the guide.",
        [
            retrieval_hit(
                "A feature branch is documented.",
                "guide.txt",
                "guide-0",
                "guide-doc",
                "Software Engineering",
            )
        ],
        confidence=0.8,
    )

    assert result.grounded is False
    assert result.no_information_found is True
    assert result.citations == []
    assert result.answer == ResponseGenerationAgent.NO_INFORMATION_MESSAGE


def test_citation_cannot_reference_evidence_omitted_from_provider_context():
    llm = MockLLM("A fact from the fourth passage. [source: fourth.txt]")
    hits = [
        retrieval_hit(
            f"Passage {index}.",
            f"source{index}.txt",
            f"chunk-{index}",
            f"doc-{index}",
            "Software Engineering",
        )
        for index in range(1, 4)
    ]
    hits.append(
        retrieval_hit(
            "A fact from the fourth passage.",
            "fourth.txt",
            "chunk-4",
            "doc-4",
            "Software Engineering",
        )
    )

    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is the fact?",
        hits,
        confidence=0.8,
    )

    assert "fourth.txt" not in llm.calls[0]["messages"][1]["content"]
    assert result.grounded is False
    assert result.no_information_found is True
    assert result.citations == []


def test_non_finite_confidence_cannot_become_high_confidence():
    for confidence in (float("nan"), float("inf")):
        llm = MockLLM("A documented fact. [source: guide.txt]")
        result = ResponseGenerationAgent(llm_client=llm).generate(
            "What is documented?",
            [
                retrieval_hit(
                    "A documented fact.",
                    "guide.txt",
                    "guide-0",
                    "guide-doc",
                    "Software Engineering",
                )
            ],
            confidence=confidence,
        )

        assert result.confidence == 0.0
        assert result.confidence_level == "LOW"
        assert result.no_information_found is True
        assert llm.calls == []


def test_provider_error_log_omits_exception_details(caplog):
    ResponseGenerationAgent(llm_client=FailingLLM()).generate(
        "What is supported by the guide?",
        [
            retrieval_hit(
                "The guide supports this documented fact.",
                "guide.txt",
                "guide-0",
                "guide-doc",
                "Software Engineering",
            )
        ],
        confidence=0.8,
    )

    assert "mock provider failure" not in caplog.text


# ---------------------------------------------------------------------------
# DEF-001 regression tests — decoupled confidence gate
# ---------------------------------------------------------------------------
import pytest


def _borderline_hit():
    """RetrievalHit with relevance_score=0.4486 — borderline valid per upstream."""
    return retrieval_hit(
        "Teams operate in two-week sprints.",
        "microservices_architecture.pdf",
        "sprint-0",
        "ms-doc",
        "Software Engineering",
        relevance_score=0.4486,
    )


def test_def001_borderline_confidence_with_sufficient_evidence_true_reaches_llm():
    """Regression: confidence=0.4486 with sufficient_evidence=True must call the LLM.

    DEF-001 root cause: the old gate fired ``confidence < 0.45`` regardless of
    whether the upstream RetrievalAgent had already declared evidence sufficient.
    This silently dropped valid borderline evidence before any LLM call.
    """
    llm = MockLLM(
        "Teams operate in two-week sprints. [source: microservices_architecture.pdf]"
    )
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is a typical agile sprint length?",
        [_borderline_hit()],
        intent="factual",
        query_type="factual",
        confidence=0.4486,
        sufficient_evidence=True,
        no_relevant_information=False,
    )

    assert len(llm.calls) == 1, (
        "LLM was not called for borderline confidence with sufficient_evidence=True. "
        "DEF-001 may have regressed."
    )
    assert result.grounded is True
    assert result.no_information_found is False
    assert result.confidence == pytest.approx(0.4486, abs=1e-6)
    assert result.confidence_level == "LOW"
    assert result.citations is not None and len(result.citations) == 1
    assert result.citations[0].filename == "microservices_architecture.pdf"


def test_def001_sufficient_evidence_false_rejects_even_with_high_confidence():
    """When the upstream agent says sufficient_evidence=False, the LLM must not be called,
    even when confidence is high.  The upstream flag is authoritative.
    """
    llm = MockLLM("A documented fact. [source: microservices_architecture.pdf]")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is a sprint?",
        [_borderline_hit()],
        intent="factual",
        query_type="factual",
        confidence=0.95,
        sufficient_evidence=False,
        no_relevant_information=False,
    )

    assert llm.calls == [], "LLM must not be called when sufficient_evidence=False."
    assert result.no_information_found is True
    assert result.grounded is False
    assert result.citations == []


def test_def001_empty_hits_rejected_even_with_sufficient_evidence_true():
    """Empty evidence list is always rejected regardless of flag state."""
    llm = MockLLM("A documented fact. [source: microservices_architecture.pdf]")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is a sprint?",
        [],
        intent="factual",
        query_type="factual",
        confidence=0.95,
        sufficient_evidence=True,
    )

    assert llm.calls == [], "LLM must not be called for empty hits."
    assert result.no_information_found is True
    assert result.grounded is False


def test_def001_provider_failure_with_borderline_confidence_fails_safely():
    """A provider failure with borderline confidence must return no_information safely,
    not raise an unhandled exception.
    """
    result = ResponseGenerationAgent(llm_client=FailingLLM()).generate(
        "What is a typical agile sprint length?",
        [_borderline_hit()],
        intent="factual",
        query_type="factual",
        confidence=0.4486,
        sufficient_evidence=True,
        no_relevant_information=False,
    )

    assert result.no_information_found is True
    assert result.grounded is False
    assert result.citations == []
    assert result.answer == ResponseGenerationAgent.NO_INFORMATION_MESSAGE


def test_def001_no_sufficient_evidence_flag_confidence_floor_still_applies():
    """When sufficient_evidence is not passed, the original confidence floor
    still rejects low-confidence evidence (backward-compatible behaviour for
    direct callers that do not supply the upstream flag).
    """
    llm = MockLLM("must not be used")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is documented?",
        [_borderline_hit()],
        intent="factual",
        query_type="factual",
        confidence=0.2,
        # sufficient_evidence intentionally omitted
    )

    assert llm.calls == [], (
        "LLM must not be called when confidence is low and sufficient_evidence is unset."
    )
    assert result.no_information_found is True
    assert result.grounded is False
