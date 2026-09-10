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
    llm = MockLLM("A feature branch is a separate line of development.")
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


def test_context_supported_procedural_query_is_grounded():
    llm = MockLLM("1. Run git checkout -b feature/name.")
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
    assert result.query_type == "procedural"
    assert "numbered steps" in llm.calls[0]["messages"][0]["content"]


def test_context_supported_comparative_query_is_grounded():
    llm = MockLLM("REST is request-response; SOAP uses a formal protocol.")
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
    llm = MockLLM("The documented dose is 650 mg.")
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
