from types import SimpleNamespace

from agents.models import RetrievalHit, ResponseResult
from agents.response_generation import ResponseGenerationAgent


class FakeLLM:
    def __init__(self, answer: str):
        self.answer = answer
        self.last_request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.last_request = request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer),
                )
            ]
        )


def hit(filename="guide.txt", score=0.8):
    return RetrievalHit(
        rank=1,
        relevance_score=score,
        distance_score=(1 / score) - 1,
        text="A feature branch is created with git checkout -b feature/name.",
        document_id="doc-1",
        filename=filename,
        chunk_id="chunk-1",
        metadata={"domain": "Software Engineering", "page": 1},
    )


def test_factual_generation_is_grounded_and_cited():
    llm = FakeLLM("Create a feature branch with the documented command. [source: guide.txt]")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is a feature branch?",
        [hit()],
        intent="factual",
        confidence=0.8,
    )
    assert isinstance(result, ResponseResult)
    assert result.grounded is True
    assert result.no_information_found is False
    assert result.confidence_level == "HIGH"
    assert result.citations[0].filename == "guide.txt"
    assert result.citations[0].chunk_id == "chunk-1"


def test_procedural_prompt_requests_supported_numbered_steps():
    llm = FakeLLM("1. Run the documented command.")
    ResponseGenerationAgent(llm_client=llm).generate(
        "How do I create a feature branch?",
        [hit()],
        intent="procedural",
        confidence=0.6,
    )
    system_prompt = llm.last_request["messages"][0]["content"]
    assert "numbered steps" in system_prompt
    assert "ONLY the supplied context" in system_prompt
    assert "outside knowledge" in system_prompt


def test_comparative_prompt_is_context_only():
    llm = FakeLLM("The context supports a comparison.")
    ResponseGenerationAgent(llm_client=llm).generate(
        "Compare two workflows.",
        [hit("one.txt"), hit("two.txt")],
        intent="comparative",
        confidence=0.8,
    )
    system_prompt = llm.last_request["messages"][0]["content"]
    user_prompt = llm.last_request["messages"][1]["content"]
    assert "structured comparison" in system_prompt
    assert "one.txt" in user_prompt
    assert "two.txt" in user_prompt
    assert "invent citations" in system_prompt


def test_no_evidence_returns_controlled_response_without_llm_call():
    llm = FakeLLM("should not be used")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is unavailable?",
        [],
        intent="factual",
        confidence=0.0,
    )
    assert result.no_information_found is True
    assert result.grounded is False
    assert result.citations == []
    assert llm.last_request is None


def test_low_confidence_returns_controlled_response():
    llm = FakeLLM("should not be used")
    result = ResponseGenerationAgent(llm_client=llm).generate(
        "What is weakly supported?",
        [hit(score=0.2)],
        intent="factual",
        confidence=0.2,
    )
    assert result.confidence_level == "LOW"
    assert result.no_information_found is True
    assert result.grounded is False
    assert llm.last_request is None
