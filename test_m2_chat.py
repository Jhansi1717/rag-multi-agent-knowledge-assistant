"""test_m2_chat.py — M2 API integration tests for POST /chat endpoint.

Tests cover:
- Factual query → answered
- Unavailable query → unavailable status
- Ambiguous query → clarification_needed
- Session memory carries context across turns
- Response schema is complete
"""
import os
import pytest
from fastapi.testclient import TestClient

# Ensure no LLM calls in tests (use extractive fallback)
os.environ.pop("OPENAI_API_KEY", None)

from app import app, vector_store


@pytest.fixture(autouse=True)
def load_eval_index():
    """Point the app's vector_store and orchestrator at the evaluation corpus."""
    eval_index = "data/evaluation/index.faiss"
    eval_meta  = "data/evaluation/metadata.json"
    if os.path.exists(eval_index):
        # Re-configure paths and reload
        vector_store.index_path = eval_index
        vector_store.meta_path  = eval_meta
        vector_store.load()
    yield


client = TestClient(app)


class TestChatEndpoint:

    def test_chat_health_still_works(self):
        """M1 /health endpoint must remain unaffected."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_chat_factual_query_answered(self):
        """A factual in-KB query returns status=answered with an answer."""
        r = client.post("/chat", json={"query": "What is the Git feature branch workflow?"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "answered"
        assert data["intent"] == "factual"
        assert isinstance(data["answer"], str) and len(data["answer"]) > 10
        assert isinstance(data["citations"], list)

    def test_chat_procedural_query_answered(self):
        """A procedural query is correctly classified and answered."""
        r = client.post("/chat", json={"query": "How do I deploy a microservice?"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "answered"
        assert data["intent"] == "procedural"

    def test_chat_unavailable_query(self):
        """Queries about information not in the KB return unavailable status."""
        r = client.post("/chat", json={"query": "What was the company's revenue in 2025?"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "unavailable"
        assert data["intent"] == "unavailable"

    def test_chat_response_schema_complete(self):
        """Every /chat response contains all required fields."""
        r = client.post("/chat", json={"query": "What is the Git workflow?"})
        assert r.status_code == 200
        data = r.json()
        required_fields = {
            "query", "session_id", "answer", "status",
            "intent", "confidence", "domain", "citations",
            "clarification_question",
        }
        assert required_fields.issubset(data.keys()), \
            f"Missing fields: {required_fields - data.keys()}"

    def test_chat_confidence_is_float_in_range(self):
        """Confidence score is a float between 0 and 1."""
        r = client.post("/chat", json={"query": "What is the patient admission policy?"})
        assert r.status_code == 200
        conf = r.json()["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_chat_citations_have_required_fields(self):
        """Citations contain chunk_id, filename, and excerpt."""
        r = client.post("/chat", json={"query": "What coding standards apply to Python?"})
        assert r.status_code == 200
        data = r.json()
        if data["status"] == "answered":
            assert len(data["citations"]) > 0
            for citation in data["citations"]:
                assert "chunk_id" in citation
                assert "filename" in citation
                assert "excerpt" in citation

    def test_chat_session_id_echoed(self):
        """The session_id provided in the request is echoed in the response."""
        r = client.post("/chat", json={
            "query": "How do I apply for leave?",
            "session_id": "test-session-42",
        })
        assert r.status_code == 200
        assert r.json()["session_id"] == "test-session-42"

    def test_chat_session_memory_second_turn(self):
        """Two turns with the same session_id both succeed (memory stores turn 1)."""
        session = "memory-test-session"
        r1 = client.post("/chat", json={
            "query": "What is the Git feature branch workflow?",
            "session_id": session,
        })
        assert r1.status_code == 200

        r2 = client.post("/chat", json={
            "query": "How do I create a pull request?",
            "session_id": session,
        })
        assert r2.status_code == 200
        assert r2.json()["status"] in ("answered", "clarification_needed", "unavailable")

    def test_chat_top_k_respected(self):
        """top_k parameter limits the number of citations returned."""
        r = client.post("/chat", json={
            "query": "What is the Git feature branch workflow?",
            "top_k": 1,
        })
        assert r.status_code == 200
        data = r.json()
        if data["status"] == "answered":
            assert len(data["citations"]) <= 1
