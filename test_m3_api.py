"""test_m3_api.py — M3 API integration tests for /chat and /sessions endpoints.

Tests cover:
- existing M2 request
- new session query
- follow-up query
- clarification flow
- history endpoint
- delete endpoint
- backward compatibility
- error cases (invalid session, pending clarification, empty query)
"""
import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from agents.models import Citation, M3ResponseResult, ResponseResult, RetrievalHit, RetrievalResult
from agents.transparency import TransparencyBuilder

os.environ.pop("OPENAI_API_KEY", None)

from app import _response_payload, app, vector_store, rate_limiter

@pytest.fixture(autouse=True)
def load_eval_index():
    original_index = vector_store.index_path
    original_metadata = vector_store.meta_path
    eval_index = "data/evaluation/index.faiss"
    eval_meta  = "data/evaluation/metadata.json"
    if os.path.exists(eval_index):
        vector_store.index_path = eval_index
        vector_store.meta_path  = eval_meta
        vector_store.load()
    yield
    vector_store.index_path = original_index
    vector_store.meta_path = original_metadata
    if os.path.exists(original_index) and os.path.exists(original_metadata):
        vector_store.load()

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    rate_limiter.records.clear()

client = TestClient(app)

def test_chat_empty_session_id_generates_uuid():
    """Omitting session_id generates a new one."""
    r = client.post("/chat", json={"query": "What is Docker?"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["session_id"] != "default"
    assert len(data["session_id"]) > 0

def test_chat_explicit_session_id_echoed():
    """Providing a session_id uses it."""
    r = client.post("/chat", json={"query": "What is Docker?", "session_id": "test-session-m3"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "test-session-m3"

def test_history_endpoint():
    """GET /sessions/{session_id}/history returns turns."""
    session_id = "history-test-session"
    client.post("/chat", json={"query": "What is a sprint?", "session_id": session_id})
    r = client.get(f"/sessions/{session_id}/history")
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == session_id
    assert len(data["history"]) == 1
    assert data["history"][0]["query"] == "What is a sprint?"

def test_delete_endpoint():
    """DELETE /sessions/{session_id} clears the session."""
    session_id = "delete-test-session"
    client.post("/chat", json={"query": "What is Docker?", "session_id": session_id})
    r = client.delete(f"/sessions/{session_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # Verify history is gone
    r2 = client.get(f"/sessions/{session_id}/history")
    assert r2.status_code == 404

def test_follow_up_query_uses_memory():
    """Follow-up queries resolve pronouns using memory context."""
    session_id = "follow-up-session"
    client.post("/chat", json={"query": "What is a sprint?", "session_id": session_id})
    r = client.post("/chat", json={"query": "How long does it last?", "session_id": session_id})
    print("\n[DEBUG FOLLOW-UP]", r.json())
    assert r.status_code == 200
    # M3 metadata should indicate context was used
    m3 = r.json().get("m3")
    assert m3 is not None
    assert m3["memory"]["used_context"] is True

def test_clarification_flow():
    """Ambiguous query -> Needs clarification -> Provide answer -> Resolved."""
    session_id = "clarification-session"

    # 1. Ambiguous query
    r1 = client.post("/chat", json={"query": "Compare the two", "session_id": session_id})
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["clarification_needed"] is True
    assert data1["status"] == "clarification_needed"
    m3 = data1["m3"]
    assert m3["clarification"]["needed"] is True
    clarification_id = m3["clarification"]["clarification_id"]

    # 2. Provide clarification answer
    r2 = client.post("/chat", json={
        "query": "Docker and Kubernetes",
        "session_id": session_id,
        "clarification_id": clarification_id
    })
    print("\n[DEBUG CLARIFICATION]", r2.json())
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["status"] != "clarification_needed"
    assert data2["m3"]["clarification"]["resolved"] is True
    assert data2["m3"]["clarification"]["clarification_answer"] == "Docker and Kubernetes"

def test_invalid_clarification_id():
    """Providing wrong clarification_id returns error."""
    session_id = "invalid-clar-session"
    r1 = client.post("/chat", json={"query": "Compare the two", "session_id": session_id})
    assert r1.status_code == 200

    r2 = client.post("/chat", json={
        "query": "Docker and Kubernetes",
        "session_id": session_id,
        "clarification_id": "wrong-id"
    })
    assert r2.status_code == 500
    assert "invalid clarification_id" in r2.json()["error"]["message"].lower()

def test_invalid_session_history():
    """Missing session returns 404 for history."""
    r = client.get("/sessions/nonexistent/history")
    assert r.status_code == 404

def test_invalid_session_delete():
    """Missing session returns 404 for delete."""
    r = client.delete("/sessions/nonexistent")
    assert r.status_code == 404

def test_empty_query():
    """Empty query should fail validation."""
    r = client.post("/chat", json={"query": "   ", "session_id": "test"})
    assert r.status_code == 400
    assert "query must not be empty" in r.json()["error"]["message"]

def test_no_result_response():
    """Querying for something completely unknown."""
    r = client.post("/chat", json={"query": "What is quantum flux capacitor overdrive?", "session_id": "no-result"})
    assert r.status_code == 200
    data = r.json()
    assert data["no_information_found"] is True


def test_serialized_citations_match_transparency_evidence():
    hit = RetrievalHit(
        rank=1,
        relevance_score=0.9,
        distance_score=(1 / 0.9) - 1,
        text="The indexed guide supports this statement.",
        document_id="doc-1",
        filename="guide.txt",
        chunk_id="chunk-1",
    )
    retrieval = RetrievalResult(
        query="What does the guide say?",
        query_type="factual",
        top_k=1,
        results=[hit],
        retrieval_confidence=0.9,
        sufficient_evidence=True,
    )
    response = ResponseResult(
        answer="The indexed guide supports this statement.",
        citations=[
            Citation(
                chunk_id="chunk-1",
                filename="guide.txt",
                document_id="doc-1",
                excerpt="The indexed guide supports this statement.",
            ),
            Citation(
                chunk_id="missing-chunk",
                filename="invented.txt",
                document_id="invented-doc",
                excerpt="Invented citation text.",
            ),
        ],
        confidence=0.9,
        confidence_level="HIGH",
        grounded=True,
        retrieval_hits=[hit],
        retrieval=retrieval,
        status="answered",
    )
    response.m3 = M3ResponseResult(
        transparency=TransparencyBuilder().build(retrieval=retrieval, response=response)
    )

    payload = _response_payload(retrieval.query, "test-session", response)
    transparency = payload["m3"]["transparency"]

    assert payload["citations"] == transparency["citations"]
    assert len(payload["citations"]) == 1
    citation = payload["citations"][0]
    evidence = next(item for item in transparency["source_evidence"] if item["chunk_id"] == citation["chunk_id"])
    assert citation["filename"] == evidence["document_name"]
    assert citation["document_id"] == evidence["document_id"]
    assert transparency["grounded"] is False

def test_upload_returns_original_filename():
    """Upload endpoint should return the original filename, not UUID."""
    from ingestion.pipeline import index_file

    # Create a temporary test file (TXT instead of PDF to avoid PyMuPDF dependency)
    original_filename = "test_document.txt"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tmp:
        tmp.write(b"Test content for document upload")
        tmp_path = tmp.name

    try:
        # Test that original_filename is preserved through the pipeline
        result = index_file(tmp_path, vector_store, original_filename=original_filename, save=False)

        # The result should contain the original filename
        assert result["filename"] == original_filename
        assert result["filename"] != os.path.basename(tmp_path)

    finally:
        # Cleanup
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except PermissionError:
            pass  # File may still be locked on Windows

def test_clarification_process_pluralization():
    """Clarification for 'the process' should use proper plural 'processes'."""
    from agents.clarification import ClarificationAgent

    agent = ClarificationAgent()
    result = agent.analyze("Explain the process")

    # Should trigger clarification for ambiguous terminology
    assert result.clarification_needed is True
    assert result.ambiguity_type == "ambiguous_terminology"

    # Should use proper pluralization "processes" not "processs"
    assert "processes" in result.clarification_question.lower()
    assert "processs" not in result.clarification_question.lower()
