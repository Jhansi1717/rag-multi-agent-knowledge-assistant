import os
import pytest
from fastapi.testclient import TestClient
from app import app
from agents.models import RetrievalHit, RetrievalResult, ResponseResult
from agents.orchestrator import Orchestrator

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health check passed.")

def test_upload_supported_file():
    test_file_path = "test_upload.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test document for ingestion pipeline validation. " * 50)
        
    with open(test_file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test_upload.txt", f, "text/plain")})
        
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "test_upload.txt"
    assert data["file_type"] == ".txt"
    assert data["chunk_count"] > 0
    assert "metadata" in data
    print("Supported file upload passed. Metadata:", data)
    
    os.remove(test_file_path)

def test_upload_unsupported_file():
    test_file_path = "test_upload.md"
    with open(test_file_path, "w") as f:
        f.write("# Hello")
        
    with open(test_file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test_upload.md", f, "text/markdown")})
        
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]
    print("Unsupported file validation passed.")
    
    os.remove(test_file_path)


def _fake_response(query: str) -> ResponseResult:
    query_type = "factual"
    if query.startswith("How do"):
        query_type = "procedural"
    elif "versus" in query:
        query_type = "comparative"
    elif query == "unclear":
        query_type = "ambiguous"

    if query_type == "ambiguous":
        return ResponseResult(
            answer="Could you provide more detail?",
            confidence=0.8,
            classification_confidence=0.8,
            confidence_level="HIGH",
            query_type=query_type,
            status="clarification_needed",
            request_id="test-request",
        )

    hits = [] if query == "unknown fact" else [
        RetrievalHit(
            rank=1,
            relevance_score=0.9,
            distance_score=0.1,
            text="Supported context.",
            document_id="doc-1",
            filename="guide.txt",
            chunk_id="chunk-1",
        )
    ]
    retrieval = RetrievalResult(
        query=query,
        query_type=query_type,
        top_k=3,
        results=hits,
        retrieval_confidence=0.9 if hits else 0.0,
        sufficient_evidence=bool(hits),
        no_relevant_information=not hits,
    )
    return ResponseResult(
        answer="The requested information is not available in the provided documents."
        if not hits
        else "Supported answer.",
        confidence=retrieval.retrieval_confidence,
        confidence_level="HIGH" if hits else "LOW",
        classification_confidence=0.9,
        grounded=bool(hits),
        no_information_found=not hits,
        query_type=query_type,
        retrieval_hits=hits,
        retrieval=retrieval,
        status="unavailable" if not hits else "answered",
        request_id="test-request",
    )


@pytest.fixture
def mocked_orchestrator(monkeypatch):
    monkeypatch.setattr(
        "app.orchestrator.handle",
        lambda query, session_id, top_k: _fake_response(query),
    )


@pytest.mark.parametrize(
    ("query", "query_type", "status"),
    [
        ("What is a sprint?", "factual", "answered"),
        ("How do I deploy a service?", "procedural", "answered"),
        ("Microservices versus monoliths", "comparative", "answered"),
        ("unclear", "ambiguous", "clarification_needed"),
        ("unknown fact", "factual", "unavailable"),
    ],
)
def test_retrieve_m2_pipeline_responses(
    mocked_orchestrator, query, query_type, status
):
    response = client.post(
        "/retrieve",
        json={"query": query, "top_k": 3, "session_id": "test-session"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "test-request"
    assert data["query"] == query
    assert data["query_type"] == query_type
    assert data["status"] == status
    assert data["classification_confidence"] > 0
    assert "retrieval" in data
    assert "clarification_needed" in data


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "valid", "top_k": 0}, {"query": "valid", "top_k": 101}],
)
def test_retrieve_rejects_invalid_requests(mocked_orchestrator, payload):
    response = client.post("/retrieve", json=payload)
    assert response.status_code == 400
    
if __name__ == "__main__":
    test_health()
    test_upload_supported_file()
    test_upload_unsupported_file()
    print("All tests passed successfully!")
