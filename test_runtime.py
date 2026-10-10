import os
import time
import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

# We will patch os.environ before importing app to test config, but for simplicity
# we can just test the behavior with current env.
os.environ["ALLOWED_ORIGINS"] = "http://testserver,https://example.com"
os.environ["RATE_LIMIT_CALLS"] = "5"
os.environ["RATE_LIMIT_WINDOW"] = "2"

from app import app, rate_limiter, vector_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def configure_runtime_settings():
    """Keep runtime tests independent of module collection order."""
    rate_limiter.calls = 5
    rate_limiter.window = 2
    rate_limiter.records.clear()
    for middleware in app.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            middleware.kwargs["allow_origins"] = [
                "http://testserver",
                "https://example.com",
            ]
    app.middleware_stack = app.build_middleware_stack()
    yield
    rate_limiter.records.clear()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers


def test_readiness_check_success(monkeypatch):
    # Mock vector store and openai key
    monkeypatch.setattr(vector_store, "index", SimpleNamespace(ntotal=1))
    monkeypatch.setattr(vector_store, "metadata", {"0": {"document_id": "doc-1"}})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["dependencies"]["vector_store"] is True
    assert data["dependencies"]["openai_configured"] is True
    assert data["dependencies"]["gemini_configured"] is False
    assert data["dependencies"]["llm_provider"] == "openai"


def test_readiness_check_db_unavailable(monkeypatch):
    # Ensure paths are mocked to not exist
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    monkeypatch.setattr(vector_store, "index", None)

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "HTTP_ERROR", "message": "Database is unavailable"}}


@pytest.mark.parametrize(
    ("vector_count", "metadata"),
    [
        (0, {}),
        (2, {"0": {"document_id": "doc-1"}}),
        (2, {"0": {"document_id": "doc-1"}, "2": {"document_id": "doc-2"}}),
    ],
)
def test_readiness_rejects_empty_or_mismatched_index(monkeypatch, vector_count, metadata):
    monkeypatch.setattr(vector_store, "index", SimpleNamespace(ntotal=vector_count))
    monkeypatch.setattr(vector_store, "metadata", metadata)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "Database is unavailable"


def test_malformed_request():
    response = client.post("/retrieve", json={"query": ""}) # Empty query should trigger 400
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "HTTP_ERROR"
    assert "query must not be empty" in response.json()["error"]["message"]

    response2 = client.post("/retrieve", json={"not_query": "foo"}) # Missing required fields
    assert response2.status_code == 422
    assert response2.json()["error"]["code"] == "VALIDATION_ERROR"


def test_request_id_propagation():
    test_id = "test-uuid-1234"
    response = client.get("/health", headers={"X-Request-ID": test_id})
    assert response.headers["X-Request-ID"] == test_id


def test_cors_allowed_origin():
    response = client.options(
        "/health",
        headers={
            "Origin": "http://testserver",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://testserver"

def test_rate_limiting():
    # We set RATE_LIMIT_CALLS=5
    # Wait out any previous requests to clear the window
    time.sleep(2.1)

    # Let's hit /chat 5 times, 6th should fail
    payload = {"query": "hello"}
    for i in range(5):
        resp = client.post("/chat", json=payload)
        # Should not be 429
        assert resp.status_code != 429

    resp_blocked = client.post("/chat", json=payload)
    assert resp_blocked.status_code == 429
    assert resp_blocked.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

def test_internal_error_hiding(monkeypatch, caplog):
    def mock_handle(*args, **kwargs):
        raise ValueError("Super secret DB error string jdbc:secret:password")

    monkeypatch.setattr("app.orchestrator.handle", mock_handle)

    # Wait out rate limit window first
    time.sleep(2.1)

    response = client.post("/retrieve", json={"query": "hello"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "HTTP_ERROR"
    assert "M3 pipeline failed." in response.json()["error"]["message"]
    assert "Super secret DB error string" not in response.text
    assert "Super secret DB error string" not in caplog.text
