import os
import re
import pytest

os.environ.pop("OPENAI_API_KEY", None)

from fastapi.testclient import TestClient
from app import app, vector_store
from typing import Dict, Any

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    original_search = vector_store.search

    def fake_search(query, top_k=3):
        q = query.lower()
        if "retrieval agent" in q and "about the agent" not in q:
            text = "The core capabilities of the Retrieval Agent include vector search and ranking."
            doc = "architecture/rag-query-flow.md"
        elif "query understanding agent" in q or "primary role" in q or "query understanding agent primary role" in q:
            text = "The primary role of the Query Understanding Agent is intent classification."
            doc = "architecture/rag-query-flow.md"
        elif "response generation agent" in q:
            text = "The Response Generation Agent generates grounded answers."
            doc = "architecture/rag-query-flow.md"
        elif "database options" in q:
            text = "Database options supported include PostgreSQL and SQLite."
            doc = "data/technical/api_docs.txt"
        elif "authentication" in q or "it work" in q:
            text = "Authentication works via JWT tokens."
            doc = "data/technical/api_docs.txt"
        else:
            return original_search(query, top_k)

        return [{"chunk_id": "mock_chunk_1", "document_id": "mock_doc_1",
                 "document_name": doc, "filename": doc, "text": text,
                 "distance_score": 0.0, "similarity_score": 0.0}]

    monkeypatch.setattr(vector_store, "search", fake_search)

    from agents.response_generation import ResponseGenerationAgent

    def fake_call_llm(self, sp, up):
        source = re.search(r"source_filename='([^']+)'", up)
        if not source:
            return self.NO_INFORMATION_MESSAGE
        return f"{up}\n[source: {source.group(1)}]"

    monkeypatch.setattr(ResponseGenerationAgent, "_call_llm", fake_call_llm)


def chat(query, session_id=None, clarification_id=None):
    payload = {"query": query, "top_k": 3}
    if session_id:
        payload["session_id"] = session_id
    if clarification_id:
        payload["clarification_id"] = clarification_id
    resp = client.post("/chat", json=payload)
    resp.raise_for_status()
    return resp.json()


def test_scenario_1_normal_query():
    """SCENARIO 1: Normal factual query -> retrieval -> grounded answer."""
    res = chat("What are the core capabilities of the Retrieval Agent?")
    assert res.get("status") == "answered", f"Expected answered, got {res.get('status')!r}"
    assert "Retrieval Agent" in res.get("answer", "")
    assert "m3" in res
    assert res["m3"]["transparency"]["retrieved_count"] > 0
    assert res["m3"]["transparency"]["grounded"] is True


def test_scenario_2_ambiguity():
    """SCENARIO 2: Vague query triggers clarification before retrieval."""
    res = chat("What about the second one?")
    assert res.get("status") == "clarification_needed"
    assert "m3" in res
    assert res["m3"]["clarification"] is not None
    assert res["m3"]["clarification"]["clarification_id"] is not None


def test_scenario_3_clarification_continuation():
    """SCENARIO 3: Ambiguous query -> clarification; resolved query -> full pipeline."""
    res1 = chat("Tell me about the agent.")
    assert res1.get("status") == "clarification_needed", f"Step 1 got {res1.get('status')!r}"
    session_id = res1["session_id"]
    clarification_id = res1["m3"]["clarification"]["clarification_id"]
    res2 = chat("I mean the retrieval agent", session_id=session_id, clarification_id=clarification_id)
    assert res2.get("status") == "answered", f"Step 2 got {res2.get('status')!r}"
    assert res2["m3"]["transparency"]["retrieved_count"] > 0
    assert "Retrieval Agent" in res2["answer"]


def test_scenario_4_follow_up_context():
    """SCENARIO 4: Follow-up query uses session memory context."""
    res1 = chat("What is the Query Understanding Agent?")
    assert res1.get("status") == "answered"
    session_id = res1["session_id"]
    res2 = chat("What is its primary role?", session_id=session_id)
    # The query should either be answered or unavailable (if docs not indexed)
    # The important thing is it should NOT trigger clarification unnecessarily
    assert res2.get("status") in ("answered", "unavailable"), f"got {res2.get('status')!r}"
    # If it's clarification_needed, that's a problem with context resolution
    assert res2.get("status") != "clarification_needed", f"Context resolution failed: got {res2.get('status')!r}"
    mem = res2["m3"]["memory"]
    # Context should have been used or available
    assert mem["used_context"] is True or len(mem["relevant_context"]) > 0


def test_scenario_5_pronoun():
    """SCENARIO 5: Pronoun 'it' refers to prior topic."""
    res1 = chat("Explain authentication.")
    session_id = res1["session_id"]
    res2 = chat("How does it work?", session_id=session_id)
    assert res2.get("status") in ("answered", "clarification_needed"), f"got {res2.get('status')!r}"
    mem = res2["m3"]["memory"]
    assert mem["used_context"] is True or res2.get("status") == "clarification_needed"


def test_scenario_6_topic_switch():
    """SCENARIO 6: New topic should not be contaminated by prior authentication context."""
    res1 = chat("Explain authentication.")
    session_id = res1["session_id"]
    res2 = chat("What database options are supported?", session_id=session_id)
    assert res2.get("status") == "answered", f"got {res2.get('status')!r}"
    answer = res2.get("answer", "").lower()
    assert "database" in answer or "postgresql" in answer or "sqlite" in answer or "options" in answer


def test_scenario_7_missing_context():
    """SCENARIO 7: Deictic without context -> clarification."""
    res = chat("What about the first one?")
    assert res.get("status") == "clarification_needed", f"got {res.get('status')!r}"


def test_scenario_10_transparency():
    """SCENARIO 10: Full transparency block: source_evidence, chunk_id, similarity_score, grounded."""
    res = chat("How does the Response Generation Agent work?")
    assert res.get("status") == "answered"
    t = res["m3"]["transparency"]
    assert "source_evidence" in t
    for src in t["source_evidence"]:
        assert "document_name" in src
        assert "chunk_id" in src
        assert "similarity_score" in src
    assert "confidence_level" in res
    assert "grounded" in t


def test_scenario_11_no_results():
    """SCENARIO 11: Out-of-KB query -> 0 retrieved, not grounded."""
    res = chat("Who won the Super Bowl in 1845?")
    t = res["m3"]["transparency"]
    assert t["retrieved_count"] == 0
    assert t["grounded"] is False


def test_scenario_12_low_confidence():
    """SCENARIO 12: Off-topic query must not be presented as authoritative."""
    res = chat("How do I cook a turkey?")
    if res.get("status") == "answered":
        assert res["confidence_level"] == "LOW", f"got {res['confidence_level']!r}"
    else:
        assert res.get("status") in ("unavailable", "clarification_needed"), f"got {res.get('status')!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
