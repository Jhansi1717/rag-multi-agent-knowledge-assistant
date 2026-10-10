"""Tests for M3 data contracts and backward compatibility."""

import dataclasses
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from agents.models import (
    StageTrace,
    EvidenceItem,
    ClarificationState,
    MemoryContext,
    TransparencyReport,
    M3ResponseResult,
    ResponseResult,
)
from app import app, RetrieveResponse

client = TestClient(app)

def test_stage_trace_defaults():
    st = StageTrace(stage_name="retrieval", status="success")
    assert st.duration_ms is None

def test_evidence_item_defaults():
    ei = EvidenceItem(
        citation_id="cit-1",
        document_name="test.txt",
        chunk_id="chk-1",
        chunk_text="text",
        similarity_score=0.9
    )
    assert ei.page is None
    assert ei.section is None

def test_clarification_state_defaults():
    cs = ClarificationState()
    assert cs.needed is False
    assert cs.options == []
    assert cs.missing_information == []

def test_memory_context_defaults():
    mc = MemoryContext()
    assert mc.recent_messages == []
    assert mc.referenced_documents == []
    assert mc.clarification_state is None

def test_transparency_report_defaults():
    tr = TransparencyReport()
    assert tr.stages == []
    assert tr.source_evidence == []
    assert tr.citations == []

def test_m3_response_result_defaults():
    m3 = M3ResponseResult()
    assert m3.clarification is None
    assert m3.memory is None
    assert m3.transparency is None

def test_response_result_backward_compat():
    rr = ResponseResult(answer="hello")
    assert rr.m3 is None
    # ensure it converts to dict cleanly
    d = dataclasses.asdict(rr)
    assert "m3" in d
    assert d["m3"] is None

def test_m3_serialization_round_trip():
    m3 = M3ResponseResult(
        clarification=ClarificationState(needed=True, original_query="hi")
    )
    d = dataclasses.asdict(m3)
    assert d["clarification"]["needed"] is True
    assert d["clarification"]["original_query"] == "hi"

def test_fastapi_schema_allows_no_m3():
    # If m3 is omitted, it should be fine. We just test the response parsing
    # of RetrieveResponse manually or by mocking.
    resp = RetrieveResponse(
        request_id="req",
        query="hi",
        query_type="factual",
        classification_confidence=0.9,
        status="answered",
        answer="hello",
        confidence=1.0,
        confidence_level="HIGH",
        citations=[],
        no_information_found=False,
        clarification_needed=False
    )
    assert resp.m3 is None

def test_fastapi_schema_with_m3():
    resp = RetrieveResponse(
        request_id="req",
        query="hi",
        query_type="factual",
        classification_confidence=0.9,
        status="answered",
        answer="hello",
        confidence=1.0,
        confidence_level="HIGH",
        citations=[],
        no_information_found=False,
        clarification_needed=False,
        m3={"clarification": {"needed": True}}
    )
    assert resp.m3 is not None
    assert resp.m3.clarification.needed is True

def test_fastapi_schema_malformed_m3():
    with pytest.raises(ValidationError):
        RetrieveResponse(
            request_id="req",
            query="hi",
            query_type="factual",
            classification_confidence=0.9,
            status="answered",
            answer="hello",
            confidence=1.0,
            confidence_level="HIGH",
            citations=[],
            no_information_found=False,
            clarification_needed=False,
            m3={"clarification": {"needed": "not_a_bool"}}  # malformed
        )
