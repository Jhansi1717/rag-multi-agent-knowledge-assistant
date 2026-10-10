"""Tests for TransparencyBuilder — verify outputs match actual pipeline values exactly."""

from agents.models import (
    Citation,
    QueryUnderstandingResult,
    ResponseResult,
    RetrievalHit,
    RetrievalResult,
    StageTrace,
)
from agents.transparency import TransparencyBuilder


def _qu(query_type="factual", routing="RETRIEVAL", reason="test reason"):
    return QueryUnderstandingResult(
        query="q", normalized_query="q",
        query_type=query_type,
        classification_confidence=0.9,
        routing=routing,
        domain="Software Engineering",
        reason=reason,
    )


def _retrieval(hits=None, confidence=0.85, filtered=1):
    return RetrievalResult(
        query="q", query_type="factual", top_k=3,
        results=hits if hits is not None else [
            RetrievalHit(
                rank=1, relevance_score=0.92,
                text="Docker is a containerization platform.",
                document_id="doc-1", filename="guide.txt", chunk_id="c-1",
                metadata={"page": 3, "section": "Overview"},
            ),
            RetrievalHit(
                rank=2, relevance_score=0.78,
                text="Kubernetes orchestrates containers.",
                document_id="doc-2", filename="infra.txt", chunk_id="c-2",
                metadata={"page": 7},
            ),
        ],
        filtered_count=filtered,
        retrieval_confidence=confidence,
        sufficient_evidence=True,
    )


def _response(grounded=True, citations=None):
    return ResponseResult(
        answer="Grounded answer.",
        confidence=0.85, grounded=grounded,
        query_type="factual",
        citations=citations or [
            Citation(chunk_id="c-1", filename="guide.txt", excerpt="Docker is a containerization platform.", document_id="doc-1"),
        ],
    )


builder = TransparencyBuilder()


# ── Core pipeline values ──────────────────────────────────────────────────────

def test_query_understanding_values_mapped():
    report = builder.build(understanding=_qu("procedural", reason="procedure cue"))
    assert report.query_type == "procedural"
    assert report.classification_reason == "procedure cue"
    assert report.routing == "RETRIEVAL"


def test_retrieval_counts_match_actual_hits():
    ret = _retrieval()
    report = builder.build(retrieval=ret)
    assert report.retrieved_count == 2
    assert report.filtered_count == 1
    assert report.retrieval_confidence == 0.85


def test_grounded_flag_matches_response():
    report_grounded = builder.build(retrieval=_retrieval(), response=_response(grounded=True))
    report_ungrounded = builder.build(retrieval=_retrieval(), response=_response(grounded=False))
    assert report_grounded.grounded is True
    assert report_ungrounded.grounded is False


# ── Source evidence ───────────────────────────────────────────────────────────

def test_source_evidence_maps_every_hit():
    ret = _retrieval()
    report = builder.build(retrieval=ret)
    assert len(report.source_evidence) == 2

    ev1 = report.source_evidence[0]
    assert ev1.citation_id == "c-1"
    assert ev1.document_name == "guide.txt"
    assert ev1.chunk_id == "c-1"
    assert ev1.chunk_text == "Docker is a containerization platform."
    assert ev1.similarity_score == 0.92
    assert ev1.document_id == "doc-1"
    assert ev1.page == 3
    assert ev1.section == "Overview"

    ev2 = report.source_evidence[1]
    assert ev2.citation_id == "c-2"
    assert ev2.document_name == "infra.txt"
    assert ev2.similarity_score == 0.78
    assert ev2.page == 7
    assert ev2.section is None  # No section in metadata


def test_evidence_scores_not_recalculated():
    """Verify similarity_score is taken verbatim from RetrievalHit, never recomputed."""
    hit = RetrievalHit(
        rank=1, relevance_score=0.123456,
        text="exact", document_id="d", filename="f", chunk_id="c",
    )
    ret = _retrieval(hits=[hit], confidence=0.5)
    report = builder.build(retrieval=ret)
    assert report.source_evidence[0].similarity_score == 0.123456
    assert report.retrieval_confidence == 0.5


# ── No-result case ────────────────────────────────────────────────────────────

def test_no_results_produces_empty_evidence():
    empty_ret = _retrieval(hits=[], confidence=0.0, filtered=0)
    no_info_response = _response(grounded=False, citations=[])
    report = builder.build(retrieval=empty_ret, response=no_info_response)
    assert report.source_evidence == []
    assert report.retrieved_count == 0
    assert report.retrieval_confidence == 0.0
    assert report.grounded is False

def test_ingestion_page_spans_and_csv_rows_map_to_evidence():
    hit = RetrievalHit(
        rank=1,
        relevance_score=0.8,
        text="A source passage.",
        document_id="doc-1",
        filename="source.pdf",
        chunk_id="chunk-1",
        metadata={"page_start": 2, "page_end": 4, "row_start": 7, "row_end": 9},
    )

    evidence = builder.build(retrieval=_retrieval(hits=[hit])).source_evidence[0]

    assert evidence.page is None
    assert evidence.page_start == 2
    assert evidence.page_end == 4
    assert evidence.row_start == 7
    assert evidence.row_end == 9
    assert evidence.section is None




# ── Low confidence ────────────────────────────────────────────────────────────

def test_low_confidence_retrieval_preserved():
    low_ret = _retrieval(confidence=0.15, filtered=2)
    report = builder.build(retrieval=low_ret, response=_response(grounded=False))
    assert report.retrieval_confidence == 0.15
    assert report.grounded is False
    # Evidence still present — transparency shows what was retrieved
    assert len(report.source_evidence) == 2


# ── Citations preserved ──────────────────────────────────────────────────────

def test_citations_preserved_from_response():
    citations = [
        Citation(chunk_id="c-1", filename="guide.txt", excerpt="Docker is a containerization platform.", document_id="doc-1"),
        Citation(chunk_id="c-2", filename="infra.txt", excerpt="Kubernetes orchestrates containers.", document_id="doc-2"),
    ]
    report = builder.build(retrieval=_retrieval(), response=_response(citations=citations))
    assert len(report.citations) == 2
    assert report.citations[0].chunk_id == "c-1"
    assert report.citations[1].chunk_id == "c-2"


def test_unmapped_or_fabricated_citations_are_omitted_and_not_grounded():
    response = _response(citations=[
        Citation(chunk_id="missing", filename="unknown.txt", excerpt="invented passage"),
    ])

    report = builder.build(retrieval=_retrieval(), response=response)

    assert report.citations == []
    assert report.grounded is False


def test_valid_citation_matches_normalized_hit_text():
    hit = RetrievalHit(
        rank=1,
        relevance_score=0.96,
        text="Docker is a containerization platform for packaging and running applications.",
        document_id="doc-1",
        filename="guide.txt",
        chunk_id="c-1",
    )
    citation = Citation(
        chunk_id="c-1",
        filename="guide.txt",
        excerpt="docker is a containerization platform for packaging and running applications",
        document_id="doc-1",
    )

    report = builder.build(retrieval=_retrieval(hits=[hit]), response=_response(grounded=True, citations=[citation]))

    assert report.citations == [citation]
    assert report.grounded is True


# ── Stages preserved ─────────────────────────────────────────────────────────

def test_stages_passed_through():
    stages = [
        StageTrace(stage_name="memory", status="completed", duration_ms=1.5),
        StageTrace(stage_name="retrieval", status="completed", duration_ms=12.3),
    ]
    report = builder.build(stages=stages)
    assert len(report.stages) == 2
    assert report.stages[0].stage_name == "memory"
    assert report.stages[1].duration_ms == 12.3


# ── Full pipeline integration ─────────────────────────────────────────────────

def test_full_pipeline_report():
    """End-to-end: all inputs produce a complete, accurate report."""
    qu = _qu("comparative", reason="comparison cue detected")
    ret = _retrieval(confidence=0.81)
    resp = _response(grounded=True)
    stages = [StageTrace(stage_name="query_understanding", status="completed")]

    report = builder.build(
        understanding=qu, retrieval=ret, response=resp, stages=stages,
    )

    assert report.query_type == "comparative"
    assert report.classification_reason == "comparison cue detected"
    assert report.routing == "RETRIEVAL"
    assert report.retrieved_count == 2
    assert report.filtered_count == 1
    assert report.retrieval_confidence == 0.81
    assert report.grounded is True
    assert len(report.source_evidence) == 2
    assert len(report.stages) == 1
    assert report.source_evidence[0].similarity_score == 0.92


def test_citation_validation_requires_matching_chunk_id_filename_and_document_id():
    """Regression test: citations must exactly match retrieval hit identifiers for provenance."""
    hit = RetrievalHit(
        rank=1,
        relevance_score=0.9,
        text="Test passage for validation.",
        document_id="doc-123",
        filename="test.txt",
        chunk_id="chunk-abc",
    )

    # Valid citation: all identifiers match
    valid_citation = Citation(
        chunk_id="chunk-abc",
        filename="test.txt",
        excerpt="test passage for validation",
        document_id="doc-123",
    )

    report = builder.build(
        retrieval=_retrieval(hits=[hit]),
        response=_response(grounded=True, citations=[valid_citation])
    )

    assert len(report.citations) == 1
    assert report.citations[0].chunk_id == "chunk-abc"
    assert report.grounded is True

    # Invalid citation: chunk_id mismatch
    invalid_chunk_id = Citation(
        chunk_id="wrong-chunk",
        filename="test.txt",
        excerpt="test passage for validation",
        document_id="doc-123",
    )

    report = builder.build(
        retrieval=_retrieval(hits=[hit]),
        response=_response(grounded=True, citations=[invalid_chunk_id])
    )

    assert len(report.citations) == 0
    assert report.grounded is False

    # Invalid citation: filename mismatch
    invalid_filename = Citation(
        chunk_id="chunk-abc",
        filename="wrong.txt",
        excerpt="test passage for validation",
        document_id="doc-123",
    )

    report = builder.build(
        retrieval=_retrieval(hits=[hit]),
        response=_response(grounded=True, citations=[invalid_filename])
    )

    assert len(report.citations) == 0
    assert report.grounded is False

    # Invalid citation: document_id mismatch (when citation has document_id)
    invalid_doc_id = Citation(
        chunk_id="chunk-abc",
        filename="test.txt",
        excerpt="test passage for validation",
        document_id="wrong-doc",
    )

    report = builder.build(
        retrieval=_retrieval(hits=[hit]),
        response=_response(grounded=True, citations=[invalid_doc_id])
    )

    assert len(report.citations) == 0
    assert report.grounded is False
