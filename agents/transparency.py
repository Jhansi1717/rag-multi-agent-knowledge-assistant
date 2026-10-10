"""Transparency builder — constructs TransparencyReport from actual pipeline outputs."""

from __future__ import annotations

import re
from typing import List, Optional

from agents.models import (
    ClarificationState,
    Citation,
    EvidenceItem,
    MemoryContext,
    QueryUnderstandingResult,
    ResponseResult,
    RetrievalResult,
    StageTrace,
    TransparencyReport,
)


class TransparencyBuilder:
    """Deterministic report builder — uses actual values only, no LLM calls."""

    def build(
        self,
        understanding: Optional[QueryUnderstandingResult] = None,
        retrieval: Optional[RetrievalResult] = None,
        response: Optional[ResponseResult] = None,
        clarification: Optional[ClarificationState] = None,
        memory: Optional[MemoryContext] = None,
        stages: Optional[List[StageTrace]] = None,
    ) -> TransparencyReport:
        report = TransparencyReport(stages=stages or [])

        # ── Query understanding fields ────────────────────────────────
        if understanding:
            report.query_type = understanding.query_type
            report.classification_reason = understanding.reason
            report.routing = understanding.routing

        # ── Retrieval fields ──────────────────────────────────────────
        if retrieval:
            report.retrieved_count = len(retrieval.results)
            report.filtered_count = retrieval.filtered_count
            report.retrieval_confidence = retrieval.retrieval_confidence
            report.source_evidence = self._build_evidence(retrieval)

        # ── Response fields ───────────────────────────────────────────
        if response:
            report.citations = self._validated_citations(response.citations, retrieval)
            report.grounded = bool(
                response.grounded
                and response.citations
                and len(report.citations) == len(response.citations)
            )

        return report

    @staticmethod
    def _normalized_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", value).strip().lower()

    @classmethod
    def _citation_matches_hit(cls, citation: Citation, hit) -> bool:
        if not citation.chunk_id or not citation.filename:
            return False
        if citation.chunk_id != hit.chunk_id:
            return False
        if citation.filename != hit.filename:
            return False
        if citation.document_id and citation.document_id != hit.document_id:
            return False

        excerpt = citation.excerpt or ""
        if not excerpt:
            return True

        normalized_excerpt = cls._normalized_text(excerpt)
        normalized_hit_text = cls._normalized_text(hit.text)
        return normalized_excerpt in normalized_hit_text

    @staticmethod
    def _validated_citations(
        citations: List[Citation],
        retrieval: Optional[RetrievalResult],
    ) -> List[Citation]:
        if retrieval is None:
            return []

        validated: List[Citation] = []
        for citation in citations:
            for hit in retrieval.results:
                if TransparencyBuilder._citation_matches_hit(citation, hit):
                    validated.append(citation)
                    break
        return validated

    @staticmethod
    def _build_evidence(retrieval: RetrievalResult) -> List[EvidenceItem]:
        """Map each RetrievalHit to an EvidenceItem using actual metadata."""
        evidence: List[EvidenceItem] = []
        for hit in retrieval.results:
            metadata = hit.metadata or {}
            page = metadata.get("page", metadata.get("page_number"))
            page_start = metadata.get("page_start", page)
            page_end = metadata.get("page_end", page)
            if page is None and page_start == page_end:
                page = page_start
            evidence.append(EvidenceItem(
                citation_id=hit.chunk_id,
                document_name=hit.filename,
                chunk_id=hit.chunk_id,
                chunk_text=hit.text,
                similarity_score=hit.relevance_score,
                page=page,
                section=metadata.get("section"),
                page_start=page_start,
                page_end=page_end,
                row_number=metadata.get("row_number"),
                row_start=metadata.get("row_start"),
                row_end=metadata.get("row_end"),
                document_id=hit.document_id,
            ))
        return evidence
