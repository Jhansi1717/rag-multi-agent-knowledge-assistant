"""M2 retrieval agent built on the existing semantic retriever."""

from __future__ import annotations

import os
from typing import Any, List, Optional

from agents.models import QueryType, RetrievalHit, RetrievalResult
from retrieval.retriever import SemanticRetriever

DEFAULT_TOP_K = 3
DEFAULT_MIN_RELEVANCE = 0.45


class RetrievalAgent:
    """Search, rank, filter, and normalize existing vector-store results.

    FAISS returns L2 distance, where lower is better.  The agent preserves that
    value as ``distance_score`` and derives the bounded relevance score
    ``1 / (1 + distance_score)`` for thresholding and confidence reporting.
    This monotonic transformation maps non-negative L2 distances to (0, 1].
    """

    def __init__(
        self,
        retriever: SemanticRetriever,
        top_k: Optional[int] = None,
        min_relevance: Optional[float] = None,
    ):
        self.retriever = retriever
        self.top_k = self._positive_int(
            top_k if top_k is not None else os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K),
            "top_k",
        )
        self.min_relevance = self._relevance_threshold(
            min_relevance
            if min_relevance is not None
            else os.getenv("RETRIEVAL_MIN_RELEVANCE", DEFAULT_MIN_RELEVANCE)
        )

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    @staticmethod
    def _relevance_threshold(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_relevance must be a number between 0 and 1") from exc
        if not 0.0 <= parsed <= 1.0:
            raise ValueError("min_relevance must be a number between 0 and 1")
        return parsed

    @staticmethod
    def _query_type(query_type: Any) -> QueryType:
        value = getattr(query_type, "query_type", query_type)
        if value not in {"factual", "procedural", "comparative", "ambiguous"}:
            raise ValueError(f"Unsupported query type: {value!r}")
        return value

    @staticmethod
    def _distance(item: dict) -> float:
        value = item.get("distance_score", item.get("similarity_score", 0.0))
        try:
            distance = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Retriever returned a non-numeric L2 distance") from exc
        if distance < 0:
            raise ValueError("Retriever returned a negative L2 distance")
        return distance

    @staticmethod
    def _relevance(distance_score: float) -> float:
        """Convert lower-is-better L2 distance into bounded relevance."""
        return 1.0 / (1.0 + distance_score)

    def retrieve(
        self,
        query: str,
        query_type: QueryType = "factual",
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
        min_relevance: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> RetrievalResult:
        requested_top_k = self._positive_int(
            top_k if top_k is not None else self.top_k,
            "top_k",
        )
        threshold = self._relevance_threshold(
            min_relevance if min_relevance is not None else self.min_relevance
        )
        normalized_type = self._query_type(query_type)
        raw_results = self.retriever.retrieve(query, top_k=requested_top_k) or []
        ranked_items = sorted(raw_results, key=self._distance)
        hits: List[RetrievalHit] = []
        filtered_count = 0

        for item in ranked_items:
            if not isinstance(item, dict):
                filtered_count += 1
                continue
            metadata = item.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            hit_domain = item.get("domain") or metadata.get("domain", "")
            if domain and hit_domain != domain:
                filtered_count += 1
                continue

            distance = self._distance(item)
            relevance = self._relevance(distance)
            if relevance < threshold:
                filtered_count += 1
                continue

            preserved_metadata = dict(metadata)
            if hit_domain:
                preserved_metadata["domain"] = hit_domain
            hits.append(
                RetrievalHit(
                    rank=len(hits) + 1,
                    relevance_score=relevance,
                    distance_score=distance,
                    text=str(item.get("text", "")),
                    document_id=str(item.get("document_id", "")),
                    filename=str(
                        item.get("filename") or item.get("document_name") or ""
                    ),
                    chunk_id=str(item.get("chunk_id", "")),
                    metadata=preserved_metadata,
                )
            )

        confidence = max((hit.relevance_score for hit in hits), default=0.0)
        return RetrievalResult(
            query=query,
            query_type=normalized_type,
            top_k=requested_top_k,
            results=hits,
            filtered_count=filtered_count,
            retrieval_confidence=confidence,
            sufficient_evidence=bool(hits),
            no_relevant_information=not hits,
        )
