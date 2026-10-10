"""M2 retrieval agent built on the existing semantic retriever."""

from __future__ import annotations

import os
import math
import re
from typing import Any, List, Optional

from agents.models import QueryType, RetrievalHit, RetrievalResult
from agents.query_understanding import QueryUnderstandingAgent
from retrieval.retriever import SemanticRetriever

DEFAULT_TOP_K = 3
DEFAULT_MIN_RELEVANCE = 0.40
DEFAULT_RELEVANCE_MARGIN = 0.10
DEFAULT_STRONG_RELEVANCE = 0.75
FINANCIAL_CONCEPTS = (
    frozenset({"earnings", "income", "revenue", "revenues", "sales", "turnover"}),
    frozenset({"margin", "profit", "profits"}),
    frozenset({
        "budget", "budgets", "cost", "costs", "expenditure", "expenditures",
        "expense", "expenses",
    }),
)
FINANCIAL_CONTEXT_TERMS = frozenset({"finance", "financial", "financing", "fiscal"})


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
        self.query_understanding = QueryUnderstandingAgent()
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
        if "distance_score" in item:
            value = item["distance_score"]
        elif "similarity_score" in item:
            value = item["similarity_score"]
        else:
            raise ValueError("Retriever returned no L2 distance score")
        try:
            distance = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Retriever returned a non-numeric L2 distance") from exc
        if not math.isfinite(distance):
            raise ValueError("Retriever returned a non-finite L2 distance")
        if distance < 0:
            raise ValueError("Retriever returned a negative L2 distance")
        return distance

    @staticmethod
    def _result_key(item: dict) -> tuple:
        return (
            item.get("chunk_id"),
            item.get("document_id"),
            item.get("filename") or item.get("document_name"),
            item.get("text"),
        )

    @staticmethod
    def _relevance(distance_score: float) -> float:
        """Convert lower-is-better L2 distance into bounded relevance."""
        return 1.0 / (1.0 + distance_score)

    @staticmethod
    def _has_direct_query_support(query: str, hit: RetrievalHit) -> bool:
        stop_words = {
            "about", "after", "also", "are", "before", "between", "could",
            "did", "does", "from", "have", "how", "into", "main", "much",
            "that", "their", "there", "these", "they", "this", "those",
            "through", "what", "when", "where", "which", "while", "will",
            "with", "would",
        }
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) >= 4 and term not in stop_words
        }
        hit_terms = set(re.findall(r"[a-z0-9]+", hit.text.lower()))
        return len(query_terms & hit_terms) >= 2

    @staticmethod
    def _has_financial_topic_support(query: str, hit: RetrievalHit) -> bool:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        requested_concepts = [
            concept for concept in FINANCIAL_CONCEPTS if query_terms.intersection(concept)
        ]
        hit_terms = set(re.findall(r"[a-z0-9]+", hit.text.lower()))
        if requested_concepts:
            return any(hit_terms.intersection(concept) for concept in requested_concepts)
        if not query_terms.intersection(FINANCIAL_CONTEXT_TERMS):
            return True
        if query_terms.intersection(FINANCIAL_CONTEXT_TERMS):
            return bool(
                hit_terms.intersection(FINANCIAL_CONTEXT_TERMS)
                or any(hit_terms.intersection(concept) for concept in FINANCIAL_CONCEPTS)
            )
        return True

    @staticmethod
    def _has_clear_relevance_signal(
        hits: List[RetrievalHit], threshold: float, query: str
    ) -> bool:
        """Guard against weak near-neighbor false positives without collapsing valid evidence.

        The absolute relevance floor is the primary gate. A strong valid result is kept
        even when multiple similar chunks are present. A weak borderline cluster is only
        rejected as a secondary signal when it is not meaningfully stronger than the
        adjacent candidate and is below the strong-evidence floor.
        """
        if not hits:
            return False

        top_hit = hits[0].relevance_score
        if top_hit < threshold:
            return False
        if not RetrievalAgent._has_financial_topic_support(query, hits[0]):
            return False

        # Weak signals (< 0.50) must have direct query support to avoid false positives.
        if top_hit < 0.50 and not RetrievalAgent._has_direct_query_support(query, hits[0]):
            return False

        if len(hits) == 1:
            return True

        second_hit = hits[1].relevance_score
        if top_hit >= DEFAULT_STRONG_RELEVANCE:
            return True

        same_document = len({hit.document_id for hit in hits if hit.document_id}) <= 1
        if same_document and top_hit >= threshold:
            return True

        if (top_hit - second_hit) >= DEFAULT_RELEVANCE_MARGIN:
            return True

        return RetrievalAgent._has_direct_query_support(query, hits[0])

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
        comparison_target_keys = []
        if normalized_type == "comparative":
            targets = self.query_understanding.extract_comparison_targets(query)
            if len(targets) >= 2:
                for target in targets:
                    target_results = self.retriever.retrieve(target, top_k=1) or []
                    raw_results.extend(target_results)
                    comparison_target_keys.append({
                        self._result_key(item)
                        for item in target_results
                        if isinstance(item, dict)
                    })

        deduplicated_by_chunk = {}
        non_dictionary_results = []
        for item in raw_results:
            if not isinstance(item, dict):
                non_dictionary_results.append(item)
                continue
            key = self._result_key(item)
            current = deduplicated_by_chunk.get(key)
            if current is None or self._distance(item) < self._distance(current):
                deduplicated_by_chunk[key] = item
        deduplicated_results = [
            *deduplicated_by_chunk.values(),
            *non_dictionary_results,
        ]
        ranked_items = sorted(deduplicated_results, key=self._distance)
        hits: List[RetrievalHit] = []
        filtered_count = 0

        for item in ranked_items:
            if not isinstance(item, dict):
                filtered_count += 1
                continue
            metadata = item.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            record_domain = item.get("domain")
            metadata_domain = metadata.get("domain")
            if record_domain and metadata_domain and record_domain != metadata_domain:
                filtered_count += 1
                continue
            hit_domain = record_domain or metadata_domain or ""
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
            if normalized_type != "comparative" and len(hits) >= requested_top_k:
                break

        if normalized_type == "comparative" and comparison_target_keys:
            reserved_hits = []
            reserved_keys = set()
            for target_keys in comparison_target_keys:
                target_hit = next(
                    (
                        hit for hit in hits
                        if (hit.chunk_id, hit.document_id, hit.filename, hit.text)
                        in target_keys
                    ),
                    None,
                )
                if target_hit is not None:
                    key = (target_hit.chunk_id, target_hit.document_id, target_hit.filename, target_hit.text)
                    if key not in reserved_keys:
                        reserved_hits.append(target_hit)
                        reserved_keys.add(key)

            selected_hits = reserved_hits[:requested_top_k]
            selected_keys = {
                (hit.chunk_id, hit.document_id, hit.filename, hit.text)
                for hit in selected_hits
            }
            selected_hits.extend(
                hit for hit in hits
                if len(selected_hits) < requested_top_k
                and (hit.chunk_id, hit.document_id, hit.filename, hit.text)
                not in selected_keys
            )
            hits = sorted(selected_hits, key=lambda hit: hit.distance_score)
            for rank, hit in enumerate(hits, start=1):
                hit.rank = rank

        confidence = max((hit.relevance_score for hit in hits), default=0.0)
        if normalized_type != "comparative" and hits:
            if threshold <= 0.0:
                pass
            elif not self._has_clear_relevance_signal(hits, threshold, query):
                filtered_count += len(hits)
                hits = []
                confidence = 0.0
            elif len(hits) > 1:
                filtered_count += len(hits) - 1
                hits = [hits[0]]
                confidence = hits[0].relevance_score
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
