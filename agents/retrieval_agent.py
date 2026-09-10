"""RetrievalAgent — embedding search with structured ranked results."""

from __future__ import annotations

from typing import List, Optional

from agents.models import RetrievalHit
from retrieval.retriever import SemanticRetriever


class RetrievalAgent:
    """Wraps SemanticRetriever and normalises hit payloads."""

    def __init__(self, retriever: SemanticRetriever):
        self.retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        domain: Optional[str] = None,
    ) -> List[RetrievalHit]:
        raw_results = self.retriever.retrieve(query, top_k=top_k)
        hits: List[RetrievalHit] = []

        for rank, item in enumerate(raw_results, start=1):
            meta = item.get("metadata") or {}
            if isinstance(meta, dict) is False:
                meta = {}
            hit_domain = item.get("domain") or meta.get("domain", "")
            if domain and hit_domain and hit_domain != domain:
                continue
            hits.append(
                RetrievalHit(
                    rank=len(hits) + 1,
                    score=float(item.get("similarity_score", 0.0)),
                    text=item.get("text", ""),
                    document_id=item.get("document_id", ""),
                    filename=item.get("filename") or item.get("document_name", ""),
                    chunk_id=item.get("chunk_id", ""),
                    metadata={**meta, "domain": hit_domain},
                )
            )

        if domain and not hits and raw_results:
            for rank, item in enumerate(raw_results, start=1):
                meta = item.get("metadata") or {}
                hits.append(
                    RetrievalHit(
                        rank=rank,
                        score=float(item.get("similarity_score", 0.0)),
                        text=item.get("text", ""),
                        document_id=item.get("document_id", ""),
                        filename=item.get("filename") or item.get("document_name", ""),
                        chunk_id=item.get("chunk_id", ""),
                        metadata=meta if isinstance(meta, dict) else {},
                    )
                )

        return hits
