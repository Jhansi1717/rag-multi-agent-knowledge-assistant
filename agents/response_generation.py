"""ResponseGenerationAgent — minimal extractive grounded responses."""

from __future__ import annotations

import re
from typing import List, Set

from agents.models import AgentResponse, Citation, RetrievalHit


class ResponseGenerationAgent:
    """Build answers from retrieved chunks without an LLM."""

    def _keywords(self, query: str) -> Set[str]:
        stop = {"a", "an", "the", "is", "are", "what", "how", "do", "does", "in", "of", "to", "for"}
        return {w.lower() for w in re.findall(r"\w+", query) if w.lower() not in stop and len(w) > 2}

    def _units(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: List[str] = []
        for para in paragraphs:
            parts = re.split(r"(?<=(?<!\d)[.!?])\s+(?=[A-Z])", para)
            units.extend(p.strip() for p in parts if p.strip() and len(p.strip()) > 20)
        return units or ([text.strip()] if text.strip() else [])

    def _best_excerpt(self, query: str, text: str) -> str:
        units = self._units(text)
        keywords = self._keywords(query)
        if not units:
            return text[:400]
        if not keywords:
            return units[0][:400]

        def score(unit: str) -> int:
            lower = unit.lower()
            return sum(1 for kw in keywords if kw in lower)

        ranked = sorted(units, key=score, reverse=True)
        return ranked[0][:400]

    def generate(
        self,
        query: str,
        hits: List[RetrievalHit],
        intent: str,
        confidence: float,
        domain: str | None = None,
    ) -> AgentResponse:
        if not hits:
            return AgentResponse(
                answer="The requested information is not available in the knowledge base.",
                citations=[],
                status="unavailable",
                intent=intent,
                confidence=confidence,
                retrieval_hits=hits,
                domain=domain,
            )

        citations: List[Citation] = []
        excerpts: List[str] = []

        for hit in hits[:3]:
            excerpt = self._best_excerpt(query, hit.text)
            citations.append(
                Citation(
                    chunk_id=hit.chunk_id,
                    filename=hit.filename,
                    excerpt=excerpt,
                    document_id=hit.document_id,
                )
            )
            excerpts.append(excerpt)

        if intent == "comparative" and len(excerpts) >= 2:
            answer = (
                f"Based on the knowledge base: {excerpts[0]} "
                f"Additionally: {excerpts[1]}"
            )
        elif intent == "procedural":
            answer = f"Procedure from {hits[0].filename}: {excerpts[0]}"
        else:
            answer = excerpts[0]

        if citations:
            sources = ", ".join(dict.fromkeys(c.filename for c in citations))
            answer += f" [source: {sources}]"

        return AgentResponse(
            answer=answer,
            citations=citations,
            status="answered",
            intent=intent,
            confidence=confidence,
            retrieval_hits=hits,
            domain=domain,
        )
