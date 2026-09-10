"""Orchestrator — deterministic multi-agent retrieval pipeline."""

from __future__ import annotations

from typing import Optional

from agents.clarification import ClarificationAgent
from agents.memory import ConversationMemoryAgent
from agents.models import AgentResponse, Citation
from agents.query_understanding import QueryUnderstandingAgent
from agents.response_generation import ResponseGenerationAgent
from agents.retrieval_agent import RetrievalAgent
from retrieval.retriever import SemanticRetriever
from vector_store.store import VectorStore


class Orchestrator:
    """Coordinate Understanding → Retrieval → Response with clarification fallback."""

    DEFAULT_TOP_K = 3
    CONFIDENCE_THRESHOLD = 1.35

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = DEFAULT_TOP_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        retriever = SemanticRetriever(vector_store)
        self.understanding = QueryUnderstandingAgent()
        self.retrieval = RetrievalAgent(retriever)
        self.response_gen = ResponseGenerationAgent()
        self.clarification = ClarificationAgent()
        self.memory = ConversationMemoryAgent()
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

    def _confidence(self, hits) -> float:
        if not hits:
            return 0.0
        score = hits[0].score
        return max(0.0, 1.0 / (1.0 + score))

    def _is_retrieval_inadequate(self, parsed, hits, confidence: float) -> bool:
        if not hits:
            return True
        if parsed.intent == "unavailable":
            return hits[0].score > self.confidence_threshold or confidence < 0.45
        return hits[0].score > self.confidence_threshold

    def handle(
        self,
        query: str,
        session_id: str = "default",
        top_k: Optional[int] = None,
    ) -> AgentResponse:
        k = top_k or self.top_k
        context = self.memory.get_recent_context(session_id)
        enriched_query = f"{context} {query}".strip() if context else query

        parsed = self.understanding.analyze(query)
        if parsed.is_ambiguous and context:
            parsed = self.understanding.analyze(enriched_query)

        hits = self.retrieval.retrieve(
            parsed.normalized_query,
            top_k=k,
            domain=parsed.domain,
        )
        confidence = self._confidence(hits)

        if parsed.intent == "unavailable":
            response = AgentResponse(
                answer="The requested information is not available in the knowledge base.",
                citations=[],
                status="unavailable",
                intent=parsed.intent,
                confidence=confidence,
                retrieval_hits=hits,
                domain=parsed.domain,
            )
            self.memory.add_turn(session_id, query, response.answer)
            return response

        if self.clarification.should_clarify(parsed, hits, confidence, 0.45):
            question = self.clarification.generate_question(parsed, hits)
            return AgentResponse(
                answer=question,
                citations=[],
                status="clarification_needed",
                intent=parsed.intent,
                confidence=confidence,
                retrieval_hits=hits,
                clarification_question=question,
                domain=parsed.domain,
            )

        if self._is_retrieval_inadequate(parsed, hits, confidence):
            if parsed.intent == "unavailable":
                return AgentResponse(
                    answer="The requested information is not available in the knowledge base.",
                    citations=[
                        Citation(
                            chunk_id=h.chunk_id,
                            filename=h.filename,
                            excerpt=h.text[:120],
                            document_id=h.document_id,
                        )
                        for h in hits[:1]
                    ] if hits else [],
                    status="unavailable",
                    intent=parsed.intent,
                    confidence=confidence,
                    retrieval_hits=hits,
                    domain=parsed.domain,
                )
            question = self.clarification.generate_question(parsed, hits)
            return AgentResponse(
                answer=question,
                citations=[],
                status="clarification_needed",
                intent=parsed.intent,
                confidence=confidence,
                retrieval_hits=hits,
                clarification_question=question,
                domain=parsed.domain,
            )

        response = self.response_gen.generate(
            parsed.normalized_query,
            hits,
            intent=parsed.intent,
            confidence=confidence,
            domain=parsed.domain,
        )
        self.memory.add_turn(session_id, query, response.answer)
        return response
