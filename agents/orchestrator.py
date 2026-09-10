"""M2.4 sequential orchestration with structured stage handoffs."""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Optional

from agents.memory import ConversationMemoryAgent
from agents.models import (
    AgentError,
    QueryUnderstandingResult,
    ResponseResult,
    RetrievalResult,
)
from agents.query_understanding import QueryUnderstandingAgent
from agents.response_generation import ResponseGenerationAgent
from agents.retrieval_agent import RetrievalAgent
from retrieval.retriever import SemanticRetriever
from vector_store.store import VectorStore


class Orchestrator:
    """Run Understanding → Retrieval → Response for every answerable query."""

    DEFAULT_TOP_K = 3

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = DEFAULT_TOP_K,
        confidence_threshold: Optional[float] = None,
        understanding: Optional[QueryUnderstandingAgent] = None,
        retrieval: Optional[RetrievalAgent] = None,
        response_gen: Optional[ResponseGenerationAgent] = None,
    ):
        self.understanding = understanding or QueryUnderstandingAgent()
        self.retrieval = retrieval or RetrievalAgent(
            SemanticRetriever(vector_store),
            top_k=top_k,
            min_relevance=confidence_threshold,
        )
        self.response_gen = response_gen or ResponseGenerationAgent()
        self.memory = ConversationMemoryAgent()
        self.top_k = top_k

    @staticmethod
    def _request_id(request_id: Optional[str]) -> str:
        return request_id or str(uuid.uuid4())

    @staticmethod
    def _error_response(
        request_id: str,
        query_type: str,
        agent: str,
        code: str,
        message: str,
    ) -> ResponseResult:
        return ResponseResult(
            answer="The request could not be completed.",
            confidence=0.0,
            grounded=False,
            no_information_found=True,
            query_type=query_type,
            status="error",
            request_id=request_id,
            error=AgentError(agent=agent, code=code, message=message),
        )

    @staticmethod
    def _clarification_response(
        request_id: str,
        parsed: QueryUnderstandingResult,
    ) -> ResponseResult:
        answer = (
            "Could you provide more detail so the request can be answered?"
        )
        return ResponseResult(
            answer=answer,
            confidence=parsed.classification_confidence,
            classification_confidence=parsed.classification_confidence,
            confidence_level="MEDIUM",
            grounded=False,
            no_information_found=False,
            query_type=parsed.query_type,
            status="clarification_needed",
            request_id=request_id,
            domain=parsed.domain,
        )

    def handle(
        self,
        query: str,
        session_id: str = "default",
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> ResponseResult:
        correlation_id = self._request_id(request_id)
        try:
            parsed = self.understanding.analyze(query)
            if not isinstance(parsed, QueryUnderstandingResult):
                raise TypeError("QueryUnderstandingAgent returned malformed output")
        except Exception as exc:
            return self._error_response(
                correlation_id,
                "ambiguous",
                "query_understanding",
                "CLASSIFICATION_ERROR",
                str(exc),
            )

        if parsed.routing == "CLARIFICATION":
            response = self._clarification_response(correlation_id, parsed)
            self.memory.add_turn(session_id, query, response.answer)
            return response

        try:
            retrieval = self.retrieval.retrieve(
                parsed.normalized_query,
                query_type=parsed.query_type,
                domain=parsed.domain,
                top_k=top_k or self.top_k,
            )
            if not isinstance(retrieval, RetrievalResult):
                raise TypeError("RetrievalAgent returned malformed output")
        except Exception as exc:
            return self._error_response(
                correlation_id,
                parsed.query_type,
                "retrieval",
                "RETRIEVAL_ERROR",
                str(exc),
            )

        try:
            response = self.response_gen.generate(
                parsed.normalized_query,
                retrieval.results,
                intent=parsed.query_type,
                query_type=parsed.query_type,
                confidence=retrieval.retrieval_confidence,
                domain=parsed.domain,
            )
            if not isinstance(response, ResponseResult):
                raise TypeError("ResponseGenerationAgent returned malformed output")
        except Exception as exc:
            return self._error_response(
                correlation_id,
                parsed.query_type,
                "response_generation",
                "RESPONSE_GENERATION_ERROR",
                str(exc),
            )

        response = replace(
            response,
            query_type=parsed.query_type,
            classification_confidence=parsed.classification_confidence,
            request_id=correlation_id,
            retrieval=retrieval,
            retrieval_hits=retrieval.results,
        )
        self.memory.add_turn(session_id, query, response.answer)
        return response
