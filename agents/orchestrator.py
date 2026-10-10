"""M3 Orchestrator — full M3 pipeline with clarification lifecycle and stage tracking."""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import List, Optional

from agents.clarification import ClarificationAgent
from agents.context_resolution import ContextResolutionAgent
from agents.memory import ConversationMemoryAgent
from agents.models import (
    AgentError,
    ClarificationResult,
    ClarificationState,
    ConversationTurn,
    ContextResolutionResult,
    M3ResponseResult,
    MemoryContext,
    QueryUnderstandingResult,
    ResponseResult,
    RetrievalResult,
    StageTrace,
    TransparencyReport,
    _STATUS_MAP,
)
from agents.query_understanding import QueryUnderstandingAgent
from agents.response_generation import ResponseGenerationAgent
from agents.retrieval_agent import RetrievalAgent
from agents.transparency import TransparencyBuilder
from retrieval.retriever import SemanticRetriever
from vector_store.store import VectorStore


class Orchestrator:
    """M3 pipeline: Memory → Clarification → Resolution → QU → Retrieval → Response."""

    DEFAULT_TOP_K = 3

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = DEFAULT_TOP_K,
        confidence_threshold: Optional[float] = None,
        understanding: Optional[QueryUnderstandingAgent] = None,
        retrieval: Optional[RetrievalAgent] = None,
        response_gen: Optional[ResponseGenerationAgent] = None,
        clarification: Optional[ClarificationAgent] = None,
    ):
        self.understanding = understanding or QueryUnderstandingAgent()
        self.retrieval = retrieval or RetrievalAgent(
            SemanticRetriever(vector_store),
            top_k=top_k,
            min_relevance=confidence_threshold,
        )
        self.response_gen = response_gen or ResponseGenerationAgent()
        self.memory = ConversationMemoryAgent()
        self.context_resolution = ContextResolutionAgent(qu_agent=self.understanding)
        # ClarificationAgent gets its own real QU agent — it must not share a
        # potentially-mocked instance from callers who inject a FakeUnderstanding.
        self.clarification_agent = clarification or ClarificationAgent()
        self.transparency_builder = TransparencyBuilder()
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _request_id(request_id: Optional[str]) -> str:
        return request_id or str(uuid.uuid4())

    @staticmethod
    def _stage(name: str, start: float) -> StageTrace:
        return StageTrace(
            stage_name=name,
            status="completed",
            duration_ms=round((time.monotonic() - start) * 1000, 2),
        )

    @staticmethod
    def _stage_error(name: str, start: float) -> StageTrace:
        return StageTrace(
            stage_name=name,
            status="error",
            duration_ms=round((time.monotonic() - start) * 1000, 2),
        )

    @staticmethod
    def _error_response(
        request_id: str,
        query_type: str,
        agent: str,
        code: str,
        message: str,
        stages: Optional[List[StageTrace]] = None,
    ) -> ResponseResult:
        return ResponseResult(
            answer="The request could not be completed.",
            confidence=0.0,
            grounded=False,
            no_information_found=True,
            query_type=query_type,
            status="error",
            orchestration_status="error",
            request_id=request_id,
            error=AgentError(agent=agent, code=code, message=message),
            m3=M3ResponseResult(
                transparency=TransparencyReport(stages=stages or [])
            ),
        )

    @staticmethod
    def _validate_understanding(parsed: QueryUnderstandingResult) -> None:
        if parsed.query_type not in {"factual", "procedural", "comparative", "ambiguous"}:
            raise ValueError("QueryUnderstandingAgent returned invalid query type")
        if parsed.routing not in {"RETRIEVAL", "CLARIFICATION"}:
            raise ValueError("QueryUnderstandingAgent returned invalid routing")
        if not parsed.normalized_query:
            raise ValueError("QueryUnderstandingAgent returned an empty normalized query")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(
        self,
        query: str,
        session_id: str = "default",
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
        clarification_id: Optional[str] = None,
    ) -> ResponseResult:
        correlation_id = self._request_id(request_id)
        stages: List[StageTrace] = []

        # ── Stage: memory ────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            self.memory.create_session(session_id)
            context = self.memory.get_relevant_context(session_id, query)
            session_turns = self.memory.get_session(session_id)
            latest_active_topics = session_turns[-1].active_topics if session_turns else []
            latest_referenced_entities = (
                session_turns[-1].referenced_entities if session_turns else []
            )
            pending_clarification = self.memory.get_pending_clarification(session_id)
        except Exception as exc:
            stages.append(self._stage_error("memory", t0))
            return self._error_response(
                correlation_id, "factual", "memory", "MEMORY_ERROR", str(exc), stages
            )
        stages.append(self._stage("memory", t0))

        # ── Stage: clarification ─────────────────────────────────────
        t0 = time.monotonic()
        resolved_query = query
        resolved_entities: List[str] = []
        clarification_answer: Optional[str] = None
        memory_context_used = False

        if pending_clarification and not pending_clarification.resolved:
            # Resume from a pending clarification — user answer is in `query`
            try:
                provided_clarification_id = clarification_id or pending_clarification.clarification_id
                resolved_query = self.memory.resolve_clarification(
                    session_id, provided_clarification_id, query
                )
                clarification_answer = pending_clarification.clarification_answer
                memory_context_used = True
                stages.append(self._stage("clarification", t0))
            except Exception as exc:
                stages.append(self._stage_error("clarification", t0))
                return self._error_response(
                    correlation_id, "ambiguous", "clarification",
                    "CLARIFICATION_RESOLUTION_ERROR", str(exc), stages
                )
        else:
            # No pending clarification — contextual resolution FIRST
            t0 = time.monotonic()
            try:
                resolution = self.context_resolution.resolve(
                    query=query,
                    context=context,
                    clarification_state=pending_clarification,
                    active_topics=latest_active_topics,
                    referenced_entities=latest_referenced_entities,
                )
                if not isinstance(resolution, ContextResolutionResult):
                    raise TypeError("ContextResolutionAgent returned malformed output")
            except Exception as exc:
                stages.append(self._stage_error("context_resolution", t0))
                return self._error_response(
                    correlation_id, "ambiguous", "context_resolution",
                    "CONTEXT_RESOLUTION_ERROR", str(exc), stages
                )
            resolved_query = (
                resolution.resolved_query if resolution.context_used else query
            )
            resolved_entities = resolution.referenced_entities
            memory_context_used = resolution.context_used

            # Then evaluate the RESOLVED query for clarification
            clarification_started = time.monotonic()
            if resolution.resolution_reason.startswith("Insufficient context"):
                clarification_result = ClarificationResult(
                    original_query=resolved_query,
                    clarification_needed=True,
                    ambiguity_type="unresolved_reference",
                    reason=resolution.resolution_reason,
                    clarification_question=(
                        "Which topic or item does that reference refer to?"
                    ),
                    clarification_id=str(uuid.uuid4()),
                )
            else:
                try:
                    clarification_result = self.clarification_agent.analyze(resolved_query)
                    if not isinstance(clarification_result, ClarificationResult):
                        raise TypeError("ClarificationAgent returned malformed output")
                except Exception as exc:
                    stages.append(self._stage_error("clarification", clarification_started))
                    return self._error_response(
                        correlation_id, "ambiguous", "clarification",
                        "CLARIFICATION_ERROR", str(exc), stages
                    )
            if clarification_result.clarification_needed:
                stages.append(self._stage("clarification", clarification_started))
                # Persist state so next turn can resume
                state = ClarificationState(
                    original_query=query,
                    needed=True,
                    clarification_id=clarification_result.clarification_id,
                    question=clarification_result.clarification_question,
                    ambiguity_type=clarification_result.ambiguity_type,
                )
                answer = clarification_result.clarification_question or (
                    "Could you provide more detail so the request can be answered?"
                )
                memory_started = time.monotonic()
                try:
                    self.memory.set_pending_clarification(session_id, state)
                    query_topics = self.understanding.extract_conversational_entities(query)
                    self.memory.append_turn(
                        session_id,
                        ConversationTurn(
                            session_id=session_id,
                            turn_index=0,
                            query=query,
                            answer=answer,
                            query_classification="ambiguous",
                            clarification_question=clarification_result.clarification_question,
                            refined_query=resolved_query if resolved_query != query else None,
                            active_topics=query_topics,
                            referenced_entities=query_topics,
                        ),
                    )
                except Exception as exc:
                    stages.append(self._stage_error("memory", memory_started))
                    return self._error_response(
                        correlation_id, "ambiguous", "memory",
                        "CLARIFICATION_STATE_ERROR", str(exc), stages
                    )

                return ResponseResult(
                    answer=answer,
                    confidence=0.5,
                    classification_confidence=0.5,
                    confidence_level="MEDIUM",
                    grounded=False,
                    no_information_found=False,
                    query_type="ambiguous",
                    status="clarification_needed",
                    orchestration_status="requires_clarification",
                    request_id=correlation_id,
                    m3=M3ResponseResult(
                        clarification=state,
                        memory=MemoryContext(
                            session_id=session_id,
                            relevant_context=context if memory_context_used else "",
                            used_context=memory_context_used,
                        ),
                        transparency=TransparencyReport(stages=stages),
                    ),
                )

            stages.append(self._stage("clarification", clarification_started))

        # ── Stage: query_understanding ────────────────────────────────
        t0 = time.monotonic()
        try:
            parsed = self.understanding.analyze(resolved_query)
            if not isinstance(parsed, QueryUnderstandingResult):
                raise TypeError("QueryUnderstandingAgent returned malformed output")
            self._validate_understanding(parsed)
        except Exception as exc:
            stages.append(self._stage_error("query_understanding", t0))
            return self._error_response(
                correlation_id, "ambiguous", "query_understanding",
                "CLASSIFICATION_ERROR", str(exc), stages
            )
        stages.append(self._stage("query_understanding", t0))

        # Safety net: QU still says CLARIFICATION (e.g. very short resolved query)
        if parsed.routing == "CLARIFICATION":
            query_topics = self.understanding.extract_conversational_entities(query)
            turn = ConversationTurn(
                session_id=session_id,
                turn_index=0,
                query=query,
                answer="Could you provide more detail so the request can be answered?",
                query_classification=parsed.query_type,
                refined_query=resolved_query if resolved_query != query else None,
                active_topics=query_topics,
                referenced_entities=query_topics,
            )
            try:
                self.memory.append_turn(session_id, turn)
            except Exception as exc:
                stages.append(self._stage_error("memory", time.monotonic()))
                return self._error_response(
                    correlation_id, parsed.query_type, "memory",
                    "MEMORY_PERSISTENCE_ERROR", str(exc), stages
                )
            return ResponseResult(
                answer="Could you provide more detail so the request can be answered?",
                confidence=parsed.classification_confidence,
                classification_confidence=parsed.classification_confidence,
                confidence_level="MEDIUM",
                grounded=False,
                no_information_found=False,
                query_type=parsed.query_type,
                status="clarification_needed",
                orchestration_status="requires_clarification",
                request_id=correlation_id,
                domain=parsed.domain,
                m3=M3ResponseResult(
                    memory=MemoryContext(
                        session_id=session_id,
                        relevant_context=context if memory_context_used else "",
                        used_context=memory_context_used,
                    ),
                    transparency=TransparencyReport(stages=stages),
                ),
            )

        # ── Stage: retrieval ──────────────────────────────────────────
        t0 = time.monotonic()
        try:
            retrieval = self.retrieval.retrieve(
                parsed.normalized_query,
                query_type=parsed.query_type,
                domain=parsed.domain,
                top_k=top_k if top_k is not None else self.top_k,
                request_id=correlation_id,
            )
            if not isinstance(retrieval, RetrievalResult):
                raise TypeError("RetrievalAgent returned malformed output")
            if retrieval.query != parsed.normalized_query:
                retrieval = replace(retrieval, query=parsed.normalized_query)
            if retrieval.sufficient_evidence and (
                not retrieval.results or retrieval.no_relevant_information
            ):
                raise ValueError("RetrievalAgent returned inconsistent evidence state")
            if retrieval.results and (
                not retrieval.sufficient_evidence or retrieval.no_relevant_information
            ):
                raise ValueError("RetrievalAgent marked returned candidates as unavailable")
            if not retrieval.results and not retrieval.no_relevant_information:
                retrieval = replace(retrieval, no_relevant_information=True)
        except Exception as exc:
            stages.append(self._stage_error("retrieval", t0))
            return self._error_response(
                correlation_id, parsed.query_type, "retrieval",
                "RETRIEVAL_ERROR", str(exc), stages
            )
        stages.append(self._stage("retrieval", t0))
        has_evidence = bool(
            retrieval.results
            and retrieval.sufficient_evidence
            and not retrieval.no_relevant_information
        )

        # ── Stage: response_generation ────────────────────────────────
        t0 = time.monotonic()
        try:
            response = self.response_gen.generate(
                parsed.normalized_query,
                retrieval.results if has_evidence else [],
                intent=parsed.query_type,
                query_type=parsed.query_type,
                confidence=retrieval.retrieval_confidence,
                domain=parsed.domain,
                request_id=correlation_id,
                sufficient_evidence=has_evidence,
                no_relevant_information=not has_evidence,
            )
            if not isinstance(response, ResponseResult):
                raise TypeError("ResponseGenerationAgent returned malformed output")
        except Exception as exc:
            stages.append(self._stage_error("response_generation", t0))
            return self._error_response(
                correlation_id, parsed.query_type, "response_generation",
                "RESPONSE_GENERATION_ERROR", str(exc), stages
            )
        if not has_evidence:
            response = replace(
                response,
                answer=ResponseGenerationAgent.NO_INFORMATION_MESSAGE,
                citations=[],
                confidence=0.0,
                confidence_level="LOW",
                grounded=False,
                no_information_found=True,
                query_type=parsed.query_type,
                retrieval_hits=[],
                domain=parsed.domain,
                status="unavailable",
                orchestration_status="no_relevant_information",
            )
        stages.append(self._stage("response_generation", t0))

        # ── Build transparency + M3 metadata ─────────────────────────
        t0 = time.monotonic()
        try:
            transparency = self.transparency_builder.build(
                understanding=parsed,
                retrieval=retrieval,
                response=response,
                stages=stages,
            )
        except Exception as exc:
            stages.append(self._stage_error("transparency", t0))
            return self._error_response(
                correlation_id, parsed.query_type, "transparency",
                "TRANSPARENCY_ERROR", str(exc), stages
            )
        memory_ctx = MemoryContext(
            session_id=session_id,
            relevant_context=context if memory_context_used else "",
            used_context=memory_context_used,
        )

        response = replace(
            response,
            query_type=parsed.query_type,
            classification_confidence=parsed.classification_confidence,
            request_id=correlation_id,
            retrieval=retrieval,
            retrieval_hits=retrieval.results if has_evidence else [],
            orchestration_status=_STATUS_MAP.get(response.status, "completed"),
            m3=M3ResponseResult(
                clarification=pending_clarification,
                memory=memory_ctx,
                transparency=transparency,
            ),
        )

        # ── Store turn in memory ──────────────────────────────────────
        # Extract active topics from query and answer for context resolution
        # Use conversational entity extraction to capture lowercase concepts
        query_topics = resolved_entities or self.understanding.extract_conversational_entities(resolved_query)
        answer_topics = self.understanding.extract_conversational_entities(response.answer)
        all_entities = list(dict.fromkeys(query_topics + answer_topics))

        turn = ConversationTurn(
            session_id=session_id,
            turn_index=0,
            query=query,
            answer=response.answer,
            query_classification=parsed.query_type,
            refined_query=resolved_query if resolved_query != query else None,
            clarification_answer=clarification_answer,
            referenced_documents=list(set(
                h.filename for h in retrieval.results if h.filename
            )),
            citations=list(set(
                f"{c.filename}#{c.chunk_id}"
                for c in response.citations
                if c.filename or c.chunk_id
            )),
            active_topics=list(dict.fromkeys(query_topics)),
            referenced_entities=(
                list(dict.fromkeys(answer_topics))
                if len(answer_topics) > 1
                else all_entities
            ),
        )
        t0 = time.monotonic()
        memory_write_started = time.monotonic()
        try:
            self.memory.append_turn(session_id, turn)
        except Exception as exc:
            stages.append(self._stage_error("memory", memory_write_started))
            return self._error_response(
                correlation_id, parsed.query_type, "memory",
                "MEMORY_PERSISTENCE_ERROR", str(exc), stages
            )

        return response
