"""Grounded LLM response generation for M2.3."""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Optional

from agents.models import Citation, QueryType, ResponseResult, RetrievalHit

logger = logging.getLogger(__name__)

try:
    import openai as _openai
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    _openai = None


class ResponseGenerationAgent:
    """Generate answers from supplied retrieval evidence only.

    Confidence policy is application-level, not calibrated probability:
    HIGH is at least 0.75, MEDIUM is 0.45-0.749..., and LOW is below 0.45.
    Confidence is supplied by retrieval and is never inferred from the LLM.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.45
    MEDIUM_CONFIDENCE_THRESHOLD = 0.45
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    NO_INFORMATION_MESSAGE = (
        "The requested information is not available in the provided documents."
    )

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_client: Any = None,
        api_key: Optional[str] = None,
    ):
        self._llm_model = llm_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self._client = llm_client
        if self._client is None and self._api_key and _openai is not None:
            self._client = _openai.OpenAI(api_key=self._api_key)

    @staticmethod
    def _query_type(value: Any) -> QueryType:
        query_type = getattr(value, "query_type", value)
        if query_type not in {"factual", "procedural", "comparative", "ambiguous"}:
            raise ValueError(f"Unsupported query type: {query_type!r}")
        return query_type

    @staticmethod
    def _confidence_level(confidence: float) -> str:
        if confidence >= ResponseGenerationAgent.HIGH_CONFIDENCE_THRESHOLD:
            return "HIGH"
        if confidence >= ResponseGenerationAgent.MEDIUM_CONFIDENCE_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _citation(hit: RetrievalHit) -> Citation:
        return Citation(
            chunk_id=hit.chunk_id,
            filename=hit.filename,
            excerpt=hit.text[:400],
            document_id=hit.document_id,
        )

    def _build_context(self, hits: list[RetrievalHit]) -> str:
        parts = []
        for index, hit in enumerate(hits[:3], start=1):
            parts.append(
                f"[{index}] source_filename={hit.filename!r} "
                f"chunk_id={hit.chunk_id!r} "
                f"relevance_score={hit.relevance_score:.6f} "
                f"distance_score={hit.distance_score:.6f} "
                f"metadata={json.dumps(hit.metadata, ensure_ascii=False, sort_keys=True)}\n"
                f"{hit.text.strip()}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _has_usable_context(hits: list[RetrievalHit]) -> bool:
        return any(
            hit.text.strip() and (hit.filename or hit.chunk_id)
            for hit in hits
        )

    def _build_prompts(
        self,
        query: str,
        query_type: QueryType,
        hits: list[RetrievalHit],
    ) -> tuple[str, str]:
        style = {
            "factual": "Give a concise explanation or direct factual answer.",
            "procedural": "Give numbered steps only when the context supports a procedure.",
            "comparative": "Use a structured comparison supported by the context.",
            "ambiguous": "State that the request needs clarification if the context cannot resolve it.",
        }[query_type]
        system = (
            "Answer using ONLY the supplied context passages. Do not use outside "
            "knowledge or fill gaps with assumptions. If the context is inadequate, "
            f"state that there is insufficient evidence and respond exactly with: "
            f"{self.NO_INFORMATION_MESSAGE} "
            "Preserve source references using only the supplied source_filename values. "
            "Use only supplied metadata= fields for source attribution. Never invent "
            "citations, URLs, document names, or facts. "
            f"{style}"
        )
        user = f"Context:\n{self._build_context(hits)}\n\nQuestion: {query}"
        return system, user

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self._client is None:
            return ""
        response = self._client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else ""

    def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
        intent: str = "factual",
        confidence: float = 0.0,
        domain: str | None = None,
        query_type: Optional[QueryType] = None,
        request_id: Optional[str] = None,
    ) -> ResponseResult:
        normalized_type = self._query_type(query_type or intent)
        confidence = max(0.0, min(1.0, float(confidence)))
        citations = [self._citation(hit) for hit in hits[:3]]
        level = self._confidence_level(confidence)

        if (
            not hits
            or not self._has_usable_context(hits)
            or confidence < self.LOW_CONFIDENCE_THRESHOLD
        ):
            return ResponseResult(
                answer=self.NO_INFORMATION_MESSAGE,
                citations=[],
                confidence=confidence,
                confidence_level=level,
                grounded=False,
                no_information_found=True,
                query_type=normalized_type,
                retrieval_hits=hits,
                domain=domain,
            )

        system_prompt, user_prompt = self._build_prompts(
            query, normalized_type, hits
        )
        try:
            answer = self._call_llm(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("Grounded LLM generation failed: %s", exc)
            answer = ""

        if not answer or answer == self.NO_INFORMATION_MESSAGE:
            return ResponseResult(
                answer=self.NO_INFORMATION_MESSAGE,
                citations=[],
                confidence=confidence,
                confidence_level=level,
                grounded=False,
                no_information_found=True,
                query_type=normalized_type,
                retrieval_hits=hits,
                domain=domain,
            )

        return ResponseResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            confidence_level=level,
            grounded=True,
            no_information_found=False,
            query_type=normalized_type,
            retrieval_hits=hits,
            domain=domain,
        )
