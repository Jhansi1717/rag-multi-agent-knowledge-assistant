"""ClarificationAgent — handle vague or low-confidence queries."""

from __future__ import annotations

from typing import List, Optional

from agents.models import QueryUnderstandingResult, RetrievalHit


class ClarificationAgent:
    """Generate clarifying questions for inadequate queries."""

    def should_clarify(
        self,
        parsed: QueryUnderstandingResult,
        hits: List[RetrievalHit],
        confidence: float,
        low_confidence_threshold: float,
    ) -> bool:
        if parsed.query_type == "ambiguous":
            return True
        return not hits or confidence < low_confidence_threshold

    def generate_question(
        self,
        parsed: QueryUnderstandingResult,
        hits: Optional[List[RetrievalHit]] = None,
    ) -> str:
        if not parsed.domain:
            return (
                "Could you specify whether your question relates to "
                "Software Engineering or Hospital Administration?"
            )
        if parsed.query_type == "ambiguous":
            return (
                f"Your question seems broad. Could you provide more detail about "
                f"what you need regarding {parsed.domain}?"
            )
        if not hits:
            return (
                f"I could not find relevant documents for your {parsed.query_type} question. "
                "Could you rephrase or add more context?"
            )
        return (
            "I found only weakly related information. "
            "Could you clarify the specific topic or document you mean?"
        )
