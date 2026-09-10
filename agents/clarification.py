"""ClarificationAgent — handle vague or low-confidence queries."""

from __future__ import annotations

from typing import List, Optional

from agents.models import ParsedQuery, RetrievalHit


class ClarificationAgent:
    """Generate clarifying questions for inadequate queries."""

    def should_clarify(
        self,
        parsed: ParsedQuery,
        hits: List[RetrievalHit],
        confidence: float,
        low_confidence_threshold: float,
    ) -> bool:
        if parsed.is_ambiguous:
            return True
        if not hits:
            return parsed.intent != "unavailable"
        if confidence < low_confidence_threshold and parsed.intent != "unavailable":
            return True
        return False

    def generate_question(
        self,
        parsed: ParsedQuery,
        hits: Optional[List[RetrievalHit]] = None,
    ) -> str:
        if not parsed.domain:
            return (
                "Could you specify whether your question relates to "
                "Software Engineering or Hospital Administration?"
            )
        if parsed.is_ambiguous:
            return (
                f"Your question seems broad. Could you provide more detail about "
                f"what you need regarding {parsed.domain}?"
            )
        if not hits:
            return (
                f"I could not find relevant documents for your {parsed.intent} question. "
                "Could you rephrase or add more context?"
            )
        return (
            "I found only weakly related information. "
            "Could you clarify the specific topic or document you mean?"
        )
