"""QueryUnderstandingAgent — normalize, classify intent, detect domain."""

from __future__ import annotations

import re
from typing import Optional

from agents.models import ParsedQuery

INTENTS = ("factual", "procedural", "comparative", "unavailable")

_UNAVAILABLE_PATTERNS = (
    r"\brevenue\b",
    r"\bCEO\b",
    r"\bNobel Prize\b",
    r"invented .{0,30} year",
    r"fiscal year",
)

_PROCEDURAL_PATTERNS = (
    r"\bhow (?:do|to|should|can)\b",
    r"\bwhat are the steps\b",
    r"\bprocedure\b",
    r"\bprotocol\b",
    r"\bfollow\b",
)

_COMPARATIVE_PATTERNS = (
    r"\bdiffer\b",
    r"\bdifference\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bcontrast\b",
)

_SOFTWARE_KEYWORDS = (
    "git", "sprint", "agile", "microservice", "programming", "code",
    "software", "branch", "pull request", "class", "logging", "rust", "go",
)

_HOSPITAL_KEYWORDS = (
    "patient", "nursing", "medication", "hospital", "admission",
    "code blue", "code red", "hand hygiene", "acetaminophen", "ibuprofen",
    "emergency", "dose",
)

_VAGUE_PRONOUNS = re.compile(r"\b(it|this|that|they|them)\b", re.I)


class QueryUnderstandingAgent:
    """Rule-based query understanding (no LLM)."""

    def normalize(self, query: str) -> str:
        text = query.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(re.search(p, q) for p in _UNAVAILABLE_PATTERNS):
            return "unavailable"
        if any(re.search(p, q) for p in _COMPARATIVE_PATTERNS):
            return "comparative"
        if any(re.search(p, q) for p in _PROCEDURAL_PATTERNS):
            return "procedural"
        return "factual"

    def detect_domain(self, query: str) -> Optional[str]:
        q = query.lower()
        software = sum(1 for kw in _SOFTWARE_KEYWORDS if kw in q)
        hospital = sum(1 for kw in _HOSPITAL_KEYWORDS if kw in q)
        if software > hospital and software > 0:
            return "Software Engineering"
        if hospital > software and hospital > 0:
            return "Hospital Administration"
        return None

    def is_ambiguous(self, query: str) -> bool:
        words = query.split()
        if len(words) <= 2:
            return True
        if len(words) <= 4 and _VAGUE_PRONOUNS.search(query):
            return True
        if query.endswith("?" ) and len(words) <= 3:
            return True
        return False

    def analyze(self, query: str) -> ParsedQuery:
        normalized = self.normalize(query)
        return ParsedQuery(
            raw_query=query,
            normalized_query=normalized,
            intent=self.classify_intent(normalized),
            domain=self.detect_domain(normalized),
            is_ambiguous=self.is_ambiguous(normalized),
        )
