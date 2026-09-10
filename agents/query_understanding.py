"""QueryUnderstandingAgent — deterministic M2 query classification."""

from __future__ import annotations

import re
from typing import List, Optional

from agents.models import QueryUnderstandingResult, QueryType

INTENTS = ("factual", "procedural", "comparative", "ambiguous")

_PROCEDURAL_PATTERNS = (
    r"\bhow (?:do|to|should|can)\b",
    r"\bwhat are the steps\b",
    r"\bprocedure\b",
    r"\bprotocol\b",
    r"\bworkflow\b",
    r"\bprocess\b",
    r"\bfollow\b",
    r"\bfive rights\b",
    r"\bmedication administration\b",
)

_COMPARATIVE_PATTERNS = (
    r"\bdiffer\b",
    r"\bdifference\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bcontrast\b",
    r"\bsimilarities\b",
    r"\bsimilarities and differences\b",
)

_SOFTWARE_KEYWORDS = (
    "git", "sprint", "agile", "microservice", "programming", "code",
    "software", "branch", "pull request", "class", "logging", "rust", "go",
    "rest", "soap", "api", "http",
)

_HOSPITAL_KEYWORDS = (
    "patient", "nursing", "medication", "hospital", "admission",
    "code blue", "code red", "hand hygiene", "acetaminophen", "ibuprofen",
    "emergency", "dose",
)

_VAGUE_PRONOUNS = re.compile(r"\b(it|this|that|they|them)\b", re.I)

# Entity extraction: capitalised tokens, quoted strings, domain keywords
_QUOTED = re.compile(r'"([^"]+)"|\u2018([^\u2019]+)\u2019')
_CAPITALISED = re.compile(r'\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*)\b')
_DOMAIN_TERMS = frozenset(
    list(_SOFTWARE_KEYWORDS) + list(_HOSPITAL_KEYWORDS)
    + ["REST", "SOAP", "ICU", "API", "HTTP", "paracetamol", "ibuprofen"]
)


class QueryUnderstandingAgent:
    """Rule-based query understanding (no LLM)."""

    def normalize(self, query: str) -> str:
        text = query.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def classify_intent(self, query: str) -> QueryType:
        q = query.lower()
        if self.is_ambiguous(query):
            return "ambiguous"
        if any(re.search(p, q) for p in _COMPARATIVE_PATTERNS):
            return "comparative"
        if any(re.search(p, q) for p in _PROCEDURAL_PATTERNS):
            return "procedural"
        return "factual"

    def _classification_details(
        self,
        query_type: QueryType,
    ) -> tuple[float, str]:
        if query_type == "ambiguous":
            return (
                0.9,
                "The query is too short, vague, or contains an unresolved referent.",
            )
        if query_type == "comparative":
            return 0.95, "A comparison cue identifies a comparative request."
        if query_type == "procedural":
            return 0.95, "A procedure cue identifies a request for steps or process."
        return 0.8, "No procedural or comparative cue was found; the query requests information."

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
        if not words:
            return True
        if len(words) <= 2:
            return True
        if len(words) <= 4 and _VAGUE_PRONOUNS.search(query):
            return True
        if query.endswith("?") and len(words) <= 3:
            return True
        return False

    def extract_entities(self, query: str) -> List[str]:
        """Return a deduplicated list of meaningful entities for M1 callers."""
        entities: List[str] = []
        for match in _QUOTED.finditer(query):
            entities.append((match.group(1) or match.group(2)).strip())
        query_lower = query.lower()
        for term in sorted(_DOMAIN_TERMS):
            if term.lower() in query_lower:
                entities.append(term)
        for match in _CAPITALISED.finditer(query):
            token = match.group(1)
            if token not in {"The", "What", "How", "Who", "When", "Where", "Why"}:
                entities.append(token)

        seen = set()
        result: List[str] = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                result.append(entity)
        return result

    def analyze(self, query: str) -> QueryUnderstandingResult:
        normalized = self.normalize(query)
        query_type = self.classify_intent(normalized)
        confidence, reason = self._classification_details(query_type)
        return QueryUnderstandingResult(
            query=query,
            normalized_query=normalized,
            query_type=query_type,
            classification_confidence=confidence,
            routing="CLARIFICATION" if query_type == "ambiguous" else "RETRIEVAL",
            domain=self.detect_domain(normalized),
            reason=reason,
        )
