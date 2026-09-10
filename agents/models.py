"""Shared agent data models for M1.4 retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedQuery:
    raw_query: str
    normalized_query: str
    intent: str
    domain: Optional[str] = None
    is_ambiguous: bool = False


@dataclass
class RetrievalHit:
    rank: int
    score: float
    text: str
    document_id: str
    filename: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    chunk_id: str
    filename: str
    excerpt: str
    document_id: str = ""


@dataclass
class AgentResponse:
    answer: str
    citations: List[Citation]
    status: str
    intent: str
    confidence: float
    retrieval_hits: List[RetrievalHit] = field(default_factory=list)
    clarification_question: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class ConversationTurn:
    session_id: str
    query: str
    answer: str
    turn_index: int
