"""Shared agent data models for M1.4 retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

QueryType = Literal["factual", "procedural", "comparative", "ambiguous"]
Routing = Literal["RETRIEVAL", "CLARIFICATION"]


@dataclass
class QueryUnderstandingResult:
    query: str
    normalized_query: str
    query_type: QueryType
    classification_confidence: float
    routing: Routing
    domain: Optional[str] = None
    reason: str = ""

    @property
    def intent(self) -> QueryType:
        """Backward-compatible alias used by the M1 orchestrator."""
        return self.query_type

    @property
    def is_ambiguous(self) -> bool:
        """Backward-compatible ambiguity flag for existing agent callers."""
        return self.query_type == "ambiguous"

    @property
    def raw_query(self) -> str:
        """Backward-compatible alias for the original query."""
        return self.query


@dataclass
class ParsedQuery:
    raw_query: str
    normalized_query: str
    intent: str
    domain: Optional[str] = None
    is_ambiguous: bool = False
    entities: List[str] = field(default_factory=list)


@dataclass(init=False)
class RetrievalHit:
    rank: int
    relevance_score: float
    distance_score: float
    text: str
    document_id: str
    filename: str
    chunk_id: str
    metadata: Dict[str, Any]

    def __init__(
        self,
        rank: int,
        relevance_score: Optional[float] = None,
        text: str = "",
        document_id: str = "",
        filename: str = "",
        chunk_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        *,
        score: Optional[float] = None,
        distance_score: Optional[float] = None,
    ):
        self.rank = rank
        self.relevance_score = (
            relevance_score if relevance_score is not None else (score or 0.0)
        )
        self.distance_score = (
            distance_score if distance_score is not None else self.relevance_score
        )
        self.text = text
        self.document_id = document_id
        self.filename = filename
        self.chunk_id = chunk_id
        self.metadata = metadata if metadata is not None else {}

    @property
    def score(self) -> float:
        """Backward-compatible alias for the M1 distance field."""
        return self.relevance_score

    @score.setter
    def score(self, value: float) -> None:
        self.relevance_score = value


@dataclass
class RetrievalResult:
    query: str
    query_type: QueryType
    top_k: int
    results: List[RetrievalHit] = field(default_factory=list)
    filtered_count: int = 0
    retrieval_confidence: float = 0.0
    sufficient_evidence: bool = False
    no_relevant_information: bool = False

    def __iter__(self):
        """Preserve list-like behavior for existing M1 orchestrator callers."""
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]


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
class ResponseResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    classification_confidence: float = 0.0
    confidence_level: str = ""
    grounded: bool = False
    no_information_found: bool = False
    query_type: QueryType = "factual"
    retrieval_hits: List[RetrievalHit] = field(default_factory=list)
    domain: Optional[str] = None
    status: str = ""
    request_id: str = ""
    retrieval: Optional[RetrievalResult] = None
    error: Optional[AgentError] = None

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "unavailable" if self.no_information_found else "answered"

    @property
    def intent(self) -> QueryType:
        """Backward-compatible alias for the query classification."""
        return self.query_type

    @property
    def clarification_question(self) -> Optional[str]:
        """Keep the existing API shape without inventing clarification text."""
        return self.answer if self.status == "clarification_needed" else None


@dataclass
class AgentError:
    agent: str
    code: str
    message: str


@dataclass
class ConversationTurn:
    session_id: str
    query: str
    answer: str
    turn_index: int
