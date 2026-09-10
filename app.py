"""FastAPI surface for the M2 knowledge assistant pipeline."""

from __future__ import annotations

import os
import shutil
import uuid
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents.models import ResponseResult
from agents.orchestrator import Orchestrator
from ingestion.pipeline import index_file
from vector_store.store import VectorStore

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional in minimal installations
    pass


app = FastAPI(title="Knowledge Assistant API", version="2.0.0")
vector_store = VectorStore()
try:
    vector_store.load()
except FileNotFoundError:
    pass
orchestrator = Orchestrator(vector_store)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3
    session_id: Optional[str] = None


class ChatRequest(RetrieveRequest):
    session_id: Optional[str] = "default"


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    domain: str = ""
    source_location: str = ""
    chunk_count: int
    chunks_added: int
    reindexed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationResponse(BaseModel):
    chunk_id: str
    filename: str
    excerpt: str
    document_id: str = ""


class RetrievalHitResponse(BaseModel):
    rank: int
    relevance_score: float
    distance_score: float
    text: str
    document_id: str
    filename: str
    chunk_id: str


class RetrievalResponse(BaseModel):
    top_k: int
    filtered_count: int
    retrieval_confidence: float
    sufficient_evidence: bool
    no_relevant_information: bool
    results: list[RetrievalHitResponse]


class ErrorResponse(BaseModel):
    agent: str
    code: str


class RetrieveResponse(BaseModel):
    request_id: str
    query: str
    query_type: str
    classification_confidence: float
    status: str
    answer: str
    confidence: float
    confidence_level: str
    citations: list[CitationResponse]
    retrieval: Optional[RetrievalResponse] = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    no_information_found: bool
    clarification_needed: bool
    error: Optional[ErrorResponse] = None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: {extension}. Supported formats: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    document_id = str(uuid.uuid4())
    temporary_path = os.path.join(UPLOAD_DIR, f"{document_id}{extension}")
    try:
        with open(temporary_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use the production validation/extraction/cleaning/chunking/indexing path.
        result = index_file(temporary_path, vector_store)
        result["filename"] = filename
        result["file_type"] = extension
        result["metadata"] = {}
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc


def _validate_request(request: RetrieveRequest) -> None:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")


def _citation_payload(response: ResponseResult) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": citation.chunk_id,
            "filename": citation.filename,
            "excerpt": citation.excerpt,
            "document_id": citation.document_id,
        }
        for citation in response.citations
    ]


def _retrieval_payload(response: ResponseResult) -> Optional[dict[str, Any]]:
    retrieval = response.retrieval
    if retrieval is None:
        return None
    return {
        "top_k": retrieval.top_k,
        "filtered_count": retrieval.filtered_count,
        "retrieval_confidence": round(retrieval.retrieval_confidence, 4),
        "sufficient_evidence": retrieval.sufficient_evidence,
        "no_relevant_information": retrieval.no_relevant_information,
        "results": [
            {
                "rank": hit.rank,
                "relevance_score": round(hit.relevance_score, 4),
                "distance_score": round(hit.distance_score, 4),
                "text": hit.text,
                "document_id": hit.document_id,
                "filename": hit.filename,
                "chunk_id": hit.chunk_id,
            }
            for hit in retrieval.results
        ],
    }


def _response_payload(query: str, response: ResponseResult) -> dict[str, Any]:
    legacy_results = [
        {
            "text": hit.text,
            "document_name": hit.filename,
            "document_id": hit.document_id,
            "chunk_id": hit.chunk_id,
            "similarity_score": hit.distance_score,
            "relevance_score": hit.relevance_score,
        }
        for hit in response.retrieval_hits
    ]
    return {
        "request_id": response.request_id,
        "query": query,
        "query_type": response.query_type,
        "classification_confidence": round(response.classification_confidence, 4),
        "status": response.status,
        "answer": response.answer,
        "confidence": round(response.confidence, 4),
        "confidence_level": response.confidence_level,
        "citations": _citation_payload(response),
        "retrieval": _retrieval_payload(response),
        "results": legacy_results,
        "no_information_found": response.no_information_found,
        "clarification_needed": response.status == "clarification_needed",
        "error": (
            {"agent": response.error.agent, "code": response.error.code}
            if response.error
            else None
        ),
    }


def _run_orchestrator(request: RetrieveRequest) -> dict[str, Any]:
    _validate_request(request)
    try:
        response = orchestrator.handle(
            query=request.query,
            session_id=request.session_id or "default",
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="M2 pipeline failed.") from exc

    if response.error is not None:
        raise HTTPException(
            status_code=500,
            detail=f"{response.error.agent} stage failed",
        )
    return _response_payload(request.query, response)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_knowledge(request: RetrieveRequest) -> dict[str, Any]:
    """Run the complete M2 Understanding → Retrieval → Generation pipeline."""
    return _run_orchestrator(request)


@app.post("/chat", response_model=RetrieveResponse)
async def chat(request: ChatRequest) -> dict[str, Any]:
    """Backward-compatible alias for the M2 pipeline endpoint."""
    return _run_orchestrator(request)
