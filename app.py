"""FastAPI surface for the M2 knowledge assistant pipeline."""

from __future__ import annotations

import os
import shutil
import uuid
import time
import json
import logging
import asyncio
import dataclasses
import re
from collections import defaultdict
from typing import Any, Optional

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
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

# --- Structured Logging Setup ---
class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "endpoint": getattr(record, "endpoint", None),
            "status": getattr(record, "status", None),
            "duration": getattr(record, "duration", None),
            "error_class": getattr(record, "error_class", None),
        }
        return json.dumps({k: v for k, v in log_obj.items() if v is not None})

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONLogFormatter())
logger.addHandler(handler)
# Remove default uvicorn handlers if needed, but we keep this simple for now.

# --- Rate Limiter Setup ---
class RateLimiter:
    def __init__(self, calls: int, window: int):
        self.calls = calls
        self.window = window
        self.records = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        self.records[client_ip] = [t for t in self.records[client_ip] if now - t < self.window]
        if len(self.records[client_ip]) >= self.calls:
            return False
        self.records[client_ip].append(now)
        return True

RATE_LIMIT_CALLS = int(os.getenv("RATE_LIMIT_CALLS", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
rate_limiter = RateLimiter(RATE_LIMIT_CALLS, RATE_LIMIT_WINDOW)

# --- App Initialization ---
app = FastAPI(title="Knowledge Assistant API", version="2.0.0")

# CORS Setup
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",")] if allowed_origins_str else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Middleware ---
@app.middleware("http")
async def runtime_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id

    # Rate limit auth-sensitive endpoints
    if request.url.path in ["/upload", "/retrieve", "/chat"]:
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests"}},
                headers={"X-Request-ID": req_id}
            )

    start_time = time.time()
    status_code = 500
    error_class = None
    try:
        response = await asyncio.wait_for(call_next(request), timeout=30.0)
        status_code = response.status_code
        response.headers["X-Request-ID"] = req_id
        return response
    except asyncio.TimeoutError:
        status_code = 504
        error_class = "TimeoutError"
        return JSONResponse(
            status_code=504,
            content={"error": {"code": "TIMEOUT", "message": "Request timed out"}},
            headers={"X-Request-ID": req_id}
        )
    except Exception as exc:
        status_code = 500
        error_class = exc.__class__.__name__
        logger.exception("Unhandled Exception", extra={"request_id": req_id})
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An internal error occurred"}},
            headers={"X-Request-ID": req_id}
        )
    finally:
        duration = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "request_id": req_id,
                "endpoint": request.url.path,
                "status": status_code,
                "duration": round(duration, 4),
                "error_class": error_class
            }
        )

# --- Exception Handlers ---
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "message": "Invalid request parameters", "details": exc.errors()}}
    )

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
    query: str = Field(..., max_length=4000)
    top_k: int = 3
    session_id: Optional[str] = Field(default=None, max_length=128)


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=4000)
    top_k: int = 3
    session_id: Optional[str] = Field(default=None, max_length=128)
    clarification_id: Optional[str] = Field(default=None, max_length=128)


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
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class DocumentMetadataResponse(BaseModel):
    document_id: str
    document_name: str
    file_type: str = ""
    domain: str = ""
    chunk_count: int
    status: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentMetadataResponse]


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
    source_location: str = ""
    domain: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    message: Optional[str] = None


class StageTraceResponse(BaseModel):
    stage_name: str
    status: str
    duration_ms: Optional[float] = None


class EvidenceItemResponse(BaseModel):
    citation_id: str
    document_name: str
    chunk_id: str
    document_id: str = ""
    chunk_text: str
    similarity_score: float
    page: Optional[int] = None
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    row_number: Optional[int] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None


class ClarificationStateResponse(BaseModel):
    original_query: str = ""
    needed: bool = False
    clarification_id: Optional[str] = None
    question: Optional[str] = None
    clarification_answer: Optional[str] = None
    options: list[str] = Field(default_factory=list)
    ambiguity_type: Optional[str] = None
    missing_information: list[str] = Field(default_factory=list)
    refined_query: Optional[str] = None
    resolved: bool = False


class MemoryContextResponse(BaseModel):
    session_id: str = ""
    turn_index: int = 0
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    relevant_context: str = ""
    used_context: bool = False
    referenced_documents: list[str] = Field(default_factory=list)
    active_topic: Optional[str] = None
    clarification_state: Optional[ClarificationStateResponse] = None


class TransparencyReportResponse(BaseModel):
    query_type: str = ""
    classification_reason: str = ""
    routing: str = ""
    retrieved_count: int = 0
    filtered_count: int = 0
    retrieval_confidence: float = 0.0
    grounded: bool = False
    stages: list[StageTraceResponse] = Field(default_factory=list)
    source_evidence: list[EvidenceItemResponse] = Field(default_factory=list)
    citations: list[CitationResponse] = Field(default_factory=list)


class M3ResponseExtension(BaseModel):
    clarification: Optional[ClarificationStateResponse] = None
    memory: Optional[MemoryContextResponse] = None
    transparency: Optional[TransparencyReportResponse] = None


class RetrieveResponse(BaseModel):
    request_id: str
    query: str
    resolved_query: Optional[str] = None
    session_id: str = "default"
    query_type: str
    intent: str = "factual"
    domain: Optional[str] = None
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
    clarification_question: Optional[str] = None
    error: Optional[ErrorResponse] = None
    m3: Optional[M3ResponseExtension] = None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/ready")
def ready_check() -> dict[str, Any]:
    index = vector_store.index
    vector_count = getattr(index, "ntotal", 0) if index is not None else 0
    metadata_count = len(vector_store.metadata)
    row_ids_match = all(str(row_id) in vector_store.metadata for row_id in range(vector_count))
    db_ok = vector_count > 0 and vector_count == metadata_count and row_ids_match

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    dependencies = {
        "vector_store": db_ok,
        "llm_provider": provider,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }
    if not db_ok:
        raise HTTPException(status_code=503, detail="Database is unavailable")
    return {"status": "ready", "dependencies": dependencies}


@app.get("/documents", response_model=DocumentsResponse)
def list_documents() -> dict[str, Any]:
    """Return indexed document metadata derived from stored chunk records."""
    grouped: dict[str, dict[str, Any]] = {}
    for record in vector_store.metadata.values():
        document_id = record.get("document_id")
        if not document_id:
            continue
        metadata = record.get("metadata") or {}
        document = grouped.setdefault(document_id, {
            "document_id": document_id,
            "document_name": (
                record.get("original_filename")
                or metadata.get("original_filename")
                or metadata.get("filename")
                or record.get("document_name")
                or record.get("filename", "")
            ),
            "file_type": record.get("file_type") or metadata.get("file_type", ""),
            "domain": record.get("domain") or metadata.get("domain", ""),
            "chunk_count": 0,
            "status": "indexed",
        })
        document["chunk_count"] += 1
    return {"documents": list(grouped.values())}


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
    indexed = False
    try:
        with open(temporary_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use the production validation/extraction/cleaning/chunking/indexing path.
        # Pass the original filename so it's stored in metadata instead of the UUID filename.
        result = index_file(
            temporary_path,
            vector_store,
            original_filename=filename,
            source_location=f"upload://{filename.casefold()}",
        )
        indexed = True
        result["filename"] = filename
        result["file_type"] = extension
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc
    finally:
        if not indexed and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                logger.warning("Failed to remove temporary upload after ingestion error.")


def _validate_request(request: RetrieveRequest) -> None:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    for field_name in ("session_id", "clarification_id"):
        value = getattr(request, field_name, None)
        if value is not None:
            _validate_identifier(value, field_name)


def _validate_identifier(value: str, field_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise HTTPException(status_code=400, detail=f"{field_name} has an invalid format")


def _citation_payload(response: ResponseResult) -> list[dict[str, Any]]:
    transparency = response.m3.transparency if response.m3 else None
    citations = (
        transparency.citations
        if transparency and transparency.citations
        else list(response.citations or [])
    )
    return [
        {
            "chunk_id": citation.chunk_id,
            "filename": citation.filename,
            "excerpt": citation.excerpt,
            "document_id": citation.document_id,
        }
        for citation in citations
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
                "source_location": hit.metadata.get("source_location", ""),
                "domain": hit.metadata.get("domain", ""),
                "metadata": hit.metadata,
            }
            for hit in retrieval.results
        ],
    }


def _response_payload(
    query: str,
    session_id: str,
    response: ResponseResult,
) -> dict[str, Any]:
    legacy_results = [
        {
            "text": hit.text,
            "document_name": hit.filename,
            "document_id": hit.document_id,
            "chunk_id": hit.chunk_id,
            "similarity_score": hit.distance_score,
            "relevance_score": hit.relevance_score,
            "source_location": hit.metadata.get("source_location", ""),
            "domain": hit.metadata.get("domain", ""),
        }
        for hit in response.retrieval_hits
    ]
    return {
        "request_id": response.request_id,
        "query": query,
        "resolved_query": response.retrieval.query if response.retrieval else None,
        "session_id": session_id,
        "query_type": response.query_type,
        "intent": response.query_type,
        "domain": response.domain,
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
        "clarification_question": response.clarification_question,
        "error": (
            {
                "agent": response.error.agent,
                "code": response.error.code,
                "message": response.error.message,
            }
            if response.error
            else None
        ),
        "m3": dataclasses.asdict(response.m3) if response.m3 else None,
    }


def _run_orchestrator(request: Any, req_id: str) -> dict[str, Any]:
    _validate_request(request)
    session_id = request.session_id or uuid.uuid4().hex
    clarification_id = getattr(request, "clarification_id", None)
    try:
        response = orchestrator.handle(
            query=request.query,
            session_id=session_id,
            top_k=request.top_k,
            clarification_id=clarification_id,
        )
        response.request_id = req_id  # override with middleware generated one
    except Exception as exc:
        logger.error(
            "M3 pipeline failed",
            extra={
                "request_id": req_id,
                "error_class": type(exc).__name__,
            },
        )
        raise HTTPException(status_code=500, detail="M3 pipeline failed.") from exc

    payload = _response_payload(
        request.query,
        session_id,
        response,
    )
    if response.error is not None:
        return JSONResponse(status_code=500, content=payload)
    return payload


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_knowledge(request: RetrieveRequest, http_request: Request) -> dict[str, Any]:
    """Run the complete M2 Understanding → Retrieval → Generation pipeline."""
    req_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    return _run_orchestrator(request, req_id)


@app.post("/chat", response_model=RetrieveResponse)
async def chat(request: ChatRequest, http_request: Request) -> dict[str, Any]:
    """Run the M3 Understanding → Clarification → Retrieval → Generation pipeline."""
    req_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    return _run_orchestrator(request, req_id)


@app.get("/sessions/{session_id}/history", response_model=dict[str, Any])
def get_session_history(session_id: str) -> dict[str, Any]:
    """Retrieve bounded conversation history."""
    _validate_identifier(session_id, "session_id")
    try:
        turns = orchestrator.memory.get_session(session_id)
        return {"session_id": session_id, "history": [dataclasses.asdict(t) for t in turns]}
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Clear conversation memory for a session."""
    _validate_identifier(session_id, "session_id")
    try:
        orchestrator.memory.clear_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")


if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
