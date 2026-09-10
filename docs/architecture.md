# Architecture Decisions — RAG Multi-Agent Knowledge Assistant

This document records the key architecture decisions made for Milestone 1, including
what was chosen, what was considered, and why. Every decision maps to implemented code.

---

## 1. Thin Vertical Slice

**Decision**: Implement one complete end-to-end path (ingestion → retrieval) rather
than partial stubs across the full system.

The Milestone 1 path:
```
Document → Extraction → Cleaning → Chunking
         → Embedding → Indexing
         → Query → Retrieval → Top-K Evidence
```

**Rationale**: A thin vertical slice produces demonstrable, testable engineering
evidence within the milestone constraint. It also validates the most foundational
assumption: that semantic retrieval works on the chosen document corpus.

---

## 2. Separation of Ingestion and Retrieval

**Decision**: Ingestion runs once at upload time and persists its outputs. Retrieval
loads from disk and runs independently per query.

**Implementation**:
- `POST /upload` triggers extraction → chunking → embedding → `VectorStore.save()`
- `POST /retrieve` calls `VectorStore.load()` at startup, then `SemanticRetriever.retrieve()`

**Rationale**: Decoupling these lifecycles means the corpus can be built incrementally,
retrieval can run without re-embedding, and each path can be tested in isolation.

---

## 3. Unified Extraction Interface

**Decision**: All format-specific logic is encapsulated in `DocumentExtractor.extract()`
in `ingestion/extractor.py`. The API layer and the vector store have no knowledge of
file formats.

**Technology Choices**:

| Format | Library | Why selected | Alternatives considered |
|---|---|---|---|
| PDF | PyMuPDF (`fitz`) | Fast, local, handles complex layouts | pdfminer, pypdf, AWS Textract |
| DOCX | python-docx | Standard library for DOCX, well-maintained | LibreOffice subprocess, docx2txt |
| TXT | Python stdlib `open()` | No dependency needed | — |
| CSV | pandas `read_csv()` | Structured read, `to_string()` produces embeddable text | csv module |

**Why sufficient for Milestone 1**: The sample corpus is synthetic text-based documents.
No OCR, image extraction, or HTML parsing is required.

---

## 4. Baseline Fixed-Size Chunking

**Decision**: Split extracted text into token-aware chunks of approximately 600 tokens with 80 tokens overlap (baseline experiment)
using `simple_chunk()` in `app.py`.

**Rationale**: Fixed-size chunking is the simplest reproducible baseline. It introduces
no ambiguity in the chunking logic and makes it easy to change chunk size and measure
the retrieval impact in a future milestone.

**Known limitation**: Character-boundary splitting ignores sentence structure. A sentence
may be split across two chunks, reducing embedding coherence for those segments.

**Planned improvement**: Semantic or sentence-boundary chunking in a future milestone.

---

## 5. Local Embedding Model

**Decision**: Use `sentence-transformers/all-MiniLM-L6-v2` running on CPU via the
`sentence-transformers` library.

**What it does**: Produces 384-dimensional dense embeddings for chunks and queries.
The same model instance is used for both so that distances are comparable.

**Why selected**:
- Runs entirely locally — no API key, no network dependency, no cost
- 22M parameters — loads quickly even on CPU
- Produces reliable English-language semantic embeddings
- Widely used as a retrieval baseline in the research community

**Alternatives considered**:
- OpenAI `text-embedding-3-small`: Higher quality but requires an API key and per-token cost
- Cohere embeddings: API dependency
- `all-mpnet-base-v2`: Better quality but 4× slower and 3× larger

**Why sufficient for Milestone 1**: Validates the retrieval pipeline and establishes a
measurable, reproducible baseline. The embedding model is swappable by changing one
constructor parameter in `VectorStore.__init__()` without touching any other code.

---

## 6. FAISS for Vector Search

**Decision**: Use `faiss-cpu` (`IndexFlatL2`) as the Milestone 1 vector index.

**What it does**: Stores all chunk embeddings in memory and returns the k nearest
neighbours for a query embedding using L2 (Euclidean) distance.

**Why selected**:
- No separate database process — runs inside the Python process
- Trivially serialised to disk with `faiss.write_index()` / `faiss.read_index()`
- Fast enough for prototype-scale corpora
- Zero infrastructure overhead

**Alternatives considered**:
- Pinecone: Cloud-only, requires API key and account
- Qdrant: Requires a Docker container or separate service
- Weaviate: Heavy infrastructure, not appropriate for local testing
- ChromaDB: Additional Python dependencies and a local service

**Why sufficient for Milestone 1**: The evaluation corpus has 8 chunks. FAISS handles
this in microseconds. The `VectorStore` class exposes `add_chunks()`, `save()`,
`load()`, and `search()` — FAISS can be replaced with any backend without changing
calling code.

---

## 7. Parallel JSON Metadata Repository

**Decision**: Store chunk metadata in a flat JSON file (`data/metadata.json`) alongside
the FAISS index, keyed by the FAISS integer row ID.

**What it does**: Enables the system to reconstruct full Retrieval Result objects from
a FAISS search, including `chunk_id`, `document_name`, `source_location`, and `text`.

**Why selected**: Zero dependencies, human-readable, suitable for prototype scale.

**Alternatives considered**: SQLite (adds a dependency), PostgreSQL (requires a running
service).

**Why sufficient for Milestone 1**: A flat JSON file is appropriate when the corpus fits
in memory and concurrent writes are not required. Migration to a database-backed metadata
layer is straightforward when scale justifies it.

---

## 8. FastAPI for the HTTP Layer

**Decision**: Use FastAPI to expose `GET /health`, `POST /upload`, and `POST /retrieve`.

**Why selected**: Native async support, automatic Pydantic request validation, interactive
API documentation at `/docs`, minimal boilerplate.

**Alternatives considered**: Flask (synchronous by default), Django (heavyweight for
a single-service API).

**Why sufficient for Milestone 1**: Three endpoints are all that is required. FastAPI
handles all of them in under 100 lines, and the `RetrieveRequest` Pydantic model
provides immediate input validation.

---

## 9. Retrieval Validated Before Generation

**Decision**: Milestone 1 does not include an LLM. The system proves that retrieval
returns correct evidence before adding generation.

**Rationale**: Combining retrieval and generation in one step makes it impossible to
distinguish a retrieval failure from a generation failure. Measuring retrieval quality
independently gives a clean, attributable baseline.

---

## 10. Honest Implementation Boundaries

Every component in this project uses exactly one of the following status labels:

**IMPLEMENTED** — Working code, executed and validated in this milestone.  
**DESIGNED** — Architecture defined, interfaces described, no runtime code.  
**FUTURE** — Intentionally deferred; not designed in detail yet.

No component is labelled IMPLEMENTED unless it has been run and produced verifiable output.