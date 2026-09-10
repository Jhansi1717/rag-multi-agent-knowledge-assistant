# M1 Research — RAG Multi-Agent Knowledge Assistant

This note records the concepts and technology choices used in the **implemented** Milestone 1 system. Items not in code are labelled **FUTURE**.

---

## RAG

Retrieval-Augmented Generation separates **knowledge storage** from **language generation**. Documents are ingested offline; at query time the system retrieves supporting chunks and (later) conditions an LLM on that evidence.

M1 implements ingestion and retrieval. M2 adds optional OpenAI-backed grounded
response generation; tests inject a mock client and do not require an API key.

Pipeline:

```text
Document → extract → clean → chunk → embed → FAISS
Query → understanding → embed → L2 search → ranked chunks → grounded response / clarification
```

---

## Embeddings

An embedding is a fixed-length vector that places semantically similar text nearby in space.

**Implemented:** `sentence-transformers` model `all-MiniLM-L6-v2`, 384 dimensions, CPU, via `EmbeddingProvider` in `vector_store/embeddings.py`. The same model encodes chunks and queries so distances are comparable.

Chosen because it runs locally (no API key), loads quickly, and is a standard retrieval baseline. Stronger or paid models (e.g. OpenAI embeddings, `all-mpnet-base-v2`) are deferred.

---

## Semantic similarity

Keyword search fails when query and document use different wording. Cosine or Euclidean distance on embeddings approximates meaning.

**Implemented:** FAISS `IndexFlatL2` (Euclidean). Lower `similarity_score` is closer. There is no calibrated 0–1 confidence in the vector store; the orchestrator derives a simple `1 / (1 + L2)` signal for clarification.

Limitation: L2 Top-1 can be a weak match. Unavailable questions still receive a neighbor.

---

## Chunking

Embeddings work on passages, not whole books. Chunk size trades context against noise.

**Implemented:** `ingestion/chunker.py` — tiktoken `cl100k_base`, default **650 tokens**, overlap **75**, max **800**. Prefers paragraph / PDF page / CSV row boundaries; oversized segments fall back to a token window.

`POST /upload` uses `ingestion.pipeline.index_file`, including token-aware
chunking, embeddings, and FAISS persistence.

---

## Vector search

FAISS stores float32 vectors in process memory and returns k nearest IDs. Chunk fields live in parallel JSON keyed by row ID.

**Implemented:** `VectorStore` (`add_chunks`, `index_document`, `save`, `load`, `search`). Duplicate `source_location` values are replaced rather than double-indexed. Persistence: `data/evaluation/index.faiss` + `metadata.json` (eval) or `data/index.faiss` (legacy samples).

Managed databases (Qdrant, Pinecone) are **FUTURE**.

---

## Multi-agent orchestration

M1 uses a **fixed sequence**, not autonomous planning and not LangGraph.

**Implemented:** `agents/orchestrator.py`

```text
Query → QueryUnderstandingAgent
     → RetrievalAgent (SemanticRetriever)
     → ResponseGenerationAgent
     ↳ ambiguous → structured clarification-needed response
```

Unavailable information is not a query-understanding category. It is determined
from retrieval evidence, and `POST /retrieve` uses the M2 orchestrator.

---

## Agent modules

| Agent | Module | Behaviour |
|---|---|---|
| Query Understanding | `query_understanding.py` | Normalize text; classify factual / procedural / comparative / ambiguous; deterministic confidence and routing |
| Retrieval | `retrieval_agent.py` | Semantic Top-K hits with L2 distance, derived relevance, threshold filtering, and metadata |
| Response Generation | `response_generation.py` | Optional grounded LLM response using retrieved context and metadata citations |
| Clarification | `clarification.py` | Legacy helper module; the M2 orchestrator returns its own structured clarification response |
| Conversation Memory | `memory.py` | Records turns per `session_id`; not a required resolution stage |

---

## Web Speech API STT/TTS

**FUTURE / designed only.** Browser `SpeechRecognition` and `SpeechSynthesis` would convert speech ↔ text at the client. The backend stays text (`/retrieve` or a future `/chat`). No STT/TTS code ships in M1.

---

## Technology choices

| Decision | Choice | Why (M1) |
|---|---|---|
| Language | Python 3.11 | Matches existing runtime |
| HTTP | FastAPI + Uvicorn | Small API, Pydantic, `/docs` |
| PDF | PyMuPDF | Local text + page metadata, no OCR |
| DOCX | python-docx | Paragraphs and tables |
| CSV | pandas | Row text with column labels |
| Tokens | tiktoken | Deterministic chunk sizes |
| Embeddings | MiniLM-L6-v2 | Local, fast baseline |
| Index | faiss-cpu IndexFlatL2 | Zero extra process; 8-chunk eval is instant |
| Agents | Plain Python classes | Deterministic; no framework |

Alternatives rejected for M1: cloud embeddings, Docker vector DBs, LangGraph, OCR, OpenAI generation.
