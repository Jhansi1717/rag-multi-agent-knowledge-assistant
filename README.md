# RAG Multi-Agent Knowledge Assistant

Local Retrieval-Augmented Generation (RAG) foundation: ingest documents, embed chunks, retrieve evidence with FAISS, and route queries through a deterministic agent layer. Milestone 1 validates retrieval quality before LLM generation or a UI.

---

## Objective

Build an end-to-end knowledge retrieval pipeline that returns sourced, ranked chunks for user queries, independent of exact keyword match.

---

## M1 scope

**In scope and implemented**

- PDF / DOCX / TXT / CSV ingestion, cleaning, paragraph-aware chunking
- `all-MiniLM-L6-v2` embeddings and FAISS `IndexFlatL2` indexing
- Semantic Top-k retrieval
- Deterministic agents (no LLM, no agent framework)
- Two-domain evaluation corpus and Hit@1/3/5 reporting
- FastAPI `GET /health`, `POST /upload`, `POST /retrieve`

**Out of scope for M1**

- LLM answer generation
- Web UI
- Web Speech API STT/TTS (documented only)
- Autonomous multi-agent reasoning

---

## Implemented features

| Feature | Where |
|---|---|
| Format extraction | `ingestion/pdf_extractor.py`, `docx_extractor.py`, `txt_extractor.py`, `csv_extractor.py` |
| Full ingest pipeline | `ingestion/pipeline.py` (`ingest_file` / `index_file`) |
| Cleaning | `ingestion/cleaner.py` |
| Chunking (650 tokens, 75 overlap, tiktoken) | `ingestion/chunker.py` |
| Embeddings | `vector_store/embeddings.py` (`EmbeddingProvider`) |
| Vector store + duplicate-source guard | `vector_store/store.py` |
| Semantic search | `retrieval/retriever.py` |
| Query understanding, retrieval, extractive response, clarification, recent-turn memory, orchestrator | `agents/` |
| HTTP retrieve/upload | `app.py` |

**Honest gap:** `POST /upload` still uses character `simple_chunk()` (500 chars) and does not call `ingestion/pipeline.py` or the orchestrator. The validated M1 path is the Python pipeline + evaluation index.

---

## Architecture

```text
Ingestion (IMPLEMENTED)
  file → validate → extract → clean → chunk → embed → FAISS + metadata.json

Query (IMPLEMENTED, Python)
  Query → QueryUnderstandingAgent → RetrievalAgent → SemanticRetriever → FAISS
       → ResponseGenerationAgent (extractive)
       ↳ ClarificationAgent if vague / low-confidence
       ↳ unavailable intent → "not available in the knowledge base"

HTTP (IMPLEMENTED, thinner)
  POST /upload → extract + simple_chunk → VectorStore
  POST /retrieve → SemanticRetriever (no agents)
```

Voice STT/TTS is a **future client boundary** (Web Speech API). The backend is text-only.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| API | FastAPI, Uvicorn, Pydantic |
| PDF / DOCX / CSV | PyMuPDF, python-docx, pandas |
| Chunk tokens | tiktoken (`cl100k_base`) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-d) |
| Vectors | faiss-cpu `IndexFlatL2` |
| Metadata | JSON (`data/evaluation/metadata.json` or `data/metadata.json`) |

LLM (`openai`) is listed in `requirements.txt` as commented future only.

---

## Repository structure

```text
rag-multi-agent-knowledge-assistant/
├── app.py
├── ingestion/          # extract, clean, chunk, pipeline
├── vector_store/       # EmbeddingProvider, VectorStore
├── retrieval/          # SemanticRetriever
├── agents/             # M1.4 agents + orchestrator
├── architecture/       # diagrams and schemas
├── docs/               # research, validation, decisions
├── evaluation/         # evaluate_retrieval.py, results.json, results.md
├── data/
│   ├── software_engineering/
│   ├── hospital_administration/
│   ├── evaluation/     # queries.json, index.faiss, metadata.json
│   ├── hr/  technical/ uploads/   # older sample / upload paths
├── generate_evaluation_corpus.py
├── index_evaluation_corpus.py
├── generate_samples.py
├── index_samples.py
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run (API)

```bash
uvicorn app:app --reload
```

- Health: `GET /health`
- Interactive docs: `http://localhost:8000/docs`

---

## Ingestion

**M1 evaluation corpus (Software Engineering + Hospital Administration):**

```bash
python generate_evaluation_corpus.py
python index_evaluation_corpus.py
```

Writes `data/evaluation/index.faiss` and `data/evaluation/metadata.json`. Report: `data/evaluation/ingestion_report.json`.

**Older HR / technical samples:**

```bash
python generate_samples.py
python index_samples.py
```

Writes `data/index.faiss` and `data/metadata.json`.

**Pipeline smoke-test** (extract → clean → chunk → provenance, no FAISS):

```bash
python run_pipeline.py
```

**HTTP upload** (character chunking, not the evaluation pipeline):

`POST /upload` with a multipart file (`.pdf`, `.docx`, `.txt`, `.csv`).

---

## Retrieval

**Python orchestrator** (agents + FAISS evaluation index): load `VectorStore(index_path="data/evaluation/index.faiss", meta_path="data/evaluation/metadata.json")`, then `Orchestrator(store).handle(query)`.

**HTTP:**

```json
POST /retrieve
{ "query": "How long is a typical agile sprint in the engineering guide?", "top_k": 3 }
```

Returns ranked chunks with `similarity_score` (L2 distance; lower is closer).

---

## Evaluation

```bash
python evaluation/evaluate_retrieval.py
```

Uses `data/evaluation/queries.json` and the evaluation FAISS index. Outputs `evaluation/results.json` and `evaluation/results.md`.

Legacy harness against `data/index.faiss`:

```bash
python evaluate_retrieval.py
```

**Tests actually used in this repo:**

```bash
python -m pytest test_m13_ingestion.py -v
python -m pytest test_m14_retrieval.py -v
python test_app.py
python test_vector_store.py
python test_retrieval.py
```

---

## Actual results

From a live run of `python evaluation/evaluate_retrieval.py` (19 queries; 15 in-KB, 4 unavailable):

| Metric | Excl. unavailable | Incl. unavailable |
|---|---|---|
| Hit@1 | 100.0% | 78.95% |
| Hit@3 | 100.0% | 78.95% |
| Hit@5 | 100.0% | 78.95% |

Evaluation index: **8 files, 8 chunks, 8 vectors**. Details: [`docs/m1-validation.md`](docs/m1-validation.md), [`evaluation/results.md`](evaluation/results.md).

---

## Limitations

- FAISS always returns a nearest neighbor; unavailable queries still get a Top-1 document.
- Extractive responses, not LLM answers.
- One chunk per short eval document can mix sections in a single vector.
- `app.py` upload/retrieve path is not the validated agent pipeline.
- No similarity threshold, re-ranking, or UI/voice.

---

## M2 roadmap

- Wire API to `ingestion/pipeline.py` and `Orchestrator`
- Similarity threshold and explicit unavailable responses
- LLM grounded generation with citations
- Section-aware chunking / hybrid BM25 + dense retrieval
- Web UI and browser Web Speech API STT/TTS (text I/O only on the server)

---

## Documentation

| File | Contents |
|---|---|
| [`docs/m1-research.md`](docs/m1-research.md) | RAG concepts and technology choices |
| [`docs/m1-validation.md`](docs/m1-validation.md) | Ingestion and retrieval validation |
| [`docs/data-models.md`](docs/data-models.md) | Schemas |
| [`docs/tech-stack.md`](docs/tech-stack.md) | Stack table |
| [`architecture/system-architecture.md`](architecture/system-architecture.md) | System diagrams |
