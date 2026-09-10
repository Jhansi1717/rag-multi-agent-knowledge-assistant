# RAG Multi-Agent Knowledge Assistant

Local Retrieval-Augmented Generation (RAG) assistant: ingest documents, embed chunks, retrieve evidence with FAISS, and route queries through a fixed sequential M2 pipeline.

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
| Query understanding, retrieval, grounded response generation, clarification, recent-turn memory, orchestrator | `agents/` |
| HTTP retrieve/upload/chat | `app.py` |

The M2 API uses the production ingestion pipeline for uploads and the orchestrator for retrieval.

---

## Milestone 2

### M2.1 Query Understanding Agent

`QueryUnderstandingAgent` is deterministic and rule-based. It returns:
`query`, `normalized_query`, `query_type`, `classification_confidence`,
`routing`, `domain`, and `reason`.

The only query categories are:

- `factual`: direct facts, definitions, properties, or values
- `procedural`: how-to, steps, workflows, protocols, or processes
- `comparative`: compare, versus, difference, contrast, or similarities
- `ambiguous`: too short, vague, or containing an unresolved referent

Unavailable information is **not** a query category. Such queries are classified
as factual and availability is decided from retrieval evidence. Classification
confidence is a deterministic application-level signal, not a calibrated
probability. Factual, procedural, and comparative queries route to `RETRIEVAL`;
ambiguous queries route to `CLARIFICATION`.

### M2.2 Retrieval Agent

`RetrievalAgent` reuses `SemanticRetriever` and FAISS. It requests configurable
Top-K results, ranks by FAISS L2 distance (lower is better), preserves
`distance_score`, and derives bounded relevance as `1 / (1 + distance_score)`.
Configurable `RETRIEVAL_TOP_K` and `RETRIEVAL_MIN_RELEVANCE` control retrieval.
Hits preserve document, chunk, filename, text, and metadata fields. Empty stores,
empty results, invalid Top-K, domain filtering, and all-filtered results produce
structured no-evidence results without unrelated-domain fallback.

### M2.3 Response Generation Agent

`ResponseGenerationAgent` uses the configured OpenAI client when available. Its
grounding prompt requires context-only answering, no outside knowledge, explicit
insufficient-evidence handling, query-type-specific style, and citations only
from retrieved metadata. Confidence levels are application-level:
`HIGH` (>= 0.75), `MEDIUM` (0.45-<0.75), and `LOW` (<0.45); they are not calibrated
probabilities. Missing or low evidence returns a controlled no-information
response rather than an unsupported answer.

### M2.4 Orchestration

```text
User Query
    |
    v
Query Understanding
    |
    v
Retrieval
    |
    v
Response Generation
    |
    v
Final Response
```

Ambiguous queries use:

```text
Ambiguous
    |
    v
Clarification route
    |
    v
Milestone 3 clarification expansion
```

The current clarification route returns a structured clarification-needed
response; it does not implement multi-turn M3 clarification behavior. The
orchestrator uses structured handoffs and a request ID, while memory records
turns but is not an additional resolution stage.

## Architecture

```text
Ingestion (IMPLEMENTED)
  file → validate → extract → clean → chunk → embed → FAISS + metadata.json

Query (IMPLEMENTED, M1 + M2)
  Query → QueryUnderstandingAgent → RetrievalAgent → SemanticRetriever → FAISS
       → ResponseGenerationAgent (grounded LLM when configured)
       ↳ ambiguous → structured clarification-needed response

HTTP (IMPLEMENTED)
  POST /upload → ingestion/pipeline.py → embeddings → FAISS + metadata
  POST /retrieve → full M2 orchestrator response
  POST /chat → compatibility alias for the same M2 pipeline
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

LLM generation uses the optional OpenAI `gpt-4o-mini` model when `OPENAI_API_KEY` is configured; otherwise the response agent returns a controlled no-information response.

### M2 response confidence policy

Response confidence is an application-level value inherited from retrieval, not a
calibrated probability. `HIGH` is `>= 0.75`, `MEDIUM` is `0.45` to `< 0.75`,
and `LOW` is `< 0.45`. Low-confidence or missing evidence produces the controlled
no-information response rather than an unsupported answer. Grounded answers are
generated only from the retrieved context supplied to the LLM.

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
- Upload: `POST /upload` multipart file (`.pdf`, `.docx`, `.txt`, `.csv`)
- Retrieve: `POST /retrieve` with `{"query": "...", "session_id": "optional", "top_k": 3}`
- Chat: `POST /chat` is a compatibility alias for `/retrieve`
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

**HTTP upload** uses the full production pipeline:

`POST /upload` with a multipart file (`.pdf`, `.docx`, `.txt`, `.csv`).

---

## Retrieval

**Python orchestrator** (agents + FAISS evaluation index): load `VectorStore(index_path="data/evaluation/index.faiss", meta_path="data/evaluation/metadata.json")`, then `Orchestrator(store).handle(query)`.

**HTTP:**

```json
POST /retrieve
{ "query": "How long is a typical agile sprint in the engineering guide?", "top_k": 3 }
```

Returns the final M2 response with query type, confidence, citations, retrieval
summary/results, no-information status, clarification status, and request ID.

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

## Actual M2 results

The recorded M1 baseline for `python evaluation/evaluate_retrieval.py`
(19 queries; 15 in-KB, 4 unavailable) was:

| Metric | Excl. unavailable | Incl. unavailable |
|---|---|---|
| Hit@1 | 100.0% | 78.95% |
| Hit@3 | 100.0% | 78.95% |
| Hit@5 | 100.0% | 78.95% |

The final M2 evaluation used 21 queries: 19 corpus queries plus two explicit
ambiguous cases. Results were generated in mock context-echo mode because no
OpenAI key was configured; no automated LLM factual-accuracy claim is made.

| Metric | Result |
|---|---:|
| Classification accuracy | 100.0% |
| Retrieval success | 5.9% |
| Grounded-response rate | 0.0% |
| Citation coverage | 0.0% |
| Ambiguous detection rate | 100.0% |
| No-evidence handling rate | 100.0% |
| End-to-end completion rate | 100.0% |

Details: [`evaluation/m2_end_to_end_results.md`](evaluation/m2_end_to_end_results.md).

---

## Limitations

- FAISS returns nearest neighbors, so the M2 relevance/evidence policy is required
  to reject weak or unrelated results.
- One chunk per short eval document can mix sections in a single vector.
- Evaluation metadata must contain domain values for domain-filtered retrieval.
- No UI or voice client is included.

---

## Future milestones

### M3 and later

- Multi-turn clarification and conversational disambiguation
- Hybrid BM25 + dense retrieval and cross-encoder reranking
- Web UI and browser Web Speech API STT/TTS
- Managed vector databases and production deployment concerns

---

## Documentation

| File | Contents |
|---|---|
| [`docs/m1-research.md`](docs/m1-research.md) | RAG concepts and technology choices |
| [`docs/m1-validation.md`](docs/m1-validation.md) | Ingestion and retrieval validation |
| [`docs/data-models.md`](docs/data-models.md) | Schemas |
| [`docs/tech-stack.md`](docs/tech-stack.md) | Stack table |
| [`architecture/system-architecture.md`](architecture/system-architecture.md) | System diagrams |
