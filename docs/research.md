# Research Documentation — RAG Multi-Agent Knowledge Assistant

## 1. RAG Architecture

Retrieval-Augmented Generation (RAG) is a design pattern that enhances a language model
by giving it access to an external knowledge base at query time. Instead of relying on
information memorised during training, the model receives retrieved document chunks as
part of its prompt, grounding its response in verifiable source material.

The two primary stages are:
- **Knowledge Ingestion**: Documents are processed, chunked, embedded, and stored in a
  vector index before any user interaction.
- **Query Processing**: A user query is embedded, the vector index is searched for
  semantically similar chunks, and those chunks are passed to a generation model.

This system implements the full ingestion and retrieval stages for Milestone 1. The
generation stage is designed but not yet implemented.

---

## 2. Retrieval Pipeline

The retrieval pipeline converts raw documents into a searchable semantic index and then
retrieves the most relevant chunks for a given query.

```text
Input Documents (PDF / DOCX / TXT / CSV)
    ↓
File Validation
    ↓
Format-Specific Text Extraction
    ↓
Text Cleaning / Normalisation
    ↓
Token-Aware Chunking (600/80 tokens baseline)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
FAISS Vector Index  +  JSON Metadata Repository
    ↓
[At query time]
Query Embedding
    ↓
FAISS L2 Nearest-Neighbour Search
    ↓
Top-K Ranked Chunks with Scores and Source Metadata
```

Each component is a separate Python module so it can be replaced or tuned independently.

---

## 3. Semantic Similarity

Semantic similarity measures how close two pieces of text are in meaning rather than
in exact wording. This system uses **cosine similarity** (via FAISS L2 distance on
normalised vectors) to compare a query embedding against chunk embeddings. A query
such as "How do I request time off?" will score highly against a chunk containing
"Employees must submit leave requests through the HR portal", even though no keywords
are shared between the two sentences.

---

## 4. Embeddings

An embedding is a fixed-length dense numerical vector that encodes the semantic content
of a piece of text. The model `all-MiniLM-L6-v2` from Sentence Transformers produces
vectors of 384 dimensions. Texts with similar meanings produce vectors that are
geometrically close; dissimilar texts produce vectors that are far apart.

The same model is used for both chunk embeddings (at ingestion) and query embeddings
(at retrieval), ensuring that distances are meaningful.

---

## 5. Chunking

Large documents are split into smaller chunks so that:
- Each embedding is focused on a single concept rather than a whole document.
- Retrieved context fits within the context window of a future LLM.
- Precision is higher — a matched chunk points to a specific paragraph, not an entire file.

The Milestone 1 baseline uses token-aware chunking of approximately 600 tokens (80 token overlap).
This is a known starting point, not an optimised value. Chunking strategy is the single
parameter most likely to affect retrieval quality and will be revisited in future milestones.

---

## 6. Vector Search

FAISS (Facebook AI Similarity Search) maintains an in-memory index of all chunk
embeddings. When a query embedding is provided, FAISS computes L2 distances to every
indexed vector and returns the `k` nearest neighbours in sub-linear time. The L2
distance score is a raw floating-point number; smaller values indicate greater
similarity.

The FAISS index is serialised to disk (`data/index.faiss`) after ingestion so that
retrieval can run independently without re-embedding the corpus.

---

## 7. Multi-Agent Query Resolution

The long-term system is designed around specialised agents that each own a discrete
responsibility in the query resolution workflow:

| Agent | Responsibility | Status |
|---|---|---|
| Query Understanding Agent | Intent classification, entity extraction, domain detection | DESIGNED |
| Retrieval Agent | Embedding query, searching FAISS, returning ranked evidence | DESIGNED |
| Response Generation Agent | Synthesising grounded answers from retrieved chunks | DESIGNED |
| Clarification Agent | Detecting ambiguous queries and requesting more detail | DESIGNED |
| Conversation Memory Agent | Tracking multi-turn context and coreference | DESIGNED |

In Milestone 1, semantic retrieval is called directly from the API layer without agent
wrappers. Agent wrappers are fully designed and will be implemented in the next milestone.

---

## 8. Agent Orchestration

The planned orchestrator coordinates agent execution. The designed workflow is:

```text
User Query
    ↓
Query Understanding Agent
    ↓
Ambiguity Check
    ├── Ambiguous → Clarification Agent → User
    └── Clear
          ↓
       Retrieval Agent
          ↓
       Evidence Sufficiency Check
          ├── Insufficient → "Unavailable" response
          └── Sufficient
                ↓
          Response Generation Agent
                ↓
          Answer + Citations
```

This orchestrator logic is fully designed but not yet implemented in code. The
implementation decision on whether to use a framework such as LangGraph will be made
when the orchestrator is built.

---

## 9. Web Speech API

The Web Speech API is a browser-native interface that provides:
- **Speech-to-Text (STT)**: Converting spoken audio to a query string.
- **Text-to-Speech (TTS)**: Reading the system's response back to the user.

The planned voice interaction layer will use the Web Speech API without requiring a
third-party audio processing backend. This component is architected for a future
milestone and is not implemented in Milestone 1.

---

## 10. Technology Decisions

### Python 3.11
**What it does**: Core implementation language.
**Why selected**: Mature ecosystem, strong ML library support, fast and well-supported on
Windows and Linux.
**Alternatives considered**: JavaScript/TypeScript (weaker ML library ecosystem).
**Why sufficient for Milestone 1**: All required libraries (FastAPI, Sentence Transformers,
FAISS, pandas) are Python-native.

---

### FastAPI
**What it does**: Provides HTTP endpoints for document upload and semantic retrieval.
**Why selected**: High-performance async framework with automatic request validation via
Pydantic. Provides interactive API documentation at `/docs` out of the box.
**Alternatives considered**: Flask (synchronous by default, less structured), Django
(too heavyweight for a microservice API).
**Why sufficient for Milestone 1**: The upload and retrieve endpoints are the only
required surfaces; FastAPI handles both cleanly in under 100 lines.

---

### PyMuPDF (`fitz`)
**What it does**: Extracts plain text from PDF files page by page.
**Why selected**: Widely used, fast, handles complex PDF layouts, runs locally.
**Alternatives considered**: `pdfminer`, `pypdf`, AWS Textract (cloud OCR).
**Why sufficient for Milestone 1**: The sample PDFs are text-based; no OCR or image
extraction is needed at this stage.

---

### python-docx
**What it does**: Reads paragraph and heading text from DOCX files.
**Why selected**: The standard Python library for DOCX, well-maintained.
**Alternatives considered**: LibreOffice subprocess calls (fragile), `docx2txt`.
**Why sufficient for Milestone 1**: Sufficient for extracting structured text from
synthetic DOCX samples.

---

### pandas
**What it does**: Reads CSV files into DataFrames, then serialises them to a string
representation for embedding.
**Why selected**: Standard data-processing library; CSV handling is trivial.
**Alternatives considered**: `csv` module (no structured representation).
**Why sufficient for Milestone 1**: Converts tabular leave data into a readable text
form that can be embedded and retrieved.

---

### Sentence Transformers — all-MiniLM-L6-v2
**What it does**: Generates 384-dimensional dense embeddings for text chunks and queries.
Runs entirely on CPU.
**Why selected**: Lightweight (22M parameters), no API key required, produces reliable
semantic embeddings for English text, widely used as a retrieval baseline.
**Alternatives considered**: OpenAI `text-embedding-3-small` (API key and cost required),
Cohere embeddings (API dependency), larger local models (too slow for fast local testing).
**Why sufficient for Milestone 1**: Produces embeddings good enough to validate the
retrieval pipeline and establish a measurable baseline. The model is swappable without
changing any retrieval code.

---

### FAISS (faiss-cpu)
**What it does**: An in-memory vector similarity search library. Stores chunk embeddings
and returns nearest neighbours for a query vector using L2 distance.
**Why selected**: Runs locally, extremely fast, no separate database process needed,
trivial to serialise to disk.
**Alternatives considered**: Pinecone (cloud-only), Qdrant (Docker service required),
Weaviate (heavyweight), ChromaDB (additional dependencies).
**Why sufficient for Milestone 1**: FAISS handles the 8-chunk prototype index in
microseconds and validates the end-to-end retrieval path. The `VectorStore` interface
is designed to allow FAISS to be replaced without changing retrieval or ingestion code.

---

### JSON Metadata Repository
**What it does**: Stores chunk-level metadata (chunk_id, document_id, document_name,
source_location, text) alongside the FAISS index.
**Why selected**: Zero dependencies, human-readable, trivial to inspect and debug.
**Alternatives considered**: SQLite (heavier for prototype use), PostgreSQL (requires
running database service).
**Why sufficient for Milestone 1**: The corpus is small enough that a flat JSON file is
sufficient and avoids introducing a database dependency.

---

## 11. Architecture Decisions

1. **Separation of ingestion and retrieval**: Ingestion runs once per document upload;
   retrieval runs per query. This improves testability and reflects production architectures.

2. **Evidence-first design**: Retrieval is validated before any LLM is added. This prevents
   poor retrieval quality from being obscured by fluent language generation.

3. **Modular extraction**: All format-specific logic lives in `ingestion/extractor.py` behind
   a single `extract(file_path)` interface. Retrieval and the API layer have no knowledge
   of file formats.

4. **Stable chunk IDs**: Every chunk carries a `chunk_id` that links it to its parent
   `document_id` and `source_location`, enabling full traceability from a retrieved vector
   back to the original byte range.

5. **Persist-then-retrieve**: The FAISS index and JSON metadata are written to disk after
   every upload. Retrieval loads from disk. This decouples the upload and query lifecycles.

---

## 12. Limitations

- **No confidence threshold**: FAISS always returns the k nearest neighbours even when
  the query topic is absent from the knowledge base. A similarity distance cutoff or an
  LLM-based sufficiency check is needed to handle unavailable-information queries correctly.

- **Fixed-size chunking**: Splitting on character count ignores sentence and paragraph
  boundaries, potentially splitting coherent sentences across chunks. This reduces
  embedding quality for those chunks.

- **No text cleaning module**: The current `ingestion/cleaner.py` is a stub. Cleaning
  is applied ad-hoc in `run_pipeline.py` using `" ".join(text.split())`. A proper
  normalisation module (removing headers, footers, artefacts) would improve embedding
  quality.

- **Single-chunk PDF extraction artefact**: The sample PDF extracted 879 characters
  because the `fpdf` library embeds text in a way that PyMuPDF can read directly.
  Scanned PDFs would require OCR.

- **In-memory index**: FAISS loads the full index into RAM on every process start.
  This is unsuitable for large corpora.

---

## 13. Future Improvements

- **Semantic chunking**: Split on sentence or paragraph boundaries rather than fixed
  character count to improve embedding coherence.
- **LLM integration**: Add the Response Generation Agent to synthesise grounded answers
  from retrieved chunks using OpenAI or an equivalent model.
- **Similarity threshold**: Introduce a minimum score cutoff to detect when the corpus
  contains no relevant information for a query.
- **Production vector database**: Replace FAISS with a persistent, scalable vector
  database such as Qdrant or Pinecone when the corpus grows.
- **Multi-agent orchestration**: Implement the designed agent workflow using LangGraph
  or a lightweight custom router.
- **Citation generation**: Expose the source chunk, document name, and page reference
  in the final response to the user.
- **Analytics and gap detection**: Track which queries return low-scoring results to
  identify knowledge gaps in the uploaded corpus.
