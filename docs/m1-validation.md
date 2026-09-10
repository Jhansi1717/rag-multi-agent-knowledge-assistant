# M1 Validation

Evidence from code, tests, and a live evaluation run. Commands are those in the repository. Unfinished work is not marked implemented.

---

## Four file formats

| Format | Library | What is extracted | Provenance metadata |
|---|---|---|---|
| PDF | PyMuPDF | Page text joined with blank lines | `page_count`, `page_texts`, empty pages |
| DOCX | python-docx | Paragraphs and tables | `paragraph_count`, `table_count` |
| TXT | stdlib UTF-8 | Full file text | `line_count` |
| CSV | pandas | `Row n: col: value \| …` | `row_count`, `column_names` |

Validation: `python -m pytest test_m13_ingestion.py -v` (all four formats). `ingestion/validator.py` checks chunk ↔ document ids and format keys.

---

## Ingestion validation

Pipeline: `validate_file_path` → `extract_document` → `clean_document` → `TextChunker.chunk_document` → `validate_provenance` → `VectorStore.index_document`.

```bash
python generate_evaluation_corpus.py
python index_evaluation_corpus.py
```

`data/evaluation/ingestion_report.json` (last successful run):

| | Count |
|---|---|
| Files ingested | 8 / 8 PASS |
| Chunks | 8 |
| Vectors | 8 |

Domains: Software Engineering (`data/software_engineering/`) and Hospital Administration (`data/hospital_administration/`). Each domain has one PDF, DOCX, TXT, and CSV.

Cleaning (`ingestion/cleaner.py`): NFKC, line endings, collapsed whitespace; headings, numbering, and table `|` markup kept.

---

## Chunking

Defaults in `ingestion/chunker.py`: **650** tokens, **75** overlap, cap **800**, paragraph/page/row first.

Eval corpus documents are short, so **each file produced 1 chunk**. Tests in `test_m13_ingestion.py` cover multi-chunk paragraph splits on longer synthetic text.

`POST /upload` now uses `ingestion.pipeline.index_file`; it does not use a
Uploads use the full production pipeline with token-aware chunking.

---

## Embeddings

`EmbeddingProvider` / alias `EmbeddingService`: `all-MiniLM-L6-v2`, float32, dim 384. Empty strings raise `ValueError`. Covered by `test_embeddings.py`.

---

## Vector indexing

FAISS `IndexFlatL2` + JSON metadata. `index_document` replaces an existing `source_location`. Persist/load/search covered by `test_vector_store.py` and `test_m13_ingestion.py` (`test_no_duplicate_indexing_same_source`).

Eval artefacts: `data/evaluation/index.faiss`, `data/evaluation/metadata.json`.

---

## Retrieval

`SemanticRetriever.retrieve(query, top_k)` → `VectorStore.search` (query embed + FAISS + metadata).

Agent path: `RetrievalAgent` adds `rank`, `score`, `text`, `document_id`, `filename`, `metadata`. Orchestrator tests: `python -m pytest test_m14_retrieval.py -v`.

HTTP: `POST /retrieve` runs the M2 orchestrator and returns a structured response.

---

## Top-1 / Top-3 / Top-5 evaluation

```bash
python evaluation/evaluate_retrieval.py
```

Dataset: `data/evaluation/queries.json` (19 queries). Hit@k = `expected_document` appears in the top-k **filenames**. Unavailable queries (`expected_document=NONE`) are excluded from the primary averages.

**Live results** (also in `evaluation/results.md`):

| | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|
| 15 in-KB queries | 100.0% | 100.0% | 100.0% |
| 19 including unavailable | 78.95% | 78.95% | 78.95% |

| Domain | Scored | Hit@1/3/5 |
|---|---|---|
| Software Engineering | 7 | 100% |
| Hospital Administration | 8 | 100% |

| Category | Scored | Hit@1/3/5 |
|---|---|---|
| factual | 6 | 100% |
| procedural | 5 | 100% |
| comparative | 4 | 100% |
| unavailable-information | 0 scored | n/a |

Every in-KB query ranked the expected file at **rank 1**.

---

## Failures

No in-KB Hit@1 misses. Failures are the four **unavailable-information** queries (FAISS still returns a neighbor):

| ID | Query (summary) | Top-1 | L2 |
|---|---|---|---|
| se_q08 | TechCorp revenue | `medication_dosage_reference.csv` | 1.773 |
| se_q09 | Who invented Python / year | `coding_standards.docx` | 1.467 |
| ha_q09 | Hospital fiscal revenue | `patient_admission_policy.pdf` | 0.873 |
| ha_q10 | Nobel Prize in Medicine | `nursing_procedures.docx` | 1.895 |

Inspected: se_q08 Top-5 is unrelated meds/git/architecture text; se_q09 ranks coding standards then a Python **language** CSV row (no inventor); ha_q09 matches hospital name, not revenue.

---

## Limitations

- Nearest-neighbor search cannot express “not in the KB” without a threshold or intent gate (orchestrator intent gate is Python-only).
- One vector per eval file mixes sections; Hit@100% is easier on a tiny corpus than on production docs.
- Extractive answers can include headings; no LLM.
- API upload/retrieve path is not the validated pipeline.
- Web Speech API, UI, and OpenAI generation are **not implemented**.
