# M1 Retrieval Evaluation Results

Metrics computed from live FAISS retrieval against `data/evaluation/queries.json`.
Hit@k = expected_document appears in the top-k filenames.
Unavailable-information queries have `expected_document=NONE`; ambiguous queries are not retrieval cases and are excluded from Hit@ metrics.

## Overall

- Total queries: **27**
- Scored (excl. unavailable): **15**
- Unavailable-information: **4**
- Unavailable queries returning no evidence: **3**
- Unavailable queries with a nearest-neighbor false positive: **1**
- Not applicable (ambiguous): **8**

| Metric | Excl. unavailable | Incl. unavailable |
|---|---|---|
| Hit@1 | 100.0% | 78.95% |
| Hit@3 | 100.0% | 78.95% |
| Hit@5 | 100.0% | 78.95% |

## Domain-wise (excl. unavailable)

| Domain | Queries | Scored | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|
| Software Engineering | 13 | 7 | 100.0% | 100.0% | 100.0% |
| Hospital Administration | 14 | 8 | 100.0% | 100.0% | 100.0% |

## Category-wise (excl. unavailable)

| Category | Queries | Scored | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|
| factual | 10 | 6 | 100.0 | 100.0 | 100.0 |
| procedural | 5 | 5 | 100.0 | 100.0 | 100.0 |
| comparative | 4 | 4 | 100.0 | 100.0 | 100.0 |
| ambiguous | 8 | 0 | n/a | n/a | n/a |

## Per-query results

| ID | Domain | Category | Expected | Top-1 | Rank | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|---|---|---|
| se_q01 | Software Engineering | factual | microservices_architecture.pdf | microservices_architecture.pdf | 1 | 1 | 1 | 1 |
| se_q02 | Software Engineering | procedural | git_workflow.txt | git_workflow.txt | 1 | 1 | 1 | 1 |
| se_q03 | Software Engineering | comparative | microservices_architecture.pdf | microservices_architecture.pdf | 1 | 1 | 1 | 1 |
| se_q04 | Software Engineering | factual | coding_standards.docx | coding_standards.docx | 1 | 1 | 1 | 1 |
| se_q05 | Software Engineering | procedural | coding_standards.docx | coding_standards.docx | 1 | 1 | 1 | 1 |
| se_q06 | Software Engineering | comparative | programming_languages.csv | programming_languages.csv | 1 | 1 | 1 | 1 |
| se_q07 | Software Engineering | factual | programming_languages.csv | programming_languages.csv | 1 | 1 | 1 | 1 |
| se_q08 | Software Engineering | factual | NONE | None | N/A | 0 | 0 | 0 |
| se_q09 | Software Engineering | factual | NONE | None | N/A | 0 | 0 | 0 |
| ha_q01 | Hospital Administration | factual | patient_admission_policy.pdf | patient_admission_policy.pdf | 1 | 1 | 1 | 1 |
| ha_q02 | Hospital Administration | factual | patient_admission_policy.pdf | patient_admission_policy.pdf | 1 | 1 | 1 | 1 |
| ha_q03 | Hospital Administration | procedural | nursing_procedures.docx | nursing_procedures.docx | 1 | 1 | 1 | 1 |
| ha_q04 | Hospital Administration | procedural | nursing_procedures.docx | nursing_procedures.docx | 1 | 1 | 1 | 1 |
| ha_q05 | Hospital Administration | comparative | emergency_protocols.txt | emergency_protocols.txt | 1 | 1 | 1 | 1 |
| ha_q06 | Hospital Administration | comparative | medication_dosage_reference.csv | medication_dosage_reference.csv | 1 | 1 | 1 | 1 |
| ha_q07 | Hospital Administration | factual | medication_dosage_reference.csv | medication_dosage_reference.csv | 1 | 1 | 1 | 1 |
| ha_q08 | Hospital Administration | procedural | emergency_protocols.txt | emergency_protocols.txt | 1 | 1 | 1 | 1 |
| ha_q09 | Hospital Administration | factual | NONE | patient_admission_policy.pdf | N/A | 0 | 0 | 0 |
| ha_q10 | Hospital Administration | factual | NONE | None | N/A | 0 | 0 | 0 |
| se_q10 | Software Engineering | ambiguous | NONE | None | N/A | N/A | N/A | N/A |
| se_q11 | Software Engineering | ambiguous | NONE | None | N/A | N/A | N/A | N/A |
| se_q12 | Software Engineering | ambiguous | NONE | None | N/A | N/A | N/A | N/A |
| se_q13 | Software Engineering | ambiguous | NONE | None | N/A | N/A | N/A | N/A |
| ha_q11 | Hospital Administration | ambiguous | NONE | None | N/A | N/A | N/A | N/A |
| ha_q12 | Hospital Administration | ambiguous | NONE | None | N/A | N/A | N/A | N/A |
| ha_q13 | Hospital Administration | ambiguous | NONE | None | N/A | N/A | N/A | N/A |
| ha_q14 | Hospital Administration | ambiguous | NONE | None | N/A | N/A | N/A | N/A |

## Failed queries

None.

## Low-relevance examples

- `ha_q09`: Top-1 `patient_admission_policy.pdf` L2=0.5337772755578493 (expected `NONE`) — What was Riverside General Hospital's total revenue last fiscal year?

## Inspected failed cases (up to 3)

No failures to inspect.
## Likely failure causes

- FAISS `IndexFlatL2` always returns a nearest neighbor, so unavailable-information queries cannot score a document hit.
- One chunk per document (short corpus) can mix unrelated sections into the same embedding, diluting comparative/factual signals.
- Characteristically similar hospital/software procedure language can cross-rank nearby documents.
- L2 distance is not calibrated as relevance; a high score can still be Top-1.

## M2 improvement ideas

- Add a similarity / confidence threshold and return `unavailable` when Top-1 is weak.
- Use smaller, section-aware chunks so sprint vs git vs coding-standard topics do not share one vector.
- Hybrid lexical + dense retrieval (BM25 + embeddings) for keyword-heavy factual queries.
- Cross-encoder re-ranking of Top-10 candidates.
- Explicit unavailable-intent routing (already in QueryUnderstandingAgent) before generation.
- Domain filter at retrieval time using query-understanding domain labels.
