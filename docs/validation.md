# Retrieval Validation Report — Milestone 1

Live results from `python evaluation/evaluate_retrieval.py`.
Corpus: `data/evaluation/` — 8 documents across 2 domains, 19 labelled queries.

---

## Summary

| Metric | Excl. unavailable (15 queries) | All 19 queries |
|---|---|---|
| **Hit@1** | **100.0%** | 78.95% |
| **Hit@3** | **100.0%** | 78.95% |
| **Hit@5** | **100.0%** | 78.95% |

> The 4 unavailable-information queries are excluded from primary Hit@ averages because FAISS always returns a nearest neighbour — there is no "correct" document to rank when the answer isn't in the knowledge base.

---

## Results by Domain

| Domain | Scored queries | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|
| Software Engineering | 7 | **100%** | **100%** | **100%** |
| Hospital Administration | 8 | **100%** | **100%** | **100%** |

---

## Results by Query Type

| Query Type | Scored queries | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|
| Factual | 6 | **100%** | **100%** | **100%** |
| Procedural | 5 | **100%** | **100%** | **100%** |
| Comparative | 4 | **100%** | **100%** | **100%** |
| Unavailable-information | 4 | — | — | — |

---

## Detailed Results

| ID | Domain | Type | Query (summary) | Expected Doc | Top-1 Doc | Rank | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|---|---|---|---|
| se_q01 | Software Eng | Factual | What is the Git feature branch workflow? | git_workflow.txt | git_workflow.txt | 1 | ✅ | ✅ | ✅ |
| se_q02 | Software Eng | Factual | What coding standard applies to Python files? | coding_standards.docx | coding_standards.docx | 1 | ✅ | ✅ | ✅ |
| se_q03 | Software Eng | Procedural | How do you deploy a microservice? | microservices_architecture.pdf | microservices_architecture.pdf | 1 | ✅ | ✅ | ✅ |
| se_q04 | Software Eng | Procedural | What are the steps for a code review? | coding_standards.docx | coding_standards.docx | 1 | ✅ | ✅ | ✅ |
| se_q05 | Software Eng | Comparative | How do REST and SOAP differ? | microservices_architecture.pdf | microservices_architecture.pdf | 1 | ✅ | ✅ | ✅ |
| se_q06 | Software Eng | Comparative | Compare compiled vs interpreted languages | programming_languages.csv | programming_languages.csv | 1 | ✅ | ✅ | ✅ |
| se_q07 | Software Eng | Comparative | What is the difference between agile and waterfall? | git_workflow.txt | git_workflow.txt | 1 | ✅ | ✅ | ✅ |
| se_q08 | Software Eng | Unavailable | What is TechCorp's annual revenue? | NONE | medication_dosage_reference.csv | — | ❌ | ❌ | ❌ |
| se_q09 | Software Eng | Unavailable | Who invented Python and in what year? | NONE | coding_standards.docx | — | ❌ | ❌ | ❌ |
| ha_q01 | Hospital Admin | Factual | What are the visiting hours for ICU patients? | emergency_protocols.txt | emergency_protocols.txt | 1 | ✅ | ✅ | ✅ |
| ha_q02 | Hospital Admin | Factual | What is the maximum paracetamol dose for adults? | medication_dosage_reference.csv | medication_dosage_reference.csv | 1 | ✅ | ✅ | ✅ |
| ha_q03 | Hospital Admin | Factual | How long is the patient admission process? | patient_admission_policy.pdf | patient_admission_policy.pdf | 1 | ✅ | ✅ | ✅ |
| ha_q04 | Hospital Admin | Procedural | What is the hand hygiene protocol? | nursing_procedures.docx | nursing_procedures.docx | 1 | ✅ | ✅ | ✅ |
| ha_q05 | Hospital Admin | Procedural | How is a patient transferred between wards? | nursing_procedures.docx | nursing_procedures.docx | 1 | ✅ | ✅ | ✅ |
| ha_q06 | Hospital Admin | Procedural | What are the steps in an emergency code blue? | emergency_protocols.txt | emergency_protocols.txt | 1 | ✅ | ✅ | ✅ |
| ha_q07 | Hospital Admin | Comparative | Compare medication dosage for children vs adults | medication_dosage_reference.csv | medication_dosage_reference.csv | 1 | ✅ | ✅ | ✅ |
| ha_q08 | Hospital Admin | Comparative | How does ICU admission differ from general ward? | patient_admission_policy.pdf | patient_admission_policy.pdf | 1 | ✅ | ✅ | ✅ |
| ha_q09 | Hospital Admin | Unavailable | What is the hospital's fiscal year revenue? | NONE | patient_admission_policy.pdf | — | ❌ | ❌ | ❌ |
| ha_q10 | Hospital Admin | Unavailable | Who won the Nobel Prize in Medicine this year? | NONE | nursing_procedures.docx | — | ❌ | ❌ | ❌ |

---

## Failure Analysis — Unavailable-Information Queries

FAISS `IndexFlatL2` always returns the closest vector — it cannot express "not in the knowledge base". These 4 queries have no correct document, so every rank is a miss.

| ID | Query | Top-1 Returned | Top-1 L2 Distance | Why it matched |
|---|---|---|---|---|
| se_q08 | TechCorp annual revenue | medication_dosage_reference.csv | 1.77 | Closest neighbour by chance; unrelated content |
| se_q09 | Who invented Python / year | coding_standards.docx | 1.47 | Python keyword overlap with coding doc |
| ha_q09 | Hospital fiscal revenue | patient_admission_policy.pdf | 0.87 | Hospital keyword overlap — highest false confidence |
| ha_q10 | Nobel Prize in Medicine | nursing_procedures.docx | 1.90 | Medicine keyword pulled nursing doc |

**M2 behavior:** Retrieval preserves L2 distance, derives relevance as
`1/(1+distance)`, applies configurable filtering, and returns structured
no-evidence state when evidence is insufficient.

---

## Corpus Details

| Domain | Format | Document | Chunks |
|---|---|---|---|
| Software Engineering | PDF | microservices_architecture.pdf | 1 |
| Software Engineering | DOCX | coding_standards.docx | 1 |
| Software Engineering | TXT | git_workflow.txt | 1 |
| Software Engineering | CSV | programming_languages.csv | 1 |
| Hospital Administration | PDF | patient_admission_policy.pdf | 1 |
| Hospital Administration | DOCX | nursing_procedures.docx | 1 |
| Hospital Administration | TXT | emergency_protocols.txt | 1 |
| Hospital Administration | CSV | medication_dosage_reference.csv | 1 |

Total: 8 documents · 8 chunks · 8 vectors in FAISS

> Each synthetic document is short enough to fit in a single chunk at 650 tokens / 75 overlap.
> Real production documents will span multiple chunks — Hit@ on a larger corpus will be harder to achieve.

---

## How to Reproduce

```bash
# 1. Generate the synthetic corpus (if not already done)
python generate_evaluation_corpus.py

# 2. Index all documents
python index_evaluation_corpus.py

# 3. Run evaluation
python evaluation/evaluate_retrieval.py

# Results written to:
#   evaluation/results.json
#   evaluation/results.md
```

---

## Known Limitations

| Limitation | Impact | Planned Fix |
|---|---|---|
| Short evaluation documents | Hit@k can be easier than production corpora | Evaluate larger multi-chunk corpora in future |
| No calibrated confidence | Application confidence is heuristic | Calibration study is future work |
| No cross-encoder reranking | Dense ranking only | Reranking is future work |

## M2 Evaluation

The final M2 end-to-end run used 20 selected queries: four factual, four
procedural, four comparative, four ambiguous, and four
unavailable-information queries. The source corpus contains Software
Engineering and Hospital Administration queries. Unavailable-information
labels remain factual for classification; evidence availability is measured at
retrieval and response stages.

| Metric | Actual result |
|---|---:|
| Classification accuracy | 100.0% |
| Retrieval evidence success | 100.0% |
| Grounded-response rate | 100.0% |
| Citation coverage | 100.0% |
| Ambiguous detection rate | 100.0% |
| No-evidence handling rate | 75.0% |
| End-to-end success rate | 95.0% |

The evaluation ran the real M2 agents and orchestrator and mocked only the
external OpenAI chat-completions API. No automated LLM factual-accuracy claim
is made for the mocked output.
Therefore these results do not measure real LLM factual accuracy or readability.
The query-level records, request IDs, agent sequence, ranked/filtered evidence,
confidence values, and response fields are in
[`../evaluation/m2_end_to_end_results.json`](../evaluation/m2_end_to_end_results.json).
