# M1.4 Retrieval Validation

**Status: BLOCKED**

Only verified `REAL_SOURCE_DOCUMENT` records can contribute to production acceptance. Generated samples, test fixtures, and mock retrievers are excluded.

## Acceptance

- Valid real-source queries: 0
- Synthetic/test-only queries: 28
- Generated-sample expected-source queries: 15
- Unavailable queries: 4
- Low-relevance queries: 1
- Comparative queries: 4
- Mock-retriever test suites classified separately: 1

| Query category | Cases |
|---|---:|
| factual | 6 |
| procedural | 5 |
| comparative | 4 |
| unavailable | 4 |
| low-relevance | 1 |

| Metric (valid real-source queries only) | Result |
|---|---:|
| Hit@1 | None |
| Hit@3 | None |
| Hit@5 | None |

`null` means not measured; it is not a zero score.

## Negative Cases

- Unavailable false positives: None (NOT PROVEN; eligible queries: 0)
- Low-relevance rejection: None (NOT PROVEN; eligible queries: 0)

## Missing Evidence

- `se_q01` expects `microservices_architecture.pdf` at `data/software_engineering/microservices_architecture.pdf` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `se_q02` expects `git_workflow.txt` at `data/software_engineering/git_workflow.txt` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `se_q03` expects `microservices_architecture.pdf` at `data/software_engineering/microservices_architecture.pdf` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `se_q04` expects `coding_standards.docx` at `data/software_engineering/coding_standards.docx` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `se_q05` expects `coding_standards.docx` at `data/software_engineering/coding_standards.docx` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `se_q06` expects `programming_languages.csv` at `data/software_engineering/programming_languages.csv` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `se_q07` expects `programming_languages.csv` at `data/software_engineering/programming_languages.csv` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q01` expects `patient_admission_policy.pdf` at `data/hospital_administration/patient_admission_policy.pdf` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q02` expects `patient_admission_policy.pdf` at `data/hospital_administration/patient_admission_policy.pdf` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q03` expects `nursing_procedures.docx` at `data/hospital_administration/nursing_procedures.docx` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q04` expects `nursing_procedures.docx` at `data/hospital_administration/nursing_procedures.docx` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q05` expects `emergency_protocols.txt` at `data/hospital_administration/emergency_protocols.txt` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q06` expects `medication_dosage_reference.csv` at `data/hospital_administration/medication_dosage_reference.csv` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q07` expects `medication_dosage_reference.csv` at `data/hospital_administration/medication_dosage_reference.csv` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- `ha_q08` expects `emergency_protocols.txt` at `data/hospital_administration/emergency_protocols.txt` (GENERATED_SAMPLE): Expected answer source is not a verified real source document.
- verified_real_source_corpus: No query has a verified REAL_SOURCE_DOCUMENT as its expected answer source.
  - Generated source requiring an original: `data/hospital_administration/emergency_protocols.txt`
  - Generated source requiring an original: `data/hospital_administration/medication_dosage_reference.csv`
  - Generated source requiring an original: `data/hospital_administration/nursing_procedures.docx`
  - Generated source requiring an original: `data/hospital_administration/patient_admission_policy.pdf`
  - Generated source requiring an original: `data/software_engineering/coding_standards.docx`
  - Generated source requiring an original: `data/software_engineering/git_workflow.txt`
  - Generated source requiring an original: `data/software_engineering/microservices_architecture.pdf`
  - Generated source requiring an original: `data/software_engineering/programming_languages.csv`
- complete_verified_negative_test_scope: Unavailable and low-relevance rejection require complete, verified source coverage for the tested domain; no such domain scope is registered.

## Query Provenance

| ID | Category | Query provenance | Expected source provenance | Production eligible |
|---|---|---|---|---|
| se_q01 | factual | TEST_FIXTURE | GENERATED_SAMPLE | False |
| se_q02 | procedural | TEST_FIXTURE | GENERATED_SAMPLE | False |
| se_q03 | comparative | TEST_FIXTURE | GENERATED_SAMPLE | False |
| se_q04 | factual | TEST_FIXTURE | GENERATED_SAMPLE | False |
| se_q05 | procedural | TEST_FIXTURE | GENERATED_SAMPLE | False |
| se_q06 | comparative | TEST_FIXTURE | GENERATED_SAMPLE | False |
| se_q07 | factual | TEST_FIXTURE | GENERATED_SAMPLE | False |
| se_q08 | factual | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| se_q09 | factual | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| ha_q01 | factual | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q02 | factual | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q03 | procedural | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q04 | procedural | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q05 | comparative | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q06 | comparative | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q07 | factual | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q08 | procedural | TEST_FIXTURE | GENERATED_SAMPLE | False |
| ha_q09 | factual | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| ha_q10 | factual | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| se_q10 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| se_q11 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| se_q12 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| se_q13 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| ha_q11 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| ha_q12 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| ha_q13 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| ha_q14 | ambiguous | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |
| m14_low_relevance_01 | low-relevance | TEST_FIXTURE | NO_EXPECTED_SOURCE | False |

## Test-Only Suites

| Suite | Provenance | Counts toward acceptance |
|---|---|---|
| `test_m14_retrieval.py::TestRetrievalAgent` | MOCK_RETRIEVER | False |
| `test_m14_retrieval.py::TestOrchestrator` | GENERATED_SAMPLE | False |
| `test_m14_retrieval.py::TestQueryUnderstanding` | TEST_FIXTURE | False |
| `test_retrieval.py::test_semantic_retrieval` | TEST_FIXTURE | False |
| `evaluation/evaluate_m2_retrieval.py` | GENERATED_SAMPLE | False |

## Decision

M1.4 is BLOCKED until original source documents with verifiable origin records, source hashes, and complete domain scope are registered and evaluated. Do not interpret generated-sample results or mock-retriever tests as production evidence.
