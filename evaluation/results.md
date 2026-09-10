# M1 Retrieval Evaluation Results

Metrics computed from live FAISS retrieval against `data/evaluation/queries.json`.
Hit@k = expected_document appears in the top-k filenames.
Unavailable-information queries have `expected_document=NONE` and are excluded from the primary Hit@ averages because no relevant document exists; FAISS still returns a nearest neighbor.

## Overall

- Total queries: **19**
- Scored (excl. unavailable): **15**
- Unavailable-information: **4**

| Metric | Excl. unavailable | Incl. unavailable |
|---|---|---|
| Hit@1 | 100.0% | 78.95% |
| Hit@3 | 100.0% | 78.95% |
| Hit@5 | 100.0% | 78.95% |

## Domain-wise (excl. unavailable)

| Domain | Queries | Scored | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|
| Software Engineering | 9 | 7 | 100.0% | 100.0% | 100.0% |
| Hospital Administration | 10 | 8 | 100.0% | 100.0% | 100.0% |

## Category-wise (excl. unavailable)

| Category | Queries | Scored | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|
| factual | 6 | 6 | 100.0 | 100.0 | 100.0 |
| procedural | 5 | 5 | 100.0 | 100.0 | 100.0 |
| comparative | 4 | 4 | 100.0 | 100.0 | 100.0 |
| unavailable-information | 4 | 0 | n/a | n/a | n/a |

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
| se_q08 | Software Engineering | unavailable-information | NONE | medication_dosage_reference.csv | N/A | 0 | 0 | 0 |
| se_q09 | Software Engineering | unavailable-information | NONE | coding_standards.docx | N/A | 0 | 0 | 0 |
| ha_q01 | Hospital Administration | factual | patient_admission_policy.pdf | patient_admission_policy.pdf | 1 | 1 | 1 | 1 |
| ha_q02 | Hospital Administration | factual | patient_admission_policy.pdf | patient_admission_policy.pdf | 1 | 1 | 1 | 1 |
| ha_q03 | Hospital Administration | procedural | nursing_procedures.docx | nursing_procedures.docx | 1 | 1 | 1 | 1 |
| ha_q04 | Hospital Administration | procedural | nursing_procedures.docx | nursing_procedures.docx | 1 | 1 | 1 | 1 |
| ha_q05 | Hospital Administration | comparative | emergency_protocols.txt | emergency_protocols.txt | 1 | 1 | 1 | 1 |
| ha_q06 | Hospital Administration | comparative | medication_dosage_reference.csv | medication_dosage_reference.csv | 1 | 1 | 1 | 1 |
| ha_q07 | Hospital Administration | factual | medication_dosage_reference.csv | medication_dosage_reference.csv | 1 | 1 | 1 | 1 |
| ha_q08 | Hospital Administration | procedural | emergency_protocols.txt | emergency_protocols.txt | 1 | 1 | 1 | 1 |
| ha_q09 | Hospital Administration | unavailable-information | NONE | patient_admission_policy.pdf | N/A | 0 | 0 | 0 |
| ha_q10 | Hospital Administration | unavailable-information | NONE | nursing_procedures.docx | N/A | 0 | 0 | 0 |

## Failed queries

### se_q08 — unavailable-information
- Query: What is the annual revenue of TechCorp International?
- Expected: `NONE`
- Top-1: `medication_dosage_reference.csv` (L2=1.7733144760131836)
- Rank of expected: None
- Cause: Unavailable-information query: FAISS still returned a nearest neighbor (medication_dosage_reference.csv). No relevant document exists in the KB.

### se_q09 — unavailable-information
- Query: Who invented the Python programming language and in what year?
- Expected: `NONE`
- Top-1: `coding_standards.docx` (L2=1.4667860269546509)
- Rank of expected: None
- Cause: Unavailable-information query: FAISS still returned a nearest neighbor (coding_standards.docx). No relevant document exists in the KB.

### ha_q09 — unavailable-information
- Query: What was Riverside General Hospital's total revenue last fiscal year?
- Expected: `NONE`
- Top-1: `patient_admission_policy.pdf` (L2=0.8734406232833862)
- Rank of expected: None
- Cause: Unavailable-information query: FAISS still returned a nearest neighbor (patient_admission_policy.pdf). No relevant document exists in the KB.

### ha_q10 — unavailable-information
- Query: Who won the Nobel Prize in Medicine this year?
- Expected: `NONE`
- Top-1: `nursing_procedures.docx` (L2=1.8947561979293823)
- Rank of expected: None
- Cause: Unavailable-information query: FAISS still returned a nearest neighbor (nursing_procedures.docx). No relevant document exists in the KB.


## Low-relevance examples

- `se_q08`: Top-1 `medication_dosage_reference.csv` L2=1.7733144760131836 (expected `NONE`) — What is the annual revenue of TechCorp International?
- `se_q09`: Top-1 `coding_standards.docx` L2=1.4667860269546509 (expected `NONE`) — Who invented the Python programming language and in what year?
- `ha_q09`: Top-1 `patient_admission_policy.pdf` L2=0.8734406232833862 (expected `NONE`) — What was Riverside General Hospital's total revenue last fiscal year?
- `ha_q10`: Top-1 `nursing_procedures.docx` L2=1.8947561979293823 (expected `NONE`) — Who won the Nobel Prize in Medicine this year?

## Inspected failed cases (up to 3)

### se_q08
- Query: What is the annual revenue of TechCorp International?
- Expected document: `NONE`
- Keywords found in Top-5: []
- Top-5:
  - #1 `medication_dosage_reference.csv` L2=1.7733 — Row 1: Medication: Acetaminophen | Standard_Dose: 650 mg | Route: Oral | Frequency: Every 6 hours | Max_Daily_Dose: 3000 mg  Row 2: Medication: Ibuprofen | Stan
  - #2 `git_workflow.txt` L2=1.8001 — Git Workflow for Software Teams  1. Branch Naming Feature branches use the format feature/<ticket-id>-short-description. Bug fixes use bugfix/<ticket-id>-short-
  - #3 `microservices_architecture.pdf` L2=1.8527 — Microservices and Agile Delivery Guide 1. Agile Sprint Cadence Teams operate in two-week sprints. Each sprint begins with planning, includes daily standups, and
  - #4 `programming_languages.csv` L2=1.8674 — Row 1: Language: Python | Paradigm: Multi-paradigm | Typing: Dynamic | Primary_Use_Case: Data science and backends | Learning_Curve: Low  Row 2: Language: Java 
  - #5 `coding_standards.docx` L2=1.8786 — Engineering Coding Standards  Naming Conventions  Classes use PascalCase. Functions and variables use snake_case. Constants use UPPER_SNAKE_CASE. Module names a

### se_q09
- Query: Who invented the Python programming language and in what year?
- Expected document: `NONE`
- Keywords found in Top-5: []
- Top-5:
  - #1 `coding_standards.docx` L2=1.4668 — Engineering Coding Standards  Naming Conventions  Classes use PascalCase. Functions and variables use snake_case. Constants use UPPER_SNAKE_CASE. Module names a
  - #2 `programming_languages.csv` L2=1.4927 — Row 1: Language: Python | Paradigm: Multi-paradigm | Typing: Dynamic | Primary_Use_Case: Data science and backends | Learning_Curve: Low  Row 2: Language: Java 
  - #3 `patient_admission_policy.pdf` L2=2.0411 — Riverside General Hospital - Patient Admission Policy 1. Registration Requirements Patients must present a government-issued photo ID and insurance card at admi
  - #4 `emergency_protocols.txt` L2=2.0449 — Riverside General Hospital - Emergency Response Protocols  1. Code Blue (Cardiac Arrest) Activate Code Blue for unresponsive patients without a pulse. Step 1: C
  - #5 `git_workflow.txt` L2=2.0503 — Git Workflow for Software Teams  1. Branch Naming Feature branches use the format feature/<ticket-id>-short-description. Bug fixes use bugfix/<ticket-id>-short-

### ha_q09
- Query: What was Riverside General Hospital's total revenue last fiscal year?
- Expected document: `NONE`
- Keywords found in Top-5: []
- Top-5:
  - #1 `patient_admission_policy.pdf` L2=0.8734 — Riverside General Hospital - Patient Admission Policy 1. Registration Requirements Patients must present a government-issued photo ID and insurance card at admi
  - #2 `emergency_protocols.txt` L2=1.4182 — Riverside General Hospital - Emergency Response Protocols  1. Code Blue (Cardiac Arrest) Activate Code Blue for unresponsive patients without a pulse. Step 1: C
  - #3 `nursing_procedures.docx` L2=1.6738 — Clinical Nursing Procedures Manual  Vital Signs Monitoring  Measure temperature, pulse, respiration, and blood pressure every 4 hours for standard inpatient uni
  - #4 `medication_dosage_reference.csv` L2=1.6871 — Row 1: Medication: Acetaminophen | Standard_Dose: 650 mg | Route: Oral | Frequency: Every 6 hours | Max_Daily_Dose: 3000 mg  Row 2: Medication: Ibuprofen | Stan
  - #5 `git_workflow.txt` L2=1.8044 — Git Workflow for Software Teams  1. Branch Naming Feature branches use the format feature/<ticket-id>-short-description. Bug fixes use bugfix/<ticket-id>-short-

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
