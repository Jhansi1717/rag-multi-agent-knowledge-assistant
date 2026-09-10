# Retrieval Validation Report

## Aggregate Metrics
- **Total Eval Queries (Excl. Unavailable)**: 6
- **Hit@1**: 83.33%
- **Hit@3**: 100.00%
- **Hit@5**: 100.00%

## Detailed Results
| ID | Domain | Type | Query | Expected | Top-1 | Rank | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|---|---|---|---|
| q1 | HR | Factual | How many annual leave days are provided? | leave_policy.pdf | leave_policy.pdf | 1 | 1 | 1 | 1 |
| q2 | HR | Procedural | How do employees apply for leave? | leave_policy.pdf | leave_policy.pdf | 1 | 1 | 1 | 1 |
| q3 | HR | Comparative | What is the difference between Annual and Sick leave carry-forward limit? | leave_allowance.csv | leave_policy.pdf | 2 | 0 | 1 | 1 |
| q4 | HR | Unavailable-information | What was the company's revenue in 2025? | NONE | leave_allowance.csv | N/A | 0 | 0 | 0 |
| q5 | Software | Factual | What HTTP method is used to remove a resource? | api_documentation.docx | api_documentation.docx | 1 | 1 | 1 | 1 |
| q6 | Software | Procedural | What are the steps to deploy a backend service? | deployment_guide.txt | deployment_guide.txt | 1 | 1 | 1 | 1 |
| q7 | Software | Comparative | How does REST differ from SOAP? | api_documentation.docx | api_documentation.docx | 1 | 1 | 1 | 1 |
| q8 | Software | Unavailable-information | Who is the CEO of the software division? | NONE | deployment_guide.txt | N/A | 0 | 0 | 0 |

## Analysis
### Failure Cases & Low-Relevance Results
Reviewing the results above allows us to spot any discrepancies between Expected and Top-1.
- **Query**: What is the difference between Annual and Sick leave carry-forward limit? (Type: Comparative)
  - Expected: leave_allowance.csv
  - Actual Top-1: leave_policy.pdf

### Limitations and Likely Causes
- **Unavailable Information**: These queries returned documents because FAISS always returns the nearest neighbor, even if the absolute distance is high. This highlights a limitation: the system needs a similarity threshold or an LLM to decide if the retrieved context actually contains the answer.
- **Comparative Queries**: Semantic embeddings sometimes struggle with relational comparisons if the exact wording differs heavily from the chunk.
