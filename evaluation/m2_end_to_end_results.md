# M2 End-to-End Evaluation

## Methodology

- Source corpus: `data\evaluation\queries.json`.
- Query count: 20 (four per category).
- Domains: Software Engineering and Hospital Administration.
- Generation path: Real QueryUnderstandingAgent -> RetrievalAgent -> ResponseGenerationAgent -> Orchestrator; only the external OpenAI chat-completions API is mocked.
- No automated factual-accuracy claim is made for mocked LLM output.

## Actual metrics

| Metric | Result |
|---|---:|
| Total Queries | 20 |
| Classification Accuracy | 100.0% |
| Retrieval Evidence Success | 100.0% |
| Grounded Response Rate | 100.0% |
| Citation Coverage | 100.0% |
| No Evidence Handling Rate | 75.0% |
| End To End Success Rate | 95.0% |
| Ambiguous Detection Rate | 100.0% |

## Query-level results

| ID | Domain | Expected | Predicted | Confidence | Routing | Evidence | Grounded | Citations | No information | Status |
|---|---|---|---|---:|---|---|---|---|---|---|
| se_q01 | Software Engineering | factual | factual | 0.800 | RETRIEVAL | yes | yes | yes | no | answered |
| se_q04 | Software Engineering | factual | factual | 0.800 | RETRIEVAL | yes | yes | yes | no | answered |
| ha_q01 | Hospital Administration | factual | factual | 0.800 | RETRIEVAL | yes | yes | yes | no | answered |
| ha_q02 | Hospital Administration | factual | factual | 0.800 | RETRIEVAL | yes | yes | yes | no | answered |
| se_q02 | Software Engineering | procedural | procedural | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| se_q05 | Software Engineering | procedural | procedural | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| ha_q03 | Hospital Administration | procedural | procedural | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| ha_q04 | Hospital Administration | procedural | procedural | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| se_q03 | Software Engineering | comparative | comparative | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| se_q06 | Software Engineering | comparative | comparative | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| ha_q05 | Hospital Administration | comparative | comparative | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| ha_q06 | Hospital Administration | comparative | comparative | 0.950 | RETRIEVAL | yes | yes | yes | no | answered |
| se_q10 | Software Engineering | ambiguous | ambiguous | 0.900 | CLARIFICATION | no | no | no | no | clarification_needed |
| se_q11 | Software Engineering | ambiguous | ambiguous | 0.900 | CLARIFICATION | no | no | no | no | clarification_needed |
| ha_q11 | Hospital Administration | ambiguous | ambiguous | 0.900 | CLARIFICATION | no | no | no | no | clarification_needed |
| ha_q12 | Hospital Administration | ambiguous | ambiguous | 0.900 | CLARIFICATION | no | no | no | no | clarification_needed |
| se_q08 | Software Engineering | unavailable-information | factual | 0.800 | RETRIEVAL | no | no | no | yes | unavailable |
| se_q09 | Software Engineering | unavailable-information | factual | 0.800 | RETRIEVAL | no | no | no | yes | unavailable |
| ha_q09 | Hospital Administration | unavailable-information | factual | 0.800 | RETRIEVAL | no | yes | yes | no | answered |
| ha_q10 | Hospital Administration | unavailable-information | factual | 0.800 | RETRIEVAL | no | no | no | yes | unavailable |
