# M2 End-to-End Evaluation

## Methodology

- Source corpus: `data\evaluation\queries.json` (19 queries).
- Domains: Software Engineering and Hospital Administration.
- Added two explicit ambiguous cases because the source corpus contains none.
- `unavailable-information` is treated as factual for classification; evidence availability is measured separately.
- Generation mode: `mock_context_echo`.
- No automated factual-accuracy claim is made for LLM output.
- Manual review fields are included for a small sample and remain pending human review.

## Actual metrics

| Metric | Result |
|---|---:|
| Total Queries | 21 |
| Classification Accuracy | 100.0% |
| Retrieval Success | 5.9% |
| Grounded Response Rate | 0.0% |
| Citation Coverage | 0.0% |
| Ambiguous Detection Rate | 100.0% |
| No Evidence Handling Rate | 100.0% |
| End To End Success Rate | 100.0% |

## Query-level results

| ID | Domain | Expected | Predicted | Routing | Evidence | Grounded | Citations | Status |
|---|---|---|---|---|---:|---:|---:|---|
| se_q01 | Software Engineering | factual | factual | RETRIEVAL | no | no | no | unavailable |
| se_q02 | Software Engineering | procedural | procedural | RETRIEVAL | no | no | no | unavailable |
| se_q03 | Software Engineering | comparative | comparative | RETRIEVAL | no | no | no | unavailable |
| se_q04 | Software Engineering | factual | factual | RETRIEVAL | no | no | no | unavailable |
| se_q05 | Software Engineering | procedural | procedural | RETRIEVAL | yes | no | no | unavailable |
| se_q06 | Software Engineering | comparative | comparative | RETRIEVAL | no | no | no | unavailable |
| se_q07 | Software Engineering | factual | factual | RETRIEVAL | no | no | no | unavailable |
| se_q08 | Software Engineering | unavailable-information | factual | RETRIEVAL | yes | no | no | unavailable |
| se_q09 | Software Engineering | unavailable-information | factual | RETRIEVAL | no | no | no | unavailable |
| ha_q01 | Hospital Administration | factual | factual | RETRIEVAL | no | no | no | unavailable |
| ha_q02 | Hospital Administration | factual | factual | RETRIEVAL | no | no | no | unavailable |
| ha_q03 | Hospital Administration | procedural | procedural | RETRIEVAL | no | no | no | unavailable |
| ha_q04 | Hospital Administration | procedural | procedural | RETRIEVAL | no | no | no | unavailable |
| ha_q05 | Hospital Administration | comparative | comparative | RETRIEVAL | no | no | no | unavailable |
| ha_q06 | Hospital Administration | comparative | comparative | RETRIEVAL | no | no | no | unavailable |
| ha_q07 | Hospital Administration | factual | factual | RETRIEVAL | no | no | no | unavailable |
| ha_q08 | Hospital Administration | procedural | procedural | RETRIEVAL | no | no | no | unavailable |
| ha_q09 | Hospital Administration | unavailable-information | factual | RETRIEVAL | no | no | no | unavailable |
| ha_q10 | Hospital Administration | unavailable-information | factual | RETRIEVAL | yes | no | no | unavailable |
| se_ambiguous_01 | Software Engineering | ambiguous | ambiguous | CLARIFICATION | no | no | no | clarification_needed |
| ha_ambiguous_01 | Hospital Administration | ambiguous | ambiguous | CLARIFICATION | no | no | no | clarification_needed |

## Manual review sample

- `se_q01`: pending human review.
- `se_q02`: pending human review.
- `se_q03`: pending human review.
- `se_q04`: pending human review.
