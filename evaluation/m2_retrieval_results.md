# M2 Retrieval Results

- Top-K: 5
- Minimum relevance: 0.45
- Relevance: `1 / (1 + distance_score)`; lower FAISS L2 distance is better.
- Calibration candidates: 0.0, 0.45, 0.48, 0.5, 0.55

## Summary

- Total queries: 27
- Available queries: 15
- Unavailable-information queries: 12
- Retrieval hit rate: 100.0%
- Empty-result rate: 40.7%
- Low-confidence rate: 40.7%
- Unavailable neighbors rejected as evidence: 11/12
- Unavailable queries with sufficient evidence: 1/12

Unavailable queries are expected to have no relevant evidence. FAISS nearest neighbors are recorded but are not counted as retrieval hits.

## Domain performance

| Domain | Total | Available | Hit rate | Empty rate | Low-confidence rate |
|---|---:|---:|---:|---:|---:|
| Hospital Administration | 14 | 8 | 100.0% | 35.7% | 35.7% |
| Software Engineering | 13 | 7 | 100.0% | 46.2% | 46.2% |

## Query-type performance

| Query type | Total | Available | Hit rate | Empty rate | Low-confidence rate |
|---|---:|---:|---:|---:|---:|
| ambiguous | 8 | 0 | n/a | 100.0% | 100.0% |
| comparative | 4 | 4 | 100.0% | 0.0% | 0.0% |
| factual | 10 | 6 | 100.0% | 30.0% | 30.0% |
| procedural | 5 | 5 | 100.0% | 0.0% | 0.0% |

## Threshold calibration

Evidence presence is measured against the expected document for the 15 available queries. No-result and low-confidence rates include all 19 queries. The score is the deterministic transform `1 / (1 + distance_score)`, not a calibrated probability.

| Threshold | Evidence hit | No-result | Low-confidence | Top-1 | Top-3 | Top-5 | Unavailable rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 100.0% | 0.0% | 40.7% | 100.0% | 100.0% | 100.0% | 0.0% |
| 0.45 | 100.0% | 40.7% | 40.7% | 100.0% | 100.0% | 100.0% | 91.7% |
| 0.48 | 73.3% | 55.6% | 55.6% | 73.3% | 73.3% | 73.3% | 91.7% |
| 0.5 | 46.7% | 70.4% | 70.4% | 46.7% | 46.7% | 46.7% | 91.7% |
| 0.55 | 33.3% | 81.5% | 81.5% | 33.3% | 33.3% | 33.3% | 100.0% |

**Selected threshold:** `0.45`.

0.45 retained all available expected documents at Top-1/3/5 while rejecting 3 of 4 unavailable-query neighbors. Higher thresholds removed relevant evidence; 0.0 retained every neighbor.

Limitations: the corpus is small, the score is derived from FAISS L2 distance rather than calibrated probability, and one unavailable query remains above the selected threshold.

## Query details

| Query | Type | Raw | Filtered | Confidence | Sufficient | No information |
|---|---|---:|---:|---:|---|---|
| How long is a typical agile sprint in the engineering guide? | factual | 5 | 1 | 0.483 | yes | no |
| What are the steps to create a feature branch in git? | procedural | 5 | 1 | 0.620 | yes | no |
| How do microservices differ from a monolithic architecture? | comparative | 5 | 1 | 0.525 | yes | no |
| What naming convention should classes use in the coding standards? | factual | 5 | 1 | 0.499 | yes | no |
| How should production errors be logged according to coding standards? | procedural | 5 | 1 | 0.455 | yes | no |
| Compare Rust and Go for systems and cloud infrastructure programming. | comparative | 5 | 1 | 0.474 | yes | no |
| Which programming languages use static typing? | factual | 5 | 1 | 0.459 | yes | no |
| What is the annual revenue of TechCorp International? | factual | 5 | 0 | 0.000 | no | yes |
| Who invented the Python programming language and in what year? | factual | 5 | 0 | 0.000 | no | yes |
| How many hours do non-emergency patients have to finalize admission paperwork? | factual | 5 | 3 | 0.560 | yes | no |
| What insurance documents are required at patient registration? | factual | 5 | 1 | 0.482 | yes | no |
| What are the five rights of medication administration? | procedural | 5 | 2 | 0.477 | yes | no |
| How should nurses perform hand hygiene before patient contact? | procedural | 5 | 1 | 0.575 | yes | no |
| What is the difference between Code Blue and Code Red emergencies? | comparative | 5 | 1 | 0.488 | yes | no |
| Compare oral versus intravenous Ibuprofen dosing limits. | comparative | 5 | 1 | 0.502 | yes | no |
| What is the standard oral dose of Acetaminophen? | factual | 5 | 1 | 0.553 | yes | no |
| What steps should staff follow during a Code Blue cardiac arrest? | procedural | 5 | 2 | 0.586 | yes | no |
| What was Riverside General Hospital's total revenue last fiscal year? | factual | 5 | 1 | 0.534 | yes | no |
| Who won the Nobel Prize in Medicine this year? | factual | 5 | 0 | 0.000 | no | yes |
| Which protocol? | ambiguous | 5 | 0 | 0.000 | no | yes |
| What about that? | ambiguous | 5 | 0 | 0.000 | no | yes |
| That process? | ambiguous | 5 | 0 | 0.000 | no | yes |
| Can you explain this? | ambiguous | 5 | 0 | 0.000 | no | yes |
| Which protocol? | ambiguous | 5 | 0 | 0.000 | no | yes |
| What about that? | ambiguous | 5 | 0 | 0.000 | no | yes |
| That process? | ambiguous | 5 | 0 | 0.000 | no | yes |
| Can you explain this? | ambiguous | 5 | 0 | 0.000 | no | yes |

## M1 comparison

Source: `evaluation\results.json`. Existing M1 metrics including unavailable queries:

- Hit@1: 55.56%
- Hit@3: 55.56%
- Hit@5: 55.56%
