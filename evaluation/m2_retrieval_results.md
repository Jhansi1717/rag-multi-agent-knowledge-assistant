# M2 Retrieval Results

- Top-K: 3
- Minimum relevance: 0.0
- Relevance: `1 / (1 + distance_score)`; lower FAISS L2 distance is better.

## Summary

- Total queries: 19
- Available queries: 15
- Unavailable-information queries: 4
- Retrieval hit rate: 0.0%
- Empty-result rate: 100.0%
- Low-confidence rate: 100.0%
- Unavailable neighbors rejected as evidence: 4/4
- Unavailable queries with sufficient evidence: 0/4

Unavailable queries are expected to have no relevant evidence. FAISS nearest neighbors are recorded but are not counted as retrieval hits.

## Domain performance

| Domain | Total | Available | Hit rate | Empty rate | Low-confidence rate |
|---|---:|---:|---:|---:|---:|
| Hospital Administration | 10 | 8 | 0.0% | 100.0% | 100.0% |
| Software Engineering | 9 | 7 | 0.0% | 100.0% | 100.0% |

## Query-type performance

| Query type | Total | Available | Hit rate | Empty rate | Low-confidence rate |
|---|---:|---:|---:|---:|---:|
| comparative | 4 | 4 | 0.0% | 100.0% | 100.0% |
| factual | 10 | 6 | 0.0% | 100.0% | 100.0% |
| procedural | 5 | 5 | 0.0% | 100.0% | 100.0% |

## Query details

| Query | Type | Raw | Filtered | Confidence | Sufficient | No information |
|---|---|---:|---:|---:|---|---|
| How long is a typical agile sprint in the engineering guide? | factual | 1 | 0 | 0.000 | no | yes |
| What are the steps to create a feature branch in git? | procedural | 1 | 0 | 0.000 | no | yes |
| How do microservices differ from a monolithic architecture? | comparative | 1 | 0 | 0.000 | no | yes |
| What naming convention should classes use in the coding standards? | factual | 1 | 0 | 0.000 | no | yes |
| How should production errors be logged according to coding standards? | procedural | 1 | 0 | 0.000 | no | yes |
| Compare Rust and Go for systems and cloud infrastructure programming. | comparative | 1 | 0 | 0.000 | no | yes |
| Which programming languages use static typing? | factual | 1 | 0 | 0.000 | no | yes |
| What is the annual revenue of TechCorp International? | factual | 1 | 0 | 0.000 | no | yes |
| Who invented the Python programming language and in what year? | factual | 1 | 0 | 0.000 | no | yes |
| How many hours do non-emergency patients have to finalize admission paperwork? | factual | 1 | 0 | 0.000 | no | yes |
| What insurance documents are required at patient registration? | factual | 1 | 0 | 0.000 | no | yes |
| What are the five rights of medication administration? | procedural | 1 | 0 | 0.000 | no | yes |
| How should nurses perform hand hygiene before patient contact? | procedural | 1 | 0 | 0.000 | no | yes |
| What is the difference between Code Blue and Code Red emergencies? | comparative | 1 | 0 | 0.000 | no | yes |
| Compare oral versus intravenous Ibuprofen dosing limits. | comparative | 1 | 0 | 0.000 | no | yes |
| What is the standard oral dose of Acetaminophen? | factual | 1 | 0 | 0.000 | no | yes |
| What steps should staff follow during a Code Blue cardiac arrest? | procedural | 1 | 0 | 0.000 | no | yes |
| What was Riverside General Hospital's total revenue last fiscal year? | factual | 1 | 0 | 0.000 | no | yes |
| Who won the Nobel Prize in Medicine this year? | factual | 1 | 0 | 0.000 | no | yes |

## M1 comparison

Source: `evaluation\results.json`. Existing M1 metrics including unavailable queries:

- Hit@1: 78.95%
- Hit@3: 78.95%
- Hit@5: 78.95%
