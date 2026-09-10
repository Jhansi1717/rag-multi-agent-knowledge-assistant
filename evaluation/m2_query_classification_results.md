# M2.1 Query Classification Results

The official M2.1 corpus uses only factual, procedural, comparative, and ambiguous classification labels. Evidence availability is determined by retrieval, not classification.

## Summary

- Total queries: 27
- Correct: 27
- Incorrect: 0
- Accuracy: 100.0%

## Per-class accuracy

| Class | Total | Correct | Accuracy |
|---|---:|---:|---:|
| factual | 10 | 10 | 100.0% |
| procedural | 5 | 5 | 100.0% |
| comparative | 4 | 4 | 100.0% |
| ambiguous | 8 | 8 | 100.0% |

## Query results

| Query | Expected | Predicted | Confidence | Routing | Correct |
|---|---|---|---:|---|---|
| How long is a typical agile sprint in the engineering guide? | factual | factual | 0.80 | RETRIEVAL | yes |
| What are the steps to create a feature branch in git? | procedural | procedural | 0.95 | RETRIEVAL | yes |
| How do microservices differ from a monolithic architecture? | comparative | comparative | 0.95 | RETRIEVAL | yes |
| What naming convention should classes use in the coding standards? | factual | factual | 0.80 | RETRIEVAL | yes |
| How should production errors be logged according to coding standards? | procedural | procedural | 0.95 | RETRIEVAL | yes |
| Compare Rust and Go for systems and cloud infrastructure programming. | comparative | comparative | 0.95 | RETRIEVAL | yes |
| Which programming languages use static typing? | factual | factual | 0.80 | RETRIEVAL | yes |
| What is the annual revenue of TechCorp International? | factual | factual | 0.80 | RETRIEVAL | yes |
| Who invented the Python programming language and in what year? | factual | factual | 0.80 | RETRIEVAL | yes |
| How many hours do non-emergency patients have to finalize admission paperwork? | factual | factual | 0.80 | RETRIEVAL | yes |
| What insurance documents are required at patient registration? | factual | factual | 0.80 | RETRIEVAL | yes |
| What are the five rights of medication administration? | procedural | procedural | 0.95 | RETRIEVAL | yes |
| How should nurses perform hand hygiene before patient contact? | procedural | procedural | 0.95 | RETRIEVAL | yes |
| What is the difference between Code Blue and Code Red emergencies? | comparative | comparative | 0.95 | RETRIEVAL | yes |
| Compare oral versus intravenous Ibuprofen dosing limits. | comparative | comparative | 0.95 | RETRIEVAL | yes |
| What is the standard oral dose of Acetaminophen? | factual | factual | 0.80 | RETRIEVAL | yes |
| What steps should staff follow during a Code Blue cardiac arrest? | procedural | procedural | 0.95 | RETRIEVAL | yes |
| What was Riverside General Hospital's total revenue last fiscal year? | factual | factual | 0.80 | RETRIEVAL | yes |
| Who won the Nobel Prize in Medicine this year? | factual | factual | 0.80 | RETRIEVAL | yes |
| Which protocol? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
| What about that? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
| That process? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
| Can you explain this? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
| Which protocol? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
| What about that? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
| That process? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
| Can you explain this? | ambiguous | ambiguous | 0.90 | CLARIFICATION | yes |
