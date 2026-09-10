# RAG Query Flow — M1 + M2

## Implemented flow

```text
User Query
  ↓
FastAPI POST /retrieve or POST /chat
  ↓
QueryUnderstandingAgent
  ↓
RetrievalAgent → SemanticRetriever → VectorStore → FAISS
  ↓
ResponseGenerationAgent
  ↓
Final ResponseResult
```

For an ambiguous query:

```text
QueryUnderstandingAgent
  ↓ routing = CLARIFICATION
Structured clarification-needed response
  ↓
M3 and later: multi-turn clarification
```

## Query understanding

Classification is deterministic and rule-based:

| Category | Classification cues | Routing |
|---|---|---|
| `factual` | definitions, facts, properties, values | `RETRIEVAL` |
| `procedural` | how, steps, workflow, protocol, process | `RETRIEVAL` |
| `comparative` | compare, versus, difference, contrast | `RETRIEVAL` |
| `ambiguous` | too short, vague, unresolved referent | `CLARIFICATION` |

`unavailable-information` is a factual query condition, not a fifth category.
Classification confidence is deterministic application-level evidence, not a
calibrated probability.

## Retrieval

The retrieval agent requests configurable Top-K results from the existing FAISS
retriever, ranks by L2 distance, and applies optional relevance filtering.
FAISS distance is preserved as `distance_score`; relevance is derived as
`1 / (1 + distance_score)`. Metadata is preserved in every hit. Empty stores,
empty results, and all-filtered results produce `sufficient_evidence=false` and
`no_relevant_information=true`.

## Response generation

When configured, the response agent calls the OpenAI chat-completions client with
a prompt that requires:

1. only supplied context;
2. no outside knowledge or invented facts;
3. explicit insufficient-evidence handling;
4. factual, procedural, or comparative style appropriate to the query;
5. citations derived only from retrieved metadata.

Confidence levels are `HIGH`, `MEDIUM`, or `LOW` according to the documented
application policy. They are not calibrated probabilities. No-information
responses are returned when evidence is absent or below the configured policy.

## API contract

### `POST /upload`

Accepts PDF, DOCX, TXT, and CSV multipart files. The route uses:

```text
validate → extract → clean → chunk → embed → FAISS + metadata
```

### `POST /retrieve`

Request:

```json
{
  "query": "What are the steps to create a feature branch?",
  "top_k": 3,
  "session_id": "optional"
}
```

Response includes `request_id`, `query`, `query_type`,
`classification_confidence`, `status`, `answer`, `confidence`,
`confidence_level`, `citations`, retrieval summary/results,
`no_information_found`, and `clarification_needed`.

### `GET /health`

Returns `{"status": "ok"}`.

## Evaluation

The final M2 evaluation uses the two existing domains and records stage-level
outputs in [`../evaluation/m2_end_to_end_results.json`](../evaluation/m2_end_to_end_results.json).
The report is in [`../evaluation/m2_end_to_end_results.md`](../evaluation/m2_end_to_end_results.md).
When no OpenAI key is configured, the evaluator uses a labeled deterministic
mock; no automated LLM factual-accuracy claim is made.

## Future: M3 and later

Multi-turn clarification, hybrid retrieval, reranking, UI, voice, and managed
vector services are not part of the implemented M1+M2 backend.
