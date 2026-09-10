# Agents — RAG Multi-Agent Knowledge Assistant

The M2 agents are plain Python classes coordinated by a fixed sequential
orchestrator. They do not use LangChain, LangGraph, autonomous planning, or
agent-to-agent negotiation.

## Implemented M2 flow

```text
                         User Query
                             │
                             ▼
                 QueryUnderstandingAgent
                             │
                ┌────────────┴────────────┐
                │                         │
          ambiguous                 clear intent
                │                         │
                ▼                         ▼
        M3 clarification           RetrievalAgent
                                          │
                                  ┌───────┴────────┐
                                  │                │
                             evidence        insufficient
                                  │                │
                                  ▼                ▼
                         ResponseGenerationAgent  No-information response
                                  │
                                  ▼
                            Final Response
```

Classification determines the question type. Retrieval determines whether the
knowledge base contains sufficient evidence. `unavailable` is therefore a
response/evidence status, not a query classification.

## Agent responsibilities

| Agent | Input | Output |
|---|---|---|
| `QueryUnderstandingAgent` | Raw query | `QueryUnderstandingResult` |
| `RetrievalAgent` | Normalized query, type, domain, Top-K | `RetrievalResult` |
| `ResponseGenerationAgent` | Query, retrieved hits, retrieval confidence | `ResponseResult` |
| `ConversationMemoryAgent` | Session turns | Recorded context only |

### Query understanding

The deterministic classifier returns exactly four query types:

- `factual`
- `procedural`
- `comparative`
- `ambiguous`

Factual, procedural, and comparative queries route to retrieval. Ambiguous
queries route to the minimal clarification-needed response; richer
multi-turn clarification is planned for M3.

Classification confidence is deterministic application-level evidence. It is
not a calibrated probability.

### Retrieval

The retrieval agent reuses the existing semantic retriever and FAISS
`IndexFlatL2` store. Results are ranked by lower-is-better L2 distance,
preserve `distance_score` separately, and expose the application relevance
score:

```text
relevance_score = 1 / (1 + distance_score)
```

Configurable Top-K and minimum relevance filtering determine whether sufficient
evidence exists. Empty stores, empty results, and all-filtered results produce
a no-information outcome; nearest-neighbor output alone is not treated as
proof that the knowledge base contains an answer.

### Response generation

When configured, the response agent uses the OpenAI provider with a grounding
prompt that requires context-only answers, no outside knowledge, explicit
insufficient-evidence handling, and citations derived only from retrieved
metadata. Without sufficient evidence or provider configuration it returns a
controlled no-information response.

Response confidence levels are application-level labels:

- `HIGH`: confidence >= 0.75
- `MEDIUM`: 0.45 <= confidence < 0.75
- `LOW`: confidence < 0.45

These labels are not calibrated probabilities.

### Memory

`ConversationMemoryAgent` records completed turns but is not a required
resolution stage. It does not interrupt the M2 sequence.

## Shared contracts

The primary handoff models are defined in `agents/models.py`:

- `QueryUnderstandingResult`
- `RetrievalHit`
- `RetrievalResult`
- `ResponseResult`
- `AgentError`

Each orchestrated response carries a `request_id` for correlation. Stage
failures return structured errors rather than silently falling back.

## Module layout

```text
agents/
├── orchestrator.py
├── query_understanding.py
├── retrieval_agent.py
├── response_generation.py
├── clarification.py
├── memory.py
└── models.py
```

## Tests

```bash
python -m pytest test_m14_retrieval.py -v
python -m pytest test_m24_orchestrator.py test_m2_response_generation.py -v
```
