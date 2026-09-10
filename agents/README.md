# Agents — RAG Multi-Agent Knowledge Assistant

All agents are **implemented in M1.4** as plain Python classes. No LLM, no agent framework.

> Run tests: `python -m pytest test_m14_retrieval.py -v`  
> Entry point: `agents/orchestrator.py` → `Orchestrator.handle(query, session_id)`

---

## Agent Summary

| Agent | File | Role | Calls |
|---|---|---|---|
| **Orchestrator** | `orchestrator.py` | Coordinates full pipeline | All agents below |
| **QueryUnderstandingAgent** | `query_understanding.py` | Classify intent, detect domain, flag ambiguity | — |
| **RetrievalAgent** | `retrieval_agent.py` | Embed query, search FAISS, return ranked hits | `SemanticRetriever` |
| **ResponseGenerationAgent** | `response_generation.py` | Extract best answer span + citations | — |
| **ClarificationAgent** | `clarification.py` | Generate clarifying question | — |
| **ConversationMemoryAgent** | `memory.py` | Store/retrieve last-N turns per session | — |

---

## Orchestrator Pipeline

```
User query + session_id
    │
    ▼
① ConversationMemoryAgent.get_recent_context(session_id)
    │  → context string (last turn)
    ▼
② QueryUnderstandingAgent.analyze(query)
    │  → ParsedQuery { intent, domain, is_ambiguous, normalized_query }
    │
    ├── intent == "unavailable"
    │       → AgentResponse { status="unavailable", answer="not in knowledge base" }
    │
    ├── is_ambiguous == True
    │       → ClarificationAgent.generate_question(parsed)
    │       → AgentResponse { status="clarification_needed" }
    │
    └── clear intent
            │
            ▼
        ③ RetrievalAgent.retrieve(normalized_query, top_k, domain)
            │  → RetrievalHit[] ranked by L2 distance
            │
            ├── no hits OR confidence < 0.45
            │       → ClarificationAgent.generate_question(parsed, hits)
            │       → AgentResponse { status="clarification_needed" }
            │
            └── sufficient evidence
                    │
                    ▼
                ④ ResponseGenerationAgent.generate(query, hits, intent, confidence)
                    │  → AgentResponse { answer, citations[], status="answered" }
                    │
                    ▼
                ⑤ ConversationMemoryAgent.add_turn(session_id, query, answer)
                    │
                    ▼
                AgentResponse returned to caller
```

---

## QueryUnderstandingAgent

**Input:** raw query `str`  
**Output:** `ParsedQuery`

```python
@dataclass
class ParsedQuery:
    raw_query: str
    normalized_query: str
    intent: str          # "factual" | "procedural" | "comparative" | "unavailable"
    domain: str | None   # "Software Engineering" | "Hospital Administration" | None
    is_ambiguous: bool
```

**Classification (rule-based, no LLM):**

| Intent | Keywords / patterns matched |
|---|---|
| `unavailable` | revenue, CEO, fiscal year, Nobel Prize, invented … year, stock price |
| `comparative` | differ, difference, compare, versus, vs, contrast, between |
| `procedural` | how do/to/should/can, steps to, procedure, protocol |
| `factual` | default — everything else |

---

## RetrievalAgent

**Input:** `normalized_query`, `top_k`, optional `domain`  
**Output:** `RetrievalHit[]`

```python
@dataclass
class RetrievalHit:
    rank: int
    score: float          # L2 distance — lower = better match
    text: str
    document_id: str
    filename: str
    chunk_id: str
    metadata: dict
```

Domain filtering: if `domain` is detected, results from other domains are removed.
Falls back to all results if filtering yields zero hits.

---

## ResponseGenerationAgent

**Input:** `query`, `hits: RetrievalHit[]`, `intent`, `confidence`, `domain`  
**Output:** `AgentResponse`

Answer construction:

| Intent | Format |
|---|---|
| `factual` | Best-matching sentence excerpt from top hit |
| `procedural` | `"Procedure from {filename}: {excerpt}"` |
| `comparative` | `"{excerpt1} Additionally: {excerpt2}"` |
| No hits | `"The requested information is not available in the knowledge base."` |

All answers append `[source: {filenames}]`. Up to 3 citations attached.

**Excerpt selection:** Splits chunk into sentences/paragraphs; scores each unit by keyword overlap with query; returns highest-scoring unit (≤400 chars).

---

## ClarificationAgent

**Input:** `ParsedQuery`, optional `RetrievalHit[]`  
**Output:** Clarifying question string

| Trigger condition | Generated question |
|---|---|
| No domain detected | "Could you specify whether your question relates to Software Engineering or Hospital Administration?" |
| Query is ambiguous (too short / vague pronouns) | "Your question seems broad. Could you provide more detail about what you need regarding {domain}?" |
| No hits found | "I could not find relevant documents. Could you rephrase or add more context?" |
| Low confidence | "I found only weakly related information. Could you clarify the specific topic?" |

---

## ConversationMemoryAgent

**Input / Output:** per-`session_id` turn history (in-memory, max 5 turns)

```python
agent.add_turn(session_id, query, answer)          # store a turn
agent.get_recent_turns(session_id, limit=3)        # retrieve last-N turns
agent.get_recent_context(session_id)               # → "Previous question: … Previous answer: …"
agent.clear(session_id)                            # wipe session
```

> Memory is in-process only — not persisted across server restarts. Redis / SQLite persistence is planned for M2.

---

## AgentResponse (unified return type)

```python
@dataclass
class AgentResponse:
    answer: str
    citations: list[Citation]
    status: str                    # "answered" | "unavailable" | "clarification_needed"
    intent: str
    confidence: float              # 1 / (1 + top1_L2_score)
    retrieval_hits: list[RetrievalHit]
    domain: str | None
    clarification_question: str | None
```

---

## Module Layout

```
agents/
├── __init__.py              # Exports: Orchestrator, all agents, all models
├── orchestrator.py          # Orchestrator.handle(query, session_id) → AgentResponse
├── query_understanding.py   # QueryUnderstandingAgent.analyze(query) → ParsedQuery
├── retrieval_agent.py       # RetrievalAgent.retrieve(query, top_k, domain) → RetrievalHit[]
├── response_generation.py   # ResponseGenerationAgent.generate(…) → AgentResponse
├── clarification.py         # ClarificationAgent.generate_question(…) → str
├── memory.py                # ConversationMemoryAgent (in-memory, max-5-turn deque)
└── models.py                # Shared dataclasses: ParsedQuery, RetrievalHit,
                             #   Citation, AgentResponse, ConversationTurn
```

---

## Tests

```bash
python -m pytest test_m14_retrieval.py -v
```

| Test class | Coverage |
|---|---|
| `TestQueryUnderstanding` | factual, procedural, comparative, unavailable intent |
| `TestOrchestrator` | factual query, procedural query, comparative query, unavailable query, low relevance → clarification, citation preservation, hit field completeness |

11 tests — all passing.
