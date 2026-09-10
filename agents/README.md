# Agents — Design Documentation

This directory contains M1.4 retrieval agents. Agents wrap existing
`SemanticRetriever` / `VectorStore` code with a deterministic orchestrator.

**Status: IMPLEMENTED IN M1.4** (no LLM, no agent framework).

---

## Designed Agents

### Query Understanding Agent
**File**: `agents/query_understanding.py`

Responsibilities:
- Classify query intent: Factual, Procedural, Comparative, Unavailable
- Extract named entities and domain (HR, Software, etc.)
- Detect whether the query is ambiguous or underspecified

Input: raw query string  
Output: `{intent, domain, entities, is_ambiguous}`

---

### Retrieval Agent
**File**: `agents/retrieval_agent.py`

Responsibilities:
- Accept a parsed query from the Query Understanding Agent
- Call `SemanticRetriever.retrieve(query, top_k)` from `retrieval/retriever.py`
- Return ranked chunks with source metadata and similarity scores

Input: query string, top_k  
Output: list of Retrieval Results

---

### Response Generation Agent
**File**: `agents/response_generation.py`

Responsibilities:
- Accept retrieved chunks as context
- Build a minimal extractive answer from retrieved chunks (no LLM in M1)
- Return a natural-language answer with inline citations
- Detect and explicitly state when retrieved context is insufficient

Input: query string, list of Retrieval Results  
Output: `{answer, citations[]}`

---

### Clarification Agent
**File**: `agents/clarification.py`

Responsibilities:
- Detect when query intent cannot be determined with confidence
- Generate a targeted clarifying question to send back to the user

Input: query string, ambiguity signal  
Output: clarifying question string

---

### Conversation Memory Agent
**File**: `agents/memory.py`

Responsibilities:
- Store per-session conversation history (turns, queries, answers)
- Resolve coreferences across turns (e.g., "it" referring to a previously mentioned topic)
- Inject relevant history into retrieval and generation context

Input: current turn, session history  
Output: enriched context for retrieval

---

## Multi-Agent Orchestrator

**File**: `agents/orchestrator.py`

Coordinates the agent pipeline:

```
Query → Query Understanding Agent
      → Ambiguity Check
          ├── Ambiguous → Clarification Agent → User
          └── Clear → Retrieval Agent
                    → Evidence Sufficiency Check
                        ├── Insufficient → "Information unavailable"
                        └── Sufficient → Response Generation Agent → Answer
```
