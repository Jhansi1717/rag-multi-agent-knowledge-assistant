# Multi-Agent Orchestration

The orchestrator coordinates specialised agents between the API and the retrieval core.
**No agent runtime code exists in M1** — this document defines the target architecture
that wraps **IMPLEMENTED IN M1** retrieval components.

---

## 1. Orchestrator Pipeline

```mermaid
flowchart TD
    IN["AgentMessage<br/>(user query + session_id)"] --> API["API Layer<br/>IMPLEMENTED IN M1"]
    API --> ORCH["Multi-Agent Orchestrator<br/>FUTURE MILESTONES"]

    ORCH --> MEM["Conversation Memory Agent<br/>FUTURE MILESTONES"]
    MEM -->|"ConversationTurn history"| ORCH

    ORCH --> QU["Query Understanding Agent<br/>FUTURE MILESTONES"]
    QU -->|"ParsedQuery<br/>intent · domain · entities"| BR{"Route"}

    BR -->|"is_ambiguous = true"| CLAR["Clarification Agent<br/>FUTURE MILESTONES"]
    CLAR -->|"clarifying question"| OUT1["AgentMessage → UI"]

    BR -->|"is_ambiguous = false"| RA["Retrieval Agent<br/>FUTURE MILESTONES"]
    RA --> SR["SemanticRetriever<br/>IMPLEMENTED IN M1"]
    SR --> VDB[("Vector DB<br/>IMPLEMENTED IN M1")]
    VDB --> RA
    RA -->|"RetrievalResult[]"| SUFF{"Evidence<br/>sufficient?"}

    SUFF -->|"no"| UNAV["Response<br/>information unavailable"]
    SUFF -->|"yes"| RG["Response Generation Agent<br/>FUTURE MILESTONES"]
    RG --> CIT["Attach Citations<br/>FUTURE MILESTONES"]
    RG --> CONF["Attach Confidence<br/>FUTURE MILESTONES"]
    CIT --> RESP["Response"]
    CONF --> RESP
    UNAV --> RESP
    RESP --> MEM2["Memory: store ConversationTurn<br/>FUTURE MILESTONES"]
    MEM2 --> OUT2["AgentMessage → UI"]

    style API fill:#d4edda,stroke:#155724
    style SR fill:#d4edda,stroke:#155724
    style VDB fill:#d4edda,stroke:#155724
    style ORCH fill:#fff3cd,stroke:#856404
    style MEM fill:#fff3cd,stroke:#856404
    style MEM2 fill:#fff3cd,stroke:#856404
    style QU fill:#fff3cd,stroke:#856404
    style CLAR fill:#fff3cd,stroke:#856404
    style RA fill:#fff3cd,stroke:#856404
    style RG fill:#fff3cd,stroke:#856404
    style CIT fill:#fff3cd,stroke:#856404
    style CONF fill:#fff3cd,stroke:#856404
    style UNAV fill:#fff3cd,stroke:#856404
    style RESP fill:#fff3cd,stroke:#856404
```

---

## 2. Agent Responsibilities

| Agent | Input | Output | Status |
|---|---|---|---|
| **Multi-Agent Orchestrator** | `AgentMessage` | `Response` or clarification `AgentMessage` | **FUTURE MILESTONES** |
| **Conversation Memory Agent** | session history | enriched context, stored `ConversationTurn` | **FUTURE MILESTONES** |
| **Query Understanding Agent** | raw query + context | intent, domain, entities, `is_ambiguous` | **FUTURE MILESTONES** |
| **Clarification Agent** | ambiguity signal | clarifying question string | **FUTURE MILESTONES** |
| **Retrieval Agent** | parsed query, `top_k` | `RetrievalResult[]` | **FUTURE MILESTONES** (wraps **IMPLEMENTED IN M1** `SemanticRetriever`) |
| **Response Generation Agent** | query + evidence | `Response` with answer | **FUTURE MILESTONES** |

---

## 3. Routing Rules (Planned)

```mermaid
stateDiagram-v2
    [*] --> LoadMemory: session_id present
    LoadMemory --> UnderstandQuery
    UnderstandQuery --> Clarify: is_ambiguous
    UnderstandQuery --> Retrieve: not ambiguous
    Clarify --> [*]: return question
    Retrieve --> CheckEvidence
    CheckEvidence --> Unavailable: insufficient evidence
    CheckEvidence --> Generate: sufficient evidence
    Unavailable --> [*]: return unavailable Response
    Generate --> AttachMeta: citations + confidence
    AttachMeta --> SaveTurn
    SaveTurn --> [*]: return Response
```

**Status:** **FUTURE MILESTONES**

---

## 4. Query Understanding — Intent Taxonomy

| Intent | Example | Used in M1 eval |
|---|---|---|
| Factual | "How many leave days?" | Yes |
| Procedural | "How do I apply for leave?" | Yes |
| Comparative | "Difference between Annual and Sick carry-forward?" | Yes |
| Unavailable | "What was revenue in 2025?" | Yes (negative test) |

Intent classification logic — **FUTURE MILESTONES**; eval harness labels — **IMPLEMENTED IN M1**.

---

## 5. Evidence Sufficiency (Future)

The orchestrator will reject weak matches before generation:

| Signal | Mechanism | Status |
|---|---|---|
| Top-1 L2 distance | Threshold gate | **FUTURE MILESTONES** |
| Domain mismatch | Query Understanding output vs chunk metadata | **FUTURE MILESTONES** |
| Empty index | Zero vectors | **IMPLEMENTED IN M1** (returns `[]`) |

---

## 6. Citations and Confidence

```mermaid
flowchart LR
    RR["RetrievalResult[]"] --> RG["Response Generation Agent"]
    RG --> ANS["answer text"]
    RR --> CIT["citations[]<br/>chunk_id · document_name"]
    RR --> CONF["confidence<br/>derived from similarity_score"]
    ANS --> RESP["Response"]
    CIT --> RESP
    CONF --> RESP
```

| Field | Producer | Status |
|---|---|---|
| `citations[]` | Response Generation Agent | **FUTURE MILESTONES** |
| `confidence` | Orchestrator / RG from retrieval scores | **FUTURE MILESTONES** |
| `similarity_score` (raw L2) | `VectorStore.search()` | **IMPLEMENTED IN M1** |

---

## 7. Planned Module Layout

| File (planned) | Role | Status |
|---|---|---|
| `agents/orchestrator.py` | State machine / LangGraph router | **FUTURE MILESTONES** |
| `agents/query_understanding.py` | Intent + domain + ambiguity | **FUTURE MILESTONES** |
| `agents/retrieval_agent.py` | Wraps `SemanticRetriever` | **FUTURE MILESTONES** |
| `agents/response_generation.py` | LLM grounded answer | **FUTURE MILESTONES** |
| `agents/clarification.py` | Clarifying questions | **FUTURE MILESTONES** |
| `agents/memory.py` | Session history | **FUTURE MILESTONES** |

Design reference: [`../agents/README.md`](../agents/README.md)
