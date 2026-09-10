# RAG Query Flow — Retrieval Pipeline

Query processing retrieves ranked evidence chunks from the vector index. In M1 the
API calls retrieval **directly**; future milestones route through the agent layer.

---

## 1. M1 Query Flow (Implemented)

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI<br/>(IMPLEMENTED IN M1)
    participant SR as SemanticRetriever<br/>(IMPLEMENTED IN M1)
    participant VS as VectorStore<br/>(IMPLEMENTED IN M1)
    participant VDB as FAISS + metadata.json<br/>(IMPLEMENTED IN M1)

    C->>API: POST /retrieve {query, top_k}
    API->>SR: retrieve(query, top_k)
    SR->>VS: search(query, top_k)
    VS->>VS: encode query → Embedding
    VS->>VDB: IndexFlatL2.search()
    VDB-->>VS: distances[], indices[]
    VS->>VDB: metadata lookup per index
    VDB-->>VS: chunk fields
    VS-->>SR: RetrievalResult[]
    SR-->>API: RetrievalResult[]
    API-->>C: {query, results[]}
```

---

## 2. Target Query Flow (Future Multi-Agent)

```mermaid
flowchart TD
    Q["User query (text)"] --> API["API Layer<br/>IMPLEMENTED IN M1"]
    API --> ORCH["Orchestrator<br/>FUTURE MILESTONES"]
    ORCH --> MEM["Conversation Memory Agent<br/>FUTURE MILESTONES"]
    MEM -->|"enriched context"| QU["Query Understanding Agent<br/>FUTURE MILESTONES"]
    QU -->|"ParsedQuery"| DEC{"Ambiguous?"}
    DEC -->|yes| CLAR["Clarification Agent<br/>FUTURE MILESTONES"]
    CLAR -->|"clarifying question"| Q
    DEC -->|no| RA["Retrieval Agent<br/>FUTURE MILESTONES"]
    RA --> SR["SemanticRetriever<br/>IMPLEMENTED IN M1"]
    SR --> VDB[("Vector DB<br/>IMPLEMENTED IN M1")]
    VDB --> RA
    RA --> SUFF{"Evidence<br/>sufficient?"}
    SUFF -->|no| UNAV["Unavailable Response<br/>FUTURE MILESTONES"]
    SUFF -->|yes| RG["Response Generation Agent<br/>FUTURE MILESTONES"]
    RG --> CIT["Citations<br/>FUTURE MILESTONES"]
    RG --> CONF["Confidence score<br/>FUTURE MILESTONES"]
    CIT --> RESP["Response"]
    CONF --> RESP
    UNAV --> RESP
    RESP --> API

    style API fill:#d4edda,stroke:#155724
    style SR fill:#d4edda,stroke:#155724
    style VDB fill:#d4edda,stroke:#155724
    style ORCH fill:#fff3cd,stroke:#856404
    style MEM fill:#fff3cd,stroke:#856404
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

## 3. Retrieval Parameters (M1)

| Parameter | Value | Status |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | **IMPLEMENTED IN M1** |
| Dimensions | 384 | **IMPLEMENTED IN M1** |
| Distance metric | L2 (Euclidean) | **IMPLEMENTED IN M1** |
| Default top_k | 3 | **IMPLEMENTED IN M1** |
| Evaluated top_k | 1, 3, 5 | **IMPLEMENTED IN M1** |
| Similarity threshold (reject low-confidence) | — | **FUTURE MILESTONES** |
| Re-ranking | — | **FUTURE MILESTONES** |

---

## 4. API Contract (M1)

| Endpoint | Method | Input | Output | Status |
|---|---|---|---|---|
| `/health` | GET | — | `{status}` | **IMPLEMENTED IN M1** |
| `/upload` | POST | multipart file | `document_id`, `chunk_count` | **IMPLEMENTED IN M1** |
| `/retrieve` | POST | `Query` | `RetrievalResult[]` | **IMPLEMENTED IN M1** |
| `/chat` (orchestrated answer) | POST | `Query` + session | `Response` | **FUTURE MILESTONES** |

---

## 5. Evaluation Harness

`evaluate_retrieval.py` runs 8 labelled queries across HR and Software domains,
reporting Hit@1, Hit@3, Hit@5. Results recorded in `docs/validation.md`.

**Status:** **IMPLEMENTED IN M1**

---

## 6. Citations and Confidence (Future)

| Signal | Source | Status |
|---|---|---|
| `similarity_score` (L2 distance) | `VectorStore.search()` | **IMPLEMENTED IN M1** (raw distance only) |
| Normalised confidence (0–1) | Derived from distance + threshold | **FUTURE MILESTONES** |
| Inline citations in answer | Response Generation Agent | **FUTURE MILESTONES** |
| "Information unavailable" guard | Orchestrator + RG Agent | **FUTURE MILESTONES** |
