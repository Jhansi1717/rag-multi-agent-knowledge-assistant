# Architecture Decisions — RAG Multi-Agent Knowledge Assistant

> This file is superseded by [`architecture-decisions.md`](architecture-decisions.md) which contains the complete, up-to-date decision log.
>
> Kept for historical reference. Do not edit — edit `architecture-decisions.md` instead.

---

See **[`architecture-decisions.md`](architecture-decisions.md)** for all current architectural decisions with accurate M1 implementation status.

Key decisions covered there:

| Decision | Summary |
|---|---|
| AD-01 | Thin vertical slice (ingestion → retrieval → agents) before LLM |
| AD-02 | Separate ingestion and query lifecycles |
| AD-03 | Unified extraction interface for PDF/DOCX/TXT/CSV |
| AD-04 | Token-aware paragraph-preserving chunking (650 tokens, 75 overlap) |
| AD-05 | Local embedding model (`all-MiniLM-L6-v2`, CPU) |
| AD-06 | FAISS IndexFlatL2 + JSON metadata |
| AD-07 | FastAPI HTTP layer (3 endpoints) |
| AD-08 | Retrieval validated before generation — Hit@1/3/5 = 100% |
| AD-09 | Deterministic multi-agent pipeline (no LangGraph) |
| AD-10 | Browser-native voice boundary |
| AD-11 | Extractive responses with inline citations |
| AD-12 | Honest status labelling — 🟢 only if tested and verified |