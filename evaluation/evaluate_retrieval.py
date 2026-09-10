"""M1 retrieval evaluation against data/evaluation/queries.json."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.retrieval_agent import RetrievalAgent
from retrieval.retriever import SemanticRetriever
from vector_store.store import VectorStore

QUERIES_PATH = os.path.join(ROOT, "data", "evaluation", "queries.json")
INDEX_PATH = os.path.join(ROOT, "data", "evaluation", "index.faiss")
META_PATH = os.path.join(ROOT, "data", "evaluation", "metadata.json")
RESULTS_JSON = os.path.join(ROOT, "evaluation", "results.json")
RESULTS_MD = os.path.join(ROOT, "evaluation", "results.md")

TOP_KS = (1, 3, 5)


def _filename(hit) -> str:
    return os.path.basename(hit.filename or "")


def _keyword_hits(text: str, keywords: List[str]) -> List[str]:
    lower = text.lower()
    found = []
    for kw in keywords:
        if kw.lower() in lower:
            found.append(kw)
    return found


def _first_rank_for_document(hits: list, expected: str) -> Optional[int]:
    expected_l = expected.lower()
    for hit in hits:
        if _filename(hit).lower() == expected_l:
            return hit.rank
    return None


def _serialize_hits(hits: list, k: int) -> List[Dict[str, Any]]:
    out = []
    for hit in hits[:k]:
        out.append({
            "rank": hit.rank,
            "score": hit.score,
            "filename": _filename(hit),
            "document_id": hit.document_id,
            "chunk_id": hit.chunk_id,
            "text_preview": (hit.text or "")[:240].replace("\n", " "),
        })
    return out


def _mean(values: List[int]) -> float:
    if not values:
        return 0.0
    return round(100.0 * sum(values) / len(values), 2)


def evaluate_query(agent: RetrievalAgent, spec: dict) -> dict:
    query = spec["query"]
    expected_doc = spec.get("expected_document") or "NONE"
    keywords = spec.get("expected_keywords") or []
    unavailable = expected_doc.upper() == "NONE" or spec.get("category") == "unavailable-information"

    hits = agent.retrieve(query, top_k=5)
    top1 = _filename(hits[0]) if hits else None
    rank = None if unavailable else _first_rank_for_document(hits, expected_doc)

    combined_text = " ".join(h.text or "" for h in hits)
    found_kw = _keyword_hits(combined_text, keywords) if keywords else []

    hit_at = {
        "hit@1": 0 if unavailable else int(rank is not None and rank <= 1),
        "hit@3": 0 if unavailable else int(rank is not None and rank <= 3),
        "hit@5": 0 if unavailable else int(rank is not None and rank <= 5),
    }

    top1_score = hits[0].score if hits else None
    low_relevance = bool(hits) and (unavailable or rank is None or (top1_score is not None and top1_score > 1.2))

    failure_reason = None
    if unavailable:
        failure_reason = (
            "Unavailable-information query: FAISS still returned a nearest neighbor "
            f"({top1}). No relevant document exists in the KB."
        )
    elif rank is None:
        failure_reason = (
            f"Expected {expected_doc} not found in Top-5. Top-1 was {top1}."
        )
    elif rank > 1:
        failure_reason = (
            f"Expected {expected_doc} ranked {rank}, so Hit@1 failed (Top-1={top1})."
        )

    failed = unavailable or rank is None or rank > 1

    return {
        "query_id": spec["query_id"],
        "domain": spec["domain"],
        "category": spec["category"],
        "query": query,
        "expected_document": expected_doc,
        "expected_keywords": keywords,
        "unavailable": unavailable,
        "top1": top1,
        "top1_score": top1_score,
        "relevant_rank": rank,
        "keywords_found_in_top5": found_kw,
        "keyword_coverage": (
            round(len(found_kw) / len(keywords), 2) if keywords else None
        ),
        "hit@1": hit_at["hit@1"],
        "hit@3": hit_at["hit@3"],
        "hit@5": hit_at["hit@5"],
        "failed": failed,
        "low_relevance": low_relevance,
        "failure_reason": failure_reason if failed or unavailable else None,
        "top1_results": _serialize_hits(hits, 1),
        "top3_results": _serialize_hits(hits, 3),
        "top5_results": _serialize_hits(hits, 5),
    }


def _group_metrics(rows: List[dict], key: str) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = {}
    for name, items in groups.items():
        scored = [r for r in items if not r["unavailable"]]
        out[name] = {
            "total_queries": len(items),
            "scored_queries": len(scored),
            "unavailable_queries": len(items) - len(scored),
            "hit@1": _mean([r["hit@1"] for r in scored]) if scored else None,
            "hit@3": _mean([r["hit@3"] for r in scored]) if scored else None,
            "hit@5": _mean([r["hit@5"] for r in scored]) if scored else None,
        }
    return out


def build_report(rows: List[dict]) -> dict:
    scored = [r for r in rows if not r["unavailable"]]
    failed = [r for r in rows if r["failed"]]
    low_rel = [r for r in rows if r["low_relevance"]]

    return {
        "index": INDEX_PATH,
        "queries_path": QUERIES_PATH,
        "total_queries": len(rows),
        "scored_queries_excluding_unavailable": len(scored),
        "unavailable_queries": len(rows) - len(scored),
        "metrics_excluding_unavailable": {
            "hit@1": _mean([r["hit@1"] for r in scored]),
            "hit@3": _mean([r["hit@3"] for r in scored]),
            "hit@5": _mean([r["hit@5"] for r in scored]),
        },
        "metrics_including_unavailable": {
            "hit@1": _mean([r["hit@1"] for r in rows]),
            "hit@3": _mean([r["hit@3"] for r in rows]),
            "hit@5": _mean([r["hit@5"] for r in rows]),
        },
        "domain_wise": _group_metrics(rows, "domain"),
        "category_wise": _group_metrics(rows, "category"),
        "failed_queries": [
            {
                "query_id": r["query_id"],
                "domain": r["domain"],
                "category": r["category"],
                "query": r["query"],
                "expected_document": r["expected_document"],
                "top1": r["top1"],
                "top1_score": r["top1_score"],
                "relevant_rank": r["relevant_rank"],
                "reason": r["failure_reason"],
            }
            for r in failed
        ],
        "low_relevance_examples": [
            {
                "query_id": r["query_id"],
                "query": r["query"],
                "top1": r["top1"],
                "top1_score": r["top1_score"],
                "expected_document": r["expected_document"],
            }
            for r in low_rel[:8]
        ],
        "queries": rows,
    }


def write_markdown(report: dict) -> str:
    m = report["metrics_excluding_unavailable"]
    m_all = report["metrics_including_unavailable"]
    lines = [
        "# M1 Retrieval Evaluation Results",
        "",
        "Metrics computed from live FAISS retrieval against `data/evaluation/queries.json`.",
        "Hit@k = expected_document appears in the top-k filenames.",
        "Unavailable-information queries have `expected_document=NONE` and are excluded from the primary Hit@ averages because no relevant document exists; FAISS still returns a nearest neighbor.",
        "",
        "## Overall",
        "",
        f"- Total queries: **{report['total_queries']}**",
        f"- Scored (excl. unavailable): **{report['scored_queries_excluding_unavailable']}**",
        f"- Unavailable-information: **{report['unavailable_queries']}**",
        "",
        "| Metric | Excl. unavailable | Incl. unavailable |",
        "|---|---|---|",
        f"| Hit@1 | {m['hit@1']}% | {m_all['hit@1']}% |",
        f"| Hit@3 | {m['hit@3']}% | {m_all['hit@3']}% |",
        f"| Hit@5 | {m['hit@5']}% | {m_all['hit@5']}% |",
        "",
        "## Domain-wise (excl. unavailable)",
        "",
        "| Domain | Queries | Scored | Hit@1 | Hit@3 | Hit@5 |",
        "|---|---|---|---|---|---|",
    ]
    for domain, stats in report["domain_wise"].items():
        lines.append(
            f"| {domain} | {stats['total_queries']} | {stats['scored_queries']} | "
            f"{stats['hit@1']}% | {stats['hit@3']}% | {stats['hit@5']}% |"
        )

    lines += [
        "",
        "## Category-wise (excl. unavailable)",
        "",
        "| Category | Queries | Scored | Hit@1 | Hit@3 | Hit@5 |",
        "|---|---|---|---|---|---|",
    ]
    for cat, stats in report["category_wise"].items():
        h1 = stats["hit@1"] if stats["hit@1"] is not None else "n/a"
        h3 = stats["hit@3"] if stats["hit@3"] is not None else "n/a"
        h5 = stats["hit@5"] if stats["hit@5"] is not None else "n/a"
        lines.append(
            f"| {cat} | {stats['total_queries']} | {stats['scored_queries']} | "
            f"{h1} | {h3} | {h5} |"
        )

    lines += ["", "## Per-query results", "",
              "| ID | Domain | Category | Expected | Top-1 | Rank | Hit@1 | Hit@3 | Hit@5 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in report["queries"]:
        rank = r["relevant_rank"] if r["relevant_rank"] is not None else "N/A"
        lines.append(
            f"| {r['query_id']} | {r['domain']} | {r['category']} | "
            f"{r['expected_document']} | {r['top1']} | {rank} | "
            f"{r['hit@1']} | {r['hit@3']} | {r['hit@5']} |"
        )

    lines += ["", "## Failed queries", ""]
    failed = report["failed_queries"]
    if not failed:
        lines.append("None.")
    else:
        for f in failed:
            lines.append(f"### {f['query_id']} — {f['category']}")
            lines.append(f"- Query: {f['query']}")
            lines.append(f"- Expected: `{f['expected_document']}`")
            lines.append(f"- Top-1: `{f['top1']}` (L2={f['top1_score']})")
            lines.append(f"- Rank of expected: {f['relevant_rank']}")
            lines.append(f"- Cause: {f['reason']}")
            lines.append("")

    lines += ["", "## Low-relevance examples", ""]
    if not report["low_relevance_examples"]:
        lines.append("None flagged.")
    else:
        for ex in report["low_relevance_examples"]:
            lines.append(
                f"- `{ex['query_id']}`: Top-1 `{ex['top1']}` L2={ex['top1_score']} "
                f"(expected `{ex['expected_document']}`) — {ex['query']}"
            )

    # Inspect first 3 failed cases in depth from full query rows
    inspect = [r for r in report["queries"] if r["failed"]][:3]
    lines += ["", "## Inspected failed cases (up to 3)", ""]
    if not inspect:
        lines.append("No failures to inspect.")
    else:
        for r in inspect:
            lines.append(f"### {r['query_id']}")
            lines.append(f"- Query: {r['query']}")
            lines.append(f"- Expected document: `{r['expected_document']}`")
            lines.append(f"- Keywords found in Top-5: {r['keywords_found_in_top5']}")
            lines.append("- Top-5:")
            for h in r["top5_results"]:
                lines.append(
                    f"  - #{h['rank']} `{h['filename']}` L2={round(h['score'], 4)} "
                    f"— {h['text_preview'][:160]}"
                )
            lines.append("")

    lines += [
        "## Likely failure causes",
        "",
        "- FAISS `IndexFlatL2` always returns a nearest neighbor, so unavailable-information queries cannot score a document hit.",
        "- One chunk per document (short corpus) can mix unrelated sections into the same embedding, diluting comparative/factual signals.",
        "- Characteristically similar hospital/software procedure language can cross-rank nearby documents.",
        "- L2 distance is not calibrated as relevance; a high score can still be Top-1.",
        "",
        "## M2 improvement ideas",
        "",
        "- Add a similarity / confidence threshold and return `unavailable` when Top-1 is weak.",
        "- Use smaller, section-aware chunks so sprint vs git vs coding-standard topics do not share one vector.",
        "- Hybrid lexical + dense retrieval (BM25 + embeddings) for keyword-heavy factual queries.",
        "- Cross-encoder re-ranking of Top-10 candidates.",
        "- Explicit unavailable-intent routing (already in QueryUnderstandingAgent) before generation.",
        "- Domain filter at retrieval time using query-understanding domain labels.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Evaluation index missing ({INDEX_PATH}). Run index_evaluation_corpus.py first."
        )
    with open(QUERIES_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    queries = payload["queries"]

    store = VectorStore(index_path=INDEX_PATH, meta_path=META_PATH)
    store.load()
    agent = RetrievalAgent(SemanticRetriever(store))

    rows = [evaluate_query(agent, spec) for spec in queries]
    report = build_report(rows)

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(write_markdown(report))

    m = report["metrics_excluding_unavailable"]
    print("Wrote", RESULTS_JSON, "and", RESULTS_MD)
    print("Total:", report["total_queries"], "scored:", report["scored_queries_excluding_unavailable"])
    print(f"Hit@1={m['hit@1']}% Hit@3={m['hit@3']}% Hit@5={m['hit@5']}%")
    print("Failed:", len(report["failed_queries"]))


if __name__ == "__main__":
    main()
