"""evaluate_m2.py — End-to-end M2 pipeline evaluation.

Runs every query in data/evaluation/queries.json through Orchestrator.handle()
and reports:
  - Intent accuracy (predicted vs labelled)
  - Status correctness (answered / unavailable)
  - Hit@1/3/5 (same metric as M1 but via the full agent pipeline)
  - Per-domain and per-intent breakdowns

Usage:
    python evaluation/evaluate_m2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from root or evaluation/ directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.orchestrator import Orchestrator
from vector_store.store import VectorStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QUERIES_PATH = ROOT / "data" / "evaluation" / "queries.json"
INDEX_PATH   = ROOT / "data" / "evaluation" / "index.faiss"
META_PATH    = ROOT / "data" / "evaluation" / "metadata.json"

UNAVAILABLE_INTENTS = {"unavailable-information", "unavailable"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus() -> VectorStore:
    vs = VectorStore(
        index_path=str(INDEX_PATH),
        meta_path=str(META_PATH),
    )
    vs.load()
    return vs


def hit_at_k(expected_file: str, hits, k: int) -> bool:
    """Return True if expected_file appears in the top-k hit filenames."""
    if not expected_file or expected_file.upper() == "NONE":
        return False
    top_k_files = {h.filename for h in hits[:k]}
    return expected_file in top_k_files


def normalise_intent(raw: str) -> str:
    mapping = {
        "unavailable-information": "unavailable",
        "unavailable_information": "unavailable",
    }
    return mapping.get(raw.lower(), raw.lower())


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run():
    # --- Verify corpus exists ---
    if not QUERIES_PATH.exists():
        sys.exit(
            f"ERROR: {QUERIES_PATH} not found. "
            "Run generate_evaluation_corpus.py and index_evaluation_corpus.py first."
        )
    if not INDEX_PATH.exists():
        sys.exit(
            f"ERROR: {INDEX_PATH} not found. Run index_evaluation_corpus.py first."
        )

    query_payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    # The repository corpus stores metadata alongside the query list.
    queries = (
        query_payload.get("queries", [])
        if isinstance(query_payload, dict)
        else query_payload
    )
    vs = load_corpus()
    orch = Orchestrator(vs)

    print(f"\n{'='*70}")
    print("M2 END-TO-END PIPELINE EVALUATION")
    print(f"Corpus : {INDEX_PATH}")
    print(f"Queries: {len(queries)} total")
    print(f"{'='*70}\n")

    results = []

    for q in queries:
        qid             = q["query_id"]
        query_text      = q["query"]
        domain_label    = q.get("domain", "")
        intent_label    = normalise_intent(
            q.get("query_type") or q.get("category", "")
        )
        expected_source = (
            q.get("expected_source")
            or q.get("expected_document")
            or "NONE"
        )
        is_unavailable  = expected_source.upper() == "NONE"

        response = orch.handle(query_text, session_id=f"eval-{qid}")
        hits = response.retrieval_hits

        predicted_intent = normalise_intent(response.intent)
        intent_correct = predicted_intent == intent_label

        h1 = hit_at_k(expected_source, hits, 1) if not is_unavailable else None
        h3 = hit_at_k(expected_source, hits, 3) if not is_unavailable else None
        h5 = hit_at_k(expected_source, hits, 5) if not is_unavailable else None

        status_correct = (
            (is_unavailable and response.status == "unavailable")
            or (not is_unavailable and response.status in ("answered", "clarification_needed"))
        )

        results.append({
            "qid": qid,
            "domain": domain_label,
            "intent_label": intent_label,
            "predicted_intent": predicted_intent,
            "intent_correct": intent_correct,
            "expected_source": expected_source,
            "top1_file": hits[0].filename if hits else "—",
            "status": response.status,
            "status_correct": status_correct,
            "h1": h1,
            "h3": h3,
            "h5": h5,
            "confidence": round(response.confidence, 4),
            "answer_preview": response.answer[:80],
        })

    # --- Metrics ---
    scorable = [r for r in results if r["h1"] is not None]
    unavailable_qs = [r for r in results if r["h1"] is None]

    intent_acc = sum(r["intent_correct"] for r in results) / len(results) * 100
    status_acc = sum(r["status_correct"] for r in results) / len(results) * 100

    def pct(hits_list, key):
        if not hits_list:
            return "n/a"
        return f"{sum(1 for r in hits_list if r[key]) / len(hits_list) * 100:.1f}%"

    # --- Print results table ---
    print(f"{'ID':<10} {'Domain':<24} {'I-Label':<12} {'I-Pred':<12} {'OK?':<5} {'Status':<22} {'H@1':<5} {'H@3':<5} {'H@5':<5}")
    print("-" * 100)
    for r in results:
        h1s = "OK" if r["h1"] else ("NA" if r["h1"] is None else "NO")
        h3s = "OK" if r["h3"] else ("NA" if r["h3"] is None else "NO")
        h5s = "OK" if r["h5"] else ("NA" if r["h5"] is None else "NO")
        ok  = "OK" if r["intent_correct"] else "NO"
        print(
            f"{r['qid']:<10} {r['domain']:<24} {r['intent_label']:<12} "
            f"{r['predicted_intent']:<12} {ok:<5} {r['status']:<22} {h1s:<5} {h3s:<5} {h5s:<5}"
        )

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total queries        : {len(results)}")
    print(f"Scorable (in-KB)     : {len(scorable)}")
    print(f"Unavailable          : {len(unavailable_qs)}")
    print()
    print(f"Intent accuracy      : {intent_acc:.1f}%  ({sum(r['intent_correct'] for r in results)}/{len(results)})")
    print(f"Status accuracy      : {status_acc:.1f}%  ({sum(r['status_correct'] for r in results)}/{len(results)})")
    print()
    print(f"Hit@1 (scorable)     : {pct(scorable, 'h1')}")
    print(f"Hit@3 (scorable)     : {pct(scorable, 'h3')}")
    print(f"Hit@5 (scorable)     : {pct(scorable, 'h5')}")

    # --- Domain breakdown ---
    print(f"\nBy domain:")
    for dom in sorted({r["domain"] for r in scorable}):
        dom_hits = [r for r in scorable if r["domain"] == dom]
        print(f"  {dom:<26} Hit@1={pct(dom_hits,'h1')}  Hit@3={pct(dom_hits,'h3')}  n={len(dom_hits)}")

    # --- Intent breakdown ---
    print(f"\nBy intent:")
    for intent in sorted({r["intent_label"] for r in scorable}):
        i_hits = [r for r in scorable if r["intent_label"] == intent]
        print(f"  {intent:<14} Hit@1={pct(i_hits,'h1')}  Hit@3={pct(i_hits,'h3')}  n={len(i_hits)}")

    # --- Save JSON results ---
    out_path = ROOT / "evaluation" / "results_m2.json"
    out_path.write_text(
        json.dumps({"summary": {
            "total": len(results),
            "scorable": len(scorable),
            "unavailable": len(unavailable_qs),
            "intent_accuracy_pct": round(intent_acc, 1),
            "status_accuracy_pct": round(status_acc, 1),
            "hit_at_1_pct": round(sum(1 for r in scorable if r["h1"]) / len(scorable) * 100, 1) if scorable else 0,
            "hit_at_3_pct": round(sum(1 for r in scorable if r["h3"]) / len(scorable) * 100, 1) if scorable else 0,
            "hit_at_5_pct": round(sum(1 for r in scorable if r["h5"]) / len(scorable) * 100, 1) if scorable else 0,
        }, "queries": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults written to: {out_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
