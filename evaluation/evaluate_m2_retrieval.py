"""Evaluate the upgraded M2 retrieval agent against the evaluation corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
QUERIES_PATH = ROOT / "data" / "evaluation" / "queries.json"
INDEX_PATH = ROOT / "data" / "evaluation" / "index.faiss"
META_PATH = ROOT / "data" / "evaluation" / "metadata.json"
M1_RESULTS_PATH = ROOT / "evaluation" / "results.json"
JSON_OUTPUT_PATH = ROOT / "evaluation" / "m2_retrieval_results.json"
MARKDOWN_OUTPUT_PATH = ROOT / "evaluation" / "m2_retrieval_results.md"

sys.path.insert(0, str(ROOT))

from agents.query_understanding import QueryUnderstandingAgent
from agents.retrieval_agent import RetrievalAgent
from retrieval.retriever import SemanticRetriever
from vector_store.store import VectorStore

TOP_K = 3
MIN_RELEVANCE = 0.0


class RecordingRetriever:
    """Record the existing retriever's raw results without duplicating search."""

    def __init__(self, retriever: SemanticRetriever):
        self.retriever = retriever
        self.last_raw_results: list[dict[str, Any]] = []

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        self.last_raw_results = self.retriever.retrieve(query, top_k=top_k) or []
        return self.last_raw_results


def load_queries() -> list[dict[str, Any]]:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return payload["queries"] if isinstance(payload, dict) else payload


def load_m1_baseline() -> dict[str, Any] | None:
    if not M1_RESULTS_PATH.exists():
        return None
    payload = json.loads(M1_RESULTS_PATH.read_text(encoding="utf-8"))
    metrics = payload.get("metrics_including_unavailable", {})
    if not metrics:
        return None
    return {
        "source": str(M1_RESULTS_PATH.relative_to(ROOT)),
        "hit_at_1": metrics.get("hit@1"),
        "hit_at_3": metrics.get("hit@3"),
        "hit_at_5": metrics.get("hit@5"),
    }


def query_type(item: dict[str, Any]) -> str:
    label = str(item.get("query_type") or item.get("category") or "").lower()
    return "factual" if label in {
        "unavailable",
        "unavailable-information",
        "unavailable_information",
    } else label


def evaluate() -> dict[str, Any]:
    if not QUERIES_PATH.exists() or not INDEX_PATH.exists():
        raise FileNotFoundError("Evaluation queries or FAISS index is missing.")

    store = VectorStore(index_path=str(INDEX_PATH), meta_path=str(META_PATH))
    store.load()
    recorder = RecordingRetriever(SemanticRetriever(store))
    agent = RetrievalAgent(
        recorder,
        top_k=TOP_K,
        min_relevance=MIN_RELEVANCE,
    )
    understanding = QueryUnderstandingAgent()
    rows: list[dict[str, Any]] = []

    for item in load_queries():
        parsed = understanding.analyze(item["query"])
        result = agent.retrieve(
            parsed.normalized_query,
            query_type=parsed.query_type,
            domain=item.get("domain"),
            top_k=TOP_K,
        )
        expected_document = item.get("expected_document", "NONE")
        unavailable = str(expected_document).upper() == "NONE"
        filtered_results = [
            {
                "rank": hit.rank,
                "filename": hit.filename,
                "chunk_id": hit.chunk_id,
                "distance_score": round(hit.distance_score, 6),
                "relevance_score": round(hit.relevance_score, 6),
                "metadata": hit.metadata,
            }
            for hit in result.results
        ]
        expected_hit = (
            any(hit["filename"] == expected_document for hit in filtered_results)
            if not unavailable
            else False
        )
        rows.append(
            {
                "query_id": item.get("query_id"),
                "query": item["query"],
                "domain": item.get("domain"),
                "query_type": parsed.query_type,
                "expected_document": expected_document,
                "top_k_before_filtering": len(recorder.last_raw_results),
                "results_after_filtering": filtered_results,
                "filtered_count": result.filtered_count,
                "retrieval_confidence": round(result.retrieval_confidence, 6),
                "sufficient_evidence": result.sufficient_evidence,
                "no_relevant_information": result.no_relevant_information,
                "expected_document_retrieved": expected_hit,
                "unavailable_query_neighbor_not_valid_evidence": (
                    unavailable and not result.sufficient_evidence
                ),
            }
        )

    available = [row for row in rows if row["expected_document"] != "NONE"]
    unavailable = [row for row in rows if row["expected_document"] == "NONE"]
    correct_hits = sum(row["expected_document_retrieved"] for row in available)
    empty_results = sum(not row["results_after_filtering"] for row in rows)
    low_confidence = sum(
        row["retrieval_confidence"] < 0.45 or row["no_relevant_information"]
        for row in rows
    )

    def grouped_metrics(key: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for value in sorted({row[key] for row in rows}):
            group = [row for row in rows if row[key] == value]
            group_available = [row for row in group if row["expected_document"] != "NONE"]
            group_hits = sum(row["expected_document_retrieved"] for row in group_available)
            output[value] = {
                "total_queries": len(group),
                "available_queries": len(group_available),
                "retrieval_hit_rate": (
                    group_hits / len(group_available) if group_available else None
                ),
                "empty_result_rate": sum(
                    not row["results_after_filtering"] for row in group
                ) / len(group),
                "low_confidence_rate": sum(
                    row["retrieval_confidence"] < 0.45
                    or row["no_relevant_information"]
                    for row in group
                ) / len(group),
            }
        return output

    summary = {
        "total_queries": len(rows),
        "available_queries": len(available),
        "unavailable_queries": len(unavailable),
        "retrieval_hit_rate": correct_hits / len(available) if available else 0.0,
        "empty_result_rate": empty_results / len(rows) if rows else 0.0,
        "low_confidence_rate": low_confidence / len(rows) if rows else 0.0,
        "unavailable_neighbors_rejected_as_evidence": sum(
            row["unavailable_query_neighbor_not_valid_evidence"] for row in unavailable
        ),
        "unavailable_queries_with_sufficient_evidence": sum(
            row["sufficient_evidence"] for row in unavailable
        ),
        "domain_performance": grouped_metrics("domain"),
        "query_type_performance": grouped_metrics("query_type"),
        "m1_baseline": load_m1_baseline(),
    }
    output = {"configuration": {
        "top_k": TOP_K,
        "minimum_relevance": MIN_RELEVANCE,
        "relevance_transform": "1 / (1 + distance_score)",
    }, "summary": summary, "queries": rows}
    JSON_OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT_PATH.write_text(render_markdown(output), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return output


def render_markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        "# M2 Retrieval Results",
        "",
        f"- Top-K: {output['configuration']['top_k']}",
        f"- Minimum relevance: {output['configuration']['minimum_relevance']}",
        "- Relevance: `1 / (1 + distance_score)`; lower FAISS L2 distance is better.",
        "",
        "## Summary",
        "",
        f"- Total queries: {summary['total_queries']}",
        f"- Available queries: {summary['available_queries']}",
        f"- Unavailable-information queries: {summary['unavailable_queries']}",
        f"- Retrieval hit rate: {summary['retrieval_hit_rate']:.1%}",
        f"- Empty-result rate: {summary['empty_result_rate']:.1%}",
        f"- Low-confidence rate: {summary['low_confidence_rate']:.1%}",
        f"- Unavailable neighbors rejected as evidence: "
        f"{summary['unavailable_neighbors_rejected_as_evidence']}/"
        f"{summary['unavailable_queries']}",
        f"- Unavailable queries with sufficient evidence: "
        f"{summary['unavailable_queries_with_sufficient_evidence']}/"
        f"{summary['unavailable_queries']}",
        "",
        "Unavailable queries are expected to have no relevant evidence. FAISS nearest "
        "neighbors are recorded but are not counted as retrieval hits.",
        "",
        "## Domain performance",
        "",
        "| Domain | Total | Available | Hit rate | Empty rate | Low-confidence rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, metrics in summary["domain_performance"].items():
        lines.append(
            f"| {domain} | {metrics['total_queries']} | {metrics['available_queries']} | "
            f"{format_rate(metrics['retrieval_hit_rate'])} | "
            f"{format_rate(metrics['empty_result_rate'])} | "
            f"{format_rate(metrics['low_confidence_rate'])} |"
        )
    lines.extend([
        "",
        "## Query-type performance",
        "",
        "| Query type | Total | Available | Hit rate | Empty rate | Low-confidence rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for query_type, metrics in summary["query_type_performance"].items():
        lines.append(
            f"| {query_type} | {metrics['total_queries']} | {metrics['available_queries']} | "
            f"{format_rate(metrics['retrieval_hit_rate'])} | "
            f"{format_rate(metrics['empty_result_rate'])} | "
            f"{format_rate(metrics['low_confidence_rate'])} |"
        )
    lines.extend(["", "## Query details", "", "| Query | Type | Raw | Filtered | Confidence | Sufficient | No information |", "|---|---|---:|---:|---:|---|---|"])
    for row in output["queries"]:
        lines.append(
            f"| {row['query'].replace('|', '\\|')} | {row['query_type']} | "
            f"{row['top_k_before_filtering']} | {len(row['results_after_filtering'])} | "
            f"{row['retrieval_confidence']:.3f} | "
            f"{'yes' if row['sufficient_evidence'] else 'no'} | "
            f"{'yes' if row['no_relevant_information'] else 'no'} |"
        )
    baseline = summary["m1_baseline"]
    if baseline:
        lines.extend([
            "",
            "## M1 comparison",
            "",
            f"Source: `{baseline['source']}`. Existing M1 metrics including unavailable queries:",
            "",
            f"- Hit@1: {baseline['hit_at_1']}%",
            f"- Hit@3: {baseline['hit_at_3']}%",
            f"- Hit@5: {baseline['hit_at_5']}%",
        ])
    return "\n".join(lines) + "\n"


def format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    evaluate()
